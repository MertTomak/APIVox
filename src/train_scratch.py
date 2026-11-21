"""Train a from-scratch model on train, pick the epoch by dev WER and, with --final, score every final split."""
import argparse
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src import data_io, features, metrics, models

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints"
TRAINED_SEEDS = (13, 42, 2026)
INT16_SCALE = 32767.0
GRAD_CLIP = 5.0
WARMUP_MAX_STEPS = 1000
# The RNN-T joint tensor is [B, T, U + 1, J], so its batches are smaller to fit a 15 GB GPU.
DEFAULT_BATCH_FRAMES = {"ctc": 20000, "aed": 20000, "rnnt": 6000}


def mfcc_cmvn(wave: np.ndarray) -> np.ndarray:
    """MFCC with per-utterance CMVN, so that the ctc-mfcc ablation differs from log-mel only in the features."""
    return features.cmvn(features.mfcc(wave))


FEATURE_FNS = {"logmel": features.logmel, "mfcc": mfcc_cmvn}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=models.HEADS, required=True, help="output head")
    parser.add_argument("--seed", type=int, required=True, help="random seed (13, 42 or 2026 with --final)")
    parser.add_argument("--final", action="store_true",
                        help="also score dev and the final test splits and write predictions and the run file")
    parser.add_argument("--features", choices=tuple(FEATURE_FNS), default="logmel",
                        help="input features (default: logmel; mfcc is the ctc-mfcc ablation)")
    parser.add_argument("--no-aug", action="store_true", help="disable speed, noise and SpecAugment (ablation)")
    parser.add_argument("--max-hours", type=float, default=None, help="debug: use only the first N hours of train")
    parser.add_argument("--epochs", type=int, default=40, help="maximum number of epochs (default: 40)")
    parser.add_argument("--patience", type=int, default=6, help="epochs without dev improvement before stopping")
    parser.add_argument("--lr", type=float, default=1e-3, help="peak AdamW learning rate (default: 1e-3)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--batch-frames", type=int, default=None,
                        help=f"padded 10 ms frames per batch (default per head: {DEFAULT_BATCH_FRAMES})")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader worker processes (default: 2)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda (default: auto)")
    args = parser.parse_args()
    args.batch_frames = args.batch_frames or DEFAULT_BATCH_FRAMES[args.head]
    return args


def model_name(args: argparse.Namespace) -> str:
    """Run name: the head, plus -mfcc or -noaug for the two ablations."""
    if args.features == "mfcc" and args.no_aug:
        raise ValueError("--features mfcc and --no-aug are separate ablations; use one at a time")
    if args.final and args.seed not in TRAINED_SEEDS:
        raise ValueError(f"--final runs use seeds {TRAINED_SEEDS}, got {args.seed}")
    suffix = "-mfcc" if args.features == "mfcc" else "-noaug" if args.no_aug else ""
    return args.head + suffix


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch and ask torch for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # CUDA CTC backward has no deterministic kernel, so torch only warns there.
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(name: str) -> torch.device:
    """Resolve "auto" to CUDA when available, otherwise CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def hardware_name(device: torch.device) -> str:
    """Describe the device the run used."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"cpu: {platform.machine()} {platform.processor()}".strip()


def limit_hours(records: list[dict], max_hours: float | None) -> list[dict]:
    """Keep records in file order until their total duration reaches max_hours."""
    if max_hours is None:
        return records
    kept, total = [], 0.0
    for record in records:
        if total >= max_hours * 3600:
            break
        kept.append(record)
        total += record["duration"]
    return kept


def drop_too_short(records: list[dict], head: str) -> tuple[list[dict], int]:
    """For CTC, drop train clips whose encoder frames at speed 1.1 cannot fit the target alignment."""
    if head != "ctc":
        return records, 0
    kept = []
    for record in records:
        samples = int(record["duration"] * features.SAMPLE_RATE / max(features.SPEED_FACTORS))
        frames = features.frame_count(samples) if samples >= features.WIN_LENGTH else 0
        if math.ceil(frames / 4) >= models.ctc_min_frames(models.encode_text(record["text"])):
            kept.append(record)
    return kept, len(records) - len(kept)


class Utterances(Dataset):
    """Clips with their targets; with augment_seed, items are augmented with an rng seeded by (seed, epoch, index)."""

    def __init__(self, records: list[dict], feature: str, augment_seed: int | None = None) -> None:
        self.feature_fn = FEATURE_FNS[feature]
        self.augment_seed = augment_seed
        self.epoch = 0
        self.targets = [models.encode_text(r["text"]) for r in records]
        self.cache: list[np.ndarray] = []
        self.n_frames: list[int] = []
        for record in records:
            wave = data_io.load_audio(record)
            self.n_frames.append(features.frame_count(len(wave)))
            if augment_seed is None:
                self.cache.append(self.feature_fn(wave).astype(np.float16))
            else:
                self.cache.append(np.round(wave * INT16_SCALE).astype(np.int16))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, list[int]]:
        if self.augment_seed is None:
            feats = self.cache[i].astype(np.float32)
        else:
            rng = np.random.default_rng([self.augment_seed, self.epoch, i])
            wave = features.augment(self.cache[i].astype(np.float32) / INT16_SCALE, rng)
            feats = features.spec_augment(self.feature_fn(wave), rng)
        return torch.from_numpy(feats), self.targets[i]


def collate(items: list[tuple[torch.Tensor, list[int]]]) -> tuple[torch.Tensor, ...]:
    """Zero-pad features to [B, T, D] and targets to [B, S]; return both with their lengths."""
    lengths = torch.tensor([f.shape[0] for f, _ in items])
    target_lengths = torch.tensor([len(t) for _, t in items])
    feats = torch.zeros(len(items), int(lengths.max()), items[0][0].shape[1])
    targets = torch.zeros(len(items), max(1, int(target_lengths.max())), dtype=torch.long)
    for b, (f, t) in enumerate(items):
        feats[b, :f.shape[0]] = f
        targets[b, :len(t)] = torch.tensor(t, dtype=torch.long)
    return feats, lengths, targets, target_lengths


def make_batches(n_frames: list[int], max_frames: int, rng: np.random.Generator | None = None) -> list[list[int]]:
    """Group clips sorted by length so that batch size times the longest clip stays within max_frames."""
    order = sorted(range(len(n_frames)), key=lambda i: n_frames[i])
    batches: list[list[int]] = []
    current: list[int] = []
    for i in order:
        if current and (len(current) + 1) * n_frames[i] > max_frames:
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    if rng is not None:
        batches = [batches[j] for j in rng.permutation(len(batches))]
    return batches


def warmup_cosine(warmup: int, total: int):
    """Learning-rate factor: s / warmup during warmup, then 0.5 * (1 + cos(pi * progress)) down to 0."""
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return factor


def train_epoch(model: torch.nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler,
                device: torch.device) -> float:
    """Run one epoch of training and return the mean loss per clip."""
    model.train()
    total, count = 0.0, 0
    for feats, lengths, targets, target_lengths in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = model.loss(feats.to(device), lengths, targets.to(device), target_lengths)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss {loss.item()}; check the CTC length filter")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(lengths)
        count += len(lengths)
    return total / count


def transcribe(model: torch.nn.Module, dataset: Utterances, device: torch.device, max_frames: int) -> list[str]:
    """Greedy-decode every clip of a dataset and return the hypotheses in dataset order."""
    model.eval()
    hyps = [""] * len(dataset)
    batches = make_batches(dataset.n_frames, max_frames)
    loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate)
    for ids, (feats, lengths, _, _) in zip(batches, loader):
        for i, out in zip(ids, model.decode(feats.to(device), lengths)):
            hyps[i] = models.decode_ids(out)
    return hyps


def fit(args: argparse.Namespace, model: torch.nn.Module, train_set: Utterances, dev_records: list[dict],
        dev_set: Utterances, device: torch.device) -> tuple[dict, int, list[dict], float, int]:
    """Train with early stopping on (dev WER, dev CER); return best weights, best epoch, curve, time and warmup."""
    total_steps = args.epochs * len(make_batches(train_set.n_frames, args.batch_frames))
    warmup = min(WARMUP_MAX_STEPS, max(1, total_steps // 10))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(warmup, total_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_key, best_state, best_epoch, waited, curve = None, {}, 0, 0, []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tick = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_set.epoch = epoch
        batches = make_batches(train_set.n_frames, args.batch_frames, np.random.default_rng([args.seed, epoch]))
        loader = DataLoader(train_set, batch_sampler=batches, collate_fn=collate, num_workers=args.workers)
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, device)
        dev = metrics.evaluate(dev_records, transcribe(model, dev_set, device, args.batch_frames))
        curve.append({"epoch": epoch, "train_loss": round(loss, 4), "dev_wer": dev["wer"], "dev_cer": dev["cer"],
                      "lr": scheduler.get_last_lr()[0], "time_s": round(time.time() - tick, 1)})
        if device.type == "cuda":
            curve[-1]["peak_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in curve[-1].items()),
              flush=True)
        if best_key is None or (dev["wer"], dev["cer"]) < best_key:
            best_key, best_epoch, waited = (dev["wer"], dev["cer"]), epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= args.patience:
                print(f"early stop: no dev improvement for {args.patience} epochs")
                break
    return best_state, best_epoch, curve, time.time() - start, warmup


def print_curve(curve: list[dict], best_epoch: int) -> None:
    """Print the dev curve as a table and mark the best epoch."""
    print("\nepoch  train_loss  dev_wer  dev_cer")
    for row in curve:
        mark = "  <- best" if row["epoch"] == best_epoch else ""
        print(f"{row['epoch']:5d}  {row['train_loss']:10.4f}  {row['dev_wer']:7.4f}  {row['dev_cer']:7.4f}{mark}")


def score_final_splits(args: argparse.Namespace, model: torch.nn.Module, name: str,
                       device: torch.device) -> dict[str, dict]:
    """Decode dev and every final split, write the predictions and return the metrics per split."""
    splits = {}
    for split in ("dev",) + data_io.FINAL_SPLITS:
        records = data_io.load_split(split, final=True)
        hyps = transcribe(model, Utterances(records, args.features), device, args.batch_frames)
        preds = [{"id": r["id"], "hyp": h} for r, h in zip(records, hyps)]
        metrics.write_preds(name, args.seed, split, preds, final=True)
        splits[split] = metrics.evaluate(records, hyps)
        print(f"{split}: wer={splits[split]['wer']:.4f} cer={splits[split]['cer']:.4f}")
    return splits


def main() -> None:
    args = parse_args()
    name = model_name(args)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    device = pick_device(args.device)
    train_records, n_dropped = drop_too_short(limit_hours(data_io.load_split("train"), args.max_hours), args.head)
    dev_records = data_io.load_split("dev")
    hours = sum(r["duration"] for r in train_records) / 3600
    dropped = f", {n_dropped} dropped as too short for CTC" if args.head == "ctc" else ""
    print(f"{name} seed {args.seed} on {hardware_name(device)}: {len(train_records)} train clips ({hours:.2f} h)"
          f"{dropped}, {len(dev_records)} dev clips")
    train_set = Utterances(train_records, args.features, None if args.no_aug else args.seed)
    dev_set = Utterances(dev_records, args.features)
    max_len = 2 * max(len(t) for t in train_set.targets)
    model = models.build_model(args.head, models.FEATURE_DIMS[args.features], max_len=max_len).to(device)
    best_state, best_epoch, curve, train_time, warmup = fit(args, model, train_set, dev_records, dev_set, device)
    print_curve(curve, best_epoch)
    if not args.final:
        print("without --final nothing is written")
        return
    model.load_state_dict(best_state)
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "args": vars(args), "max_len": max_len, "best_epoch": best_epoch},
               CHECKPOINT_DIR / f"{name}_seed{args.seed}.pt")
    config = {**vars(args), "hidden": models.HIDDEN, "layers": models.LAYERS, "dropout": models.DROPOUT,
              "warmup_steps": warmup, "n_train_clips": len(train_records), "n_train_dropped": n_dropped,
              "dev_curve": curve}
    if args.head == "aed":
        config.update(max_decode_len=max_len, label_smoothing=models.LABEL_SMOOTHING,
                      attention_dim=models.ATTENTION_DIM)
    if args.head == "rnnt":
        config.update(joint_dim=models.JOINT_DIM, max_symbols_per_frame=models.MAX_SYMBOLS_PER_FRAME)
    path = metrics.write_result(
        model=name, seed=args.seed, checkpoint=None, revision=None, config=config,
        train_time_s=round(train_time, 1), hardware=hardware_name(device), n_params=models.count_params(model),
        best_epoch=best_epoch, splits=score_final_splits(args, model, name, device),
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
