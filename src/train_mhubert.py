"""Fine-tune HuBERT base utter-project/mHuBERT-147 with a new CTC head over our character set."""
import argparse
import json
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import HubertForCTC, Wav2Vec2FeatureExtractor, get_linear_schedule_with_warmup

from src import data_io, metrics
from src.eval_zero_shot import EVAL_SPLITS, hardware_name, make_batches, pick_device
from src.models import BLANK, ctc_collapse, decode_ids, encode_text
from src.train_whisper import FIXTURE, seed_worker, set_seed

MODEL_NAME = "mhubert_ft"
CHECKPOINT = "utter-project/mHuBERT-147"
REVISION = "7ad3fc0bc5106c58c9c13526abccad527150d135"
KSC_TRAINING = ("not stated: the model card and its pre-training manifests (CommonVoice v11, MLS, VoxLingua107, "
                "VoxPopuli and others) name neither KSC nor KSC2")
CONFIG = {
    "head_lr": 1e-3,
    "encoder_lr": 3e-5,
    "weight_decay": 0.0,
    "warmup_ratio": 0.1,
    "batch_seconds": 100.0,
    "epochs": 10,
    "grad_clip": 1.0,
    "mask_time_prob": 0.05,
    "eval_batch_seconds": 160.0,
    "decoding": "greedy CTC with src.models.ctc_collapse",
}
SWEEP_ENCODER_LRS = (3e-5, 5e-5)
SWEEP_SEED = 42
IGNORE_INDEX = -100


class SpeechDataset(Dataset):
    """Yields normalized raw waveforms and CTC targets (CHAR2ID + 1, so 0 stays the blank)."""

    def __init__(self, records: list[dict], extractor: Wav2Vec2FeatureExtractor) -> None:
        self.records = records
        self.extractor = extractor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[np.ndarray, list[int]]:
        record = self.records[i]
        wave = self.extractor(data_io.load_audio(record), sampling_rate=data_io.SAMPLE_RATE).input_values[0]
        return np.asarray(wave, dtype=np.float32), encode_text(record["text"])


def collate(batch: list[tuple[np.ndarray, list[int]]]) -> dict[str, torch.Tensor]:
    """Zero-pad waveforms (no attention mask, as for group-norm HuBERT), keep true lengths, pad labels with -100."""
    width = max(len(w) for w, _ in batch)
    waves = torch.zeros(len(batch), width)
    labels = torch.full((len(batch), max(len(t) for _, t in batch)), IGNORE_INDEX, dtype=torch.long)
    for row, (wave, target) in enumerate(batch):
        waves[row, :len(wave)] = torch.from_numpy(wave)
        labels[row, :len(target)] = torch.tensor(target)
    return {"input_values": waves, "lengths": torch.tensor([len(w) for w, _ in batch]), "labels": labels}


def ctc_loss(model: HubertForCTC, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """CTC loss over each clip's own frame count, so padded frames are never aligned to text."""
    labels = batch["labels"]
    frames = model._get_feat_extract_output_lengths(batch["lengths"])
    log_probs = torch.log_softmax(model(input_values=batch["input_values"]).logits.float(), dim=-1)
    return torch.nn.functional.ctc_loss(
        log_probs.transpose(0, 1), labels[labels != IGNORE_INDEX], frames, (labels != IGNORE_INDEX).sum(dim=-1),
        blank=BLANK, reduction="mean", zero_infinity=True,
    )


def load_trainable(device: torch.device) -> tuple[Wav2Vec2FeatureExtractor, HubertForCTC]:
    """Load mHuBERT-147 at its pinned revision with a new CTC head and a frozen convolutional encoder."""
    source = {"pretrained_model_name_or_path": CHECKPOINT, "revision": REVISION}
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(**source)
    model = HubertForCTC.from_pretrained(
        **source, vocab_size=len(data_io.CHARS) + 1, pad_token_id=BLANK, mask_time_prob=CONFIG["mask_time_prob"],
        dtype=torch.float32,
    )
    model.freeze_feature_encoder()
    return extractor, model.to(device)


def make_loader(records: list[dict], extractor: Wav2Vec2FeatureExtractor, seed: int, workers: int,
                epoch: int) -> DataLoader:
    """Duration-bucketed batches in a seeded random order that changes every epoch."""
    batches = make_batches([r["duration"] for r in records], CONFIG["batch_seconds"])
    order = np.random.default_rng(seed + epoch).permutation(len(batches))
    return DataLoader(SpeechDataset(records, extractor), batch_sampler=[batches[i] for i in order],
                      num_workers=workers, collate_fn=collate, worker_init_fn=partial(seed_worker, seed=seed))


def optimizer_for(model: HubertForCTC, encoder_lr: float) -> torch.optim.Optimizer:
    """AdamW with one learning rate for the new head and another for the pre-trained Transformer."""
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("lm_head")]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("lm_head")]
    return torch.optim.AdamW([{"params": head, "lr": CONFIG["head_lr"]}, {"params": body, "lr": encoder_lr}],
                             weight_decay=CONFIG["weight_decay"])


def train_epoch(model: HubertForCTC, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LRScheduler, scaler: torch.amp.GradScaler,
                max_steps: int | None) -> float:
    """Run one epoch of CTC training with fp16 autocast on CUDA and return the mean loss."""
    model.train()
    losses = []
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=model.device.type == "cuda"):
            loss = ctc_loss(model, batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step}/{len(loader)}  loss {np.mean(losses[-100:]):.4f}", flush=True)
        if max_steps and step >= max_steps:
            break
    return float(np.mean(losses))


def decode(model: HubertForCTC, extractor: Wav2Vec2FeatureExtractor, records: list[dict]) -> list[str]:
    """Greedy CTC decoding: argmax per frame within each clip's own length, then ctc_collapse."""
    model.eval()
    dataset = SpeechDataset(records, extractor)
    hyps = [""] * len(records)
    for batch in make_batches([r["duration"] for r in records], CONFIG["eval_batch_seconds"]):
        waves = [dataset[i][0] for i in batch]
        items = collate([(w, [BLANK]) for w in waves])
        lengths = model._get_feat_extract_output_lengths(items["lengths"])
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                                     enabled=model.device.type == "cuda"):
            ids = model(input_values=items["input_values"].to(model.device)).logits.argmax(dim=-1).cpu()
        for row, i in enumerate(batch):
            hyps[i] = decode_ids(ctc_collapse(ids[row, :int(lengths[row])].tolist()))
    return hyps


def fit(model: HubertForCTC, extractor: Wav2Vec2FeatureExtractor, train: list[dict], dev: list[dict],
        encoder_lr: float, epochs: int, seed: int, workers: int,
        max_steps: int | None = None) -> tuple[dict, int, list[dict]]:
    """Train, score dev WER after every epoch and return the best weights, the best epoch and the dev curve."""
    steps_per_epoch = len(make_loader(train, extractor, seed, 0, 0))
    total = epochs * (min(steps_per_epoch, max_steps) if max_steps else steps_per_epoch)
    optimizer = optimizer_for(model, encoder_lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(CONFIG["warmup_ratio"] * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    best, best_state, best_epoch, curve = (float("inf"), float("inf")), {}, 0, []
    for epoch in range(1, epochs + 1):
        loader = make_loader(train, extractor, seed, workers, epoch)
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, max_steps)
        m = metrics.evaluate(dev, decode(model, extractor, dev))
        curve.append({"epoch": epoch, "train_loss": loss, "dev_wer": m["wer"], "dev_cer": m["cer"]})
        print(f"epoch {epoch}: train loss {loss:.4f}  dev wer {m['wer']:.4f}  dev cer {m['cer']:.4f}", flush=True)
        if (m["wer"], m["cer"]) < best:
            best, best_epoch = (m["wer"], m["cer"]), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_epoch, curve


def read_fixture() -> list[dict]:
    """Read and validate the fixture clips used for the CPU smoke run."""
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        data_io.validate(record)
    return records


def run_smoke(args: argparse.Namespace, device: torch.device) -> None:
    """Take one optimizer step on the fixture train clips and score the fixture dev clips; writes nothing."""
    set_seed(args.seed)
    records = read_fixture()
    train, dev = [r for r in records if r["split"] == "train"], [r for r in records if r["split"] == "dev"]
    extractor, model = load_trainable(device)
    fit(model, extractor, train, dev, CONFIG["encoder_lr"], 1, args.seed, 0, max_steps=1)


def run_sweep(args: argparse.Namespace, device: torch.device) -> None:
    """Train one epoch per candidate encoder learning rate with seed 42 and print the dev table; writes nothing."""
    train, dev = data_io.load_split("train"), data_io.load_split("dev")
    rows = []
    for lr in SWEEP_ENCODER_LRS:
        set_seed(SWEEP_SEED)
        extractor, model = load_trainable(device)
        rows.append((lr, fit(model, extractor, train, dev, lr, 1, SWEEP_SEED, args.workers)[2][-1]))
        del extractor, model
    print(f"\n{'encoder_lr':>10} {'train_loss':>10} {'dev_wer':>8} {'dev_cer':>8}")
    for lr, c in rows:
        print(f"{lr:>10.0e} {c['train_loss']:>10.4f} {c['dev_wer']:>8.4f} {c['dev_cer']:>8.4f}")


def write_final(model: HubertForCTC, extractor: Wav2Vec2FeatureExtractor, args: argparse.Namespace,
                device: torch.device, best_epoch: int, curve: list[dict], train_time: float) -> None:
    """Decode dev and the final splits with the best weights and write predictions and the run file."""
    splits = {}
    for split in EVAL_SPLITS:
        records = data_io.load_split(split, final=True)
        hyps = decode(model, extractor, records)
        metrics.write_preds(MODEL_NAME, args.seed, split, [{"id": r["id"], "hyp": h} for r, h in zip(records, hyps)],
                            final=True)
        splits[split] = metrics.evaluate(records, hyps)
        print(f"{split}: wer {splits[split]['wer']:.4f}  cer {splits[split]['cer']:.4f}")
    config = {**CONFIG, "architecture": "HuBERT base + CTC head over data_io.CHARS (blank = 0)",
              "initialized_from": CHECKPOINT, "precision": "fp16 autocast" if device.type == "cuda" else "fp32",
              "frozen": "convolutional feature encoder", "ksc_training": KSC_TRAINING, "dev_curve": curve}
    path = metrics.write_result(
        model=MODEL_NAME, seed=args.seed, checkpoint=CHECKPOINT, revision=REVISION, config=config,
        train_time_s=round(train_time, 1), hardware=hardware_name(device),
        n_params=sum(p.numel() for p in model.parameters()), best_epoch=best_epoch, splits=splits,
    )
    print(f"wrote {path}")


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    """Fine-tune with the fixed config; with --final also write predictions and the run file."""
    set_seed(args.seed)
    train, dev = data_io.load_split("train"), data_io.load_split("dev")
    extractor, model = load_trainable(device)
    start = time.perf_counter()
    best_state, best_epoch, curve = fit(model, extractor, train, dev, CONFIG["encoder_lr"], CONFIG["epochs"],
                                        args.seed, args.workers)
    train_time = time.perf_counter() - start
    print(f"best epoch {best_epoch}, train time {train_time:.0f} s")
    if args.final:
        model.load_state_dict(best_state)
        write_final(model, extractor, args, device, best_epoch, curve, train_time)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true",
                      help="after training, write predictions and the run file for dev and the final splits")
    mode.add_argument("--lr-sweep", action="store_true",
                      help=f"one epoch per encoder learning rate in {SWEEP_ENCODER_LRS} with seed {SWEEP_SEED}")
    mode.add_argument("--smoke", action="store_true", help="one training step on the fixture clips, writes nothing")
    parser.add_argument("--seed", type=int, default=42, help="random seed (13, 42 or 2026)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="torch device")
    parser.add_argument("--workers", type=int, default=2, help="data loader worker processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if args.smoke:
        run_smoke(args, device)
    elif args.lr_sweep:
        run_sweep(args, device)
    else:
        run_train(args, device)


if __name__ == "__main__":
    main()
