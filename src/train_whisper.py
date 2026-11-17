"""Fine-tune the Transformer encoder-decoder openai/whisper-base.en on the train split."""
import argparse
import json
import random
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor, get_linear_schedule_with_warmup

from src import data_io, metrics
from src.eval_zero_shot import EVAL_SPLITS, hardware_name, load_model, pick_device, transcribe

MODEL_NAME = "whisper_base_ft"
FIXTURE = data_io.ROOT / "tests" / "fixtures" / "mini.jsonl"
CONFIG = {
    "checkpoint": "openai/whisper-base.en",
    "lr": 1e-5,
    "weight_decay": 0.0,
    "warmup_steps": 500,
    "batch_size": 16,
    "epochs": 5,
    "grad_clip": 1.0,
    "eval_batch_size": 32,
    "decoding": "greedy",
}
SWEEP_LRS = (5e-6, 1e-5, 3e-5)
SWEEP_SEED = 42
IGNORE_INDEX = -100


class SpeechDataset(Dataset):
    """Yields Whisper log-mel features and token ids of the written target for each record."""

    def __init__(self, records: list[dict], processor: WhisperProcessor) -> None:
        self.records = records
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tuple[np.ndarray, list[int]]:
        record = self.records[i]
        wave = data_io.load_audio(record)
        feats = self.processor.feature_extractor(wave, sampling_rate=data_io.SAMPLE_RATE, return_tensors="np")
        return feats.input_features[0], self.processor.tokenizer(record["written"]).input_ids


def make_collate(start_id: int):
    """Build a collate function that stacks features and pads labels with -100."""

    def collate(batch: list[tuple[np.ndarray, list[int]]]) -> dict[str, torch.Tensor]:
        labels = [ids[1:] if ids and ids[0] == start_id else ids for _, ids in batch]
        width = max(len(ids) for ids in labels)
        padded = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        for row, ids in enumerate(labels):
            padded[row, :len(ids)] = torch.tensor(ids)
        feats = torch.from_numpy(np.stack([f for f, _ in batch]))
        return {"input_features": feats, "labels": padded}

    return collate


def allow_target_chars(model: WhisperForConditionalGeneration, processor: WhisperProcessor) -> None:
    """Stop suppressing generation tokens made only of target characters, such as "_"."""
    config = model.generation_config
    allowed = set(data_io.CHARS)
    config.suppress_tokens = [
        t for t in config.suppress_tokens
        if not (processor.tokenizer.decode([t]).strip() and set(processor.tokenizer.decode([t])) <= allowed)
    ]


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch and ask cuDNN for deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int, seed: int) -> None:
    """Seed numpy inside a data loader worker."""
    np.random.seed(seed + worker_id)


def make_loader(records: list[dict], processor: WhisperProcessor, seed: int, workers: int) -> DataLoader:
    """Shuffled train loader with a seeded generator and seeded workers."""
    return DataLoader(
        SpeechDataset(records, processor),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=workers,
        collate_fn=make_collate(processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")),
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=partial(seed_worker, seed=seed),
    )


def train_epoch(
    model: WhisperForConditionalGeneration,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    max_steps: int | None,
) -> float:
    """Run one epoch of teacher-forced cross-entropy training and return the mean loss."""
    model.train()
    use_amp = model.device.type == "cuda"
    losses = []
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            loss = model(**batch).loss
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        losses.append(loss.item())
        if max_steps and step >= max_steps:
            break
    return float(np.mean(losses))


def decode(model: WhisperForConditionalGeneration, processor: WhisperProcessor, records: list[dict]) -> list[str]:
    """Greedy transcription in eval mode, with fp16 autocast on CUDA."""
    model.eval()
    with torch.autocast("cuda", dtype=torch.float16, enabled=model.device.type == "cuda"):
        return transcribe(records, processor, model, CONFIG["eval_batch_size"])


def fit(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train: list[dict],
    dev: list[dict],
    lr: float,
    epochs: int,
    seed: int,
    workers: int,
    max_steps: int | None = None,
) -> tuple[dict, int, list[dict]]:
    """Train, score dev api_em after every epoch and return the best weights, best epoch and dev curve."""
    model.float()
    allow_target_chars(model, processor)
    loader = make_loader(train, processor, seed, workers)
    total = epochs * (min(len(loader), max_steps) if max_steps else len(loader))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CONFIG["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, min(CONFIG["warmup_steps"], total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=model.device.type == "cuda")
    best, curve = (-1.0, float("inf")), []
    best_state, best_epoch = {}, 0
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, max_steps)
        m = metrics.evaluate(dev, decode(model, processor, dev))
        curve.append({"epoch": epoch, "train_loss": loss, "dev_api_em": m["api_em"], "dev_wer": m["wer"]})
        print(f"epoch {epoch}: train loss {loss:.4f}  dev api_em {m['api_em']:.4f}  dev wer {m['wer']:.4f}")
        if m["api_em"] > best[0] or (m["api_em"] == best[0] and m["wer"] < best[1]):
            best, best_epoch = (m["api_em"], m["wer"]), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_epoch, curve


def read_fixture() -> list[dict]:
    """Read the validated fixture clips used for the CPU smoke run."""
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    for record in records:
        data_io.validate(record)
    return records


def run_smoke(args: argparse.Namespace, device: torch.device) -> None:
    """Take one optimizer step on the fixture clips and decode them; writes nothing."""
    set_seed(args.seed)
    records = read_fixture()
    processor, model = load_model(CONFIG["checkpoint"], device)
    fit(model, processor, records, records, CONFIG["lr"], 1, args.seed, 0, max_steps=1)


def run_sweep(args: argparse.Namespace, device: torch.device) -> None:
    """Train one epoch per candidate learning rate with seed 42 and print the dev table; writes nothing."""
    train, dev = data_io.load_split("train"), data_io.load_split("dev")
    rows = []
    for lr in SWEEP_LRS:
        set_seed(SWEEP_SEED)
        processor, model = load_model(CONFIG["checkpoint"], device)
        _, _, curve = fit(model, processor, train, dev, lr, 1, SWEEP_SEED, args.workers)
        rows.append((lr, curve[-1]))
        del processor, model
    print(f"\n{'lr':>8} {'train_loss':>10} {'dev_api_em':>10} {'dev_wer':>8}")
    for lr, c in rows:
        print(f"{lr:>8.0e} {c['train_loss']:>10.4f} {c['dev_api_em']:>10.4f} {c['dev_wer']:>8.4f}")


def run_train(args: argparse.Namespace, device: torch.device) -> None:
    """Fine-tune with the fixed config; with --final also write predictions and the run file."""
    set_seed(args.seed)
    train, dev = data_io.load_split("train"), data_io.load_split("dev")
    processor, model = load_model(CONFIG["checkpoint"], device)
    start = time.perf_counter()
    best_state, best_epoch, curve = fit(model, processor, train, dev, CONFIG["lr"], CONFIG["epochs"],
                                        args.seed, args.workers)
    train_time = time.perf_counter() - start
    print(f"best epoch {best_epoch}, train time {train_time:.0f} s")
    if not args.final:
        return
    model.load_state_dict(best_state)
    splits = {}
    for split in EVAL_SPLITS:
        records = data_io.load_split(split, final=True)
        hyps = decode(model, processor, records)
        metrics.write_preds(MODEL_NAME, args.seed, split, [{"id": r["id"], "hyp": h} for r, h in zip(records, hyps)],
                            final=True)
        splits[split] = metrics.evaluate(records, hyps)
        print(f"{split}: api_em {splits[split]['api_em']:.4f}  wer {splits[split]['wer']:.4f}")
    config = {**CONFIG, "precision": "fp16 autocast" if device.type == "cuda" else "fp32", "dev_curve": curve}
    n_params = sum(p.numel() for p in model.parameters())
    path = metrics.write_result(MODEL_NAME, args.seed, config, round(train_time, 1), hardware_name(device),
                                n_params, best_epoch, splits)
    print(f"wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true",
                      help="after training, write predictions and the run file for dev and all final splits")
    mode.add_argument("--lr-sweep", action="store_true",
                      help=f"one epoch per learning rate in {SWEEP_LRS} with seed {SWEEP_SEED}, dev only")
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
