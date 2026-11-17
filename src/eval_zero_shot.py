"""Transcribe with pretrained Whisper models without training and apply a small rule converter."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src import data_io, metrics

MODELS = {
    "whisper_base_zs": "openai/whisper-base.en",
    "whisper_large_zs": "openai/whisper-large-v3",
}
SEED = 0
EVAL_SPLITS = ("dev",) + data_io.FINAL_SPLITS
SYMBOL_WORDS = {"dot": ".", "underscore": "_"}
MAX_NEW_TOKENS = 128


def _flush_letters(run: list[str]) -> list[str]:
    """Join a run of spelled letters, keeping a leading article "a" separate."""
    if len(run) >= 3 and run[0] == "a":
        return ["a", "".join(run[1:])]
    return ["".join(run)] if len(run) >= 2 else run


def join_spelled_letters(tokens: list[str]) -> list[str]:
    """Join two or more consecutive single-letter tokens into one token ("n p" -> "np")."""
    out: list[str] = []
    run: list[str] = []
    for tok in tokens:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        out.extend(_flush_letters(run))
        run = []
        out.append(tok)
    return out + _flush_letters(run)


def join_symbol_words(tokens: list[str]) -> list[str]:
    """Replace "dot" and "underscore" between two words by "." and "_" and glue the neighbours."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        symbol = SYMBOL_WORDS.get(tokens[i].lower())
        if symbol and out and i + 1 < len(tokens):
            out[-1] += symbol + tokens[i + 1]
            i += 2
        else:
            out.append(tokens[i])
            i += 1
    return out


def convert(raw: str) -> str:
    """Apply the rule converter to a raw hypothesis; no inventory and no learning."""
    tokens = data_io.normalize_hyp(raw).split()
    return " ".join(join_symbol_words(join_spelled_letters(tokens)))


def pick_device(name: str) -> torch.device:
    """Resolve "auto" to CUDA when available, otherwise CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def hardware_name(device: torch.device) -> str:
    """Describe the device the run used."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type}: {platform.machine()} {platform.processor()}".strip()


def load_model(checkpoint: str, device: torch.device) -> tuple[WhisperProcessor, WhisperForConditionalGeneration]:
    """Load a Whisper checkpoint in fp16 on CUDA and fp32 elsewhere."""
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(checkpoint)
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint, dtype=dtype).to(device).eval()
    return processor, model


def generate_kwargs(model: WhisperForConditionalGeneration) -> dict:
    """Greedy decoding; multilingual checkpoints are forced to English transcription."""
    kwargs = {"num_beams": 1, "do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}
    if getattr(model.generation_config, "is_multilingual", False):
        kwargs.update(language="en", task="transcribe")
    return kwargs


def transcribe(
    records: list[dict],
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    batch_size: int,
) -> list[str]:
    """Return one raw hypothesis per record, in record order."""
    hyps: list[str] = []
    for start in range(0, len(records), batch_size):
        waves = [data_io.load_audio(r) for r in records[start:start + batch_size]]
        inputs = processor.feature_extractor(waves, sampling_rate=data_io.SAMPLE_RATE, return_tensors="pt")
        feats = inputs.input_features.to(model.device, dtype=model.dtype)
        with torch.inference_mode():
            ids = model.generate(feats, **generate_kwargs(model))
        hyps.extend(t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True))
        print(f"  {min(start + batch_size, len(records))}/{len(records)}", end="\r", flush=True)
    print()
    return hyps


def read_manifest(path: Path) -> list[dict]:
    """Read a dev manifest or prompt file; clips without an "audio" field are looked up in data/raw/audio."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("split") != "dev":
            raise ValueError(f"--manifest accepts dev records only, got split {record.get('split')!r}")
        record.setdefault("audio", f"data/raw/audio/{record['engine']}/{record['id']}.opus")
        if (data_io.ROOT / record["audio"]).is_file():
            records.append(record)
    if not records:
        raise FileNotFoundError(f"no record in {path} has an audio file on disk")
    return records


def case_insensitive_em(records: list[dict], hyps: list[str]) -> float:
    """API exact match with both sides lowercased, computed by metrics.evaluate."""
    lowered = [{**r, "api": r["api"].lower(), "written": r["written"].lower()} for r in records]
    return metrics.evaluate(lowered, [h.lower() for h in hyps], n_boot=1)["api_em"]


def style_table(name: str, records: list[dict], outputs: dict[str, list[str]]) -> None:
    """Print api_em, case-insensitive api_em and wer for every output and verbalization style."""
    print(f"\n{name}: {len(records)} clips")
    print(f"{'output':<10} {'style':<9} {'n':>4} {'api_em':>7} {'95% CI':>15} {'nocase_em':>9} {'wer':>6}")
    groups = {"all": list(range(len(records)))}
    for i, r in enumerate(records):
        groups.setdefault(str(r.get("style")), []).append(i)
    for output, hyps in outputs.items():
        for style, ids in groups.items():
            recs, hs = [records[i] for i in ids], [hyps[i] for i in ids]
            m = metrics.evaluate(recs, hs)
            ci = f"[{m['api_em_ci'][0]:.3f}, {m['api_em_ci'][1]:.3f}]"
            print(f"{output:<10} {style:<9} {len(ids):>4} {m['api_em']:>7.3f} {ci:>15} "
                  f"{case_insensitive_em(recs, hs):>9.3f} {m['wer']:>6.3f}")


def print_examples(records: list[dict], raw: list[str], converted: list[str], n: int) -> None:
    """Print n seeded example pairs with the target, both outputs and whether the API matched."""
    picks = np.random.default_rng(SEED).permutation(len(records))[:n]
    for i in sorted(picks):
        hit = records[i]["api"] in data_io.normalize_hyp(converted[i]).split()
        print(f"[{'ok' if hit else 'xx'}] {records[i]['style']}\n  target:    {records[i]['written']}\n"
              f"  raw:       {raw[i]}\n  converted: {converted[i]}")


def run_dev(args: argparse.Namespace, device: torch.device) -> None:
    """Score the chosen models on dev (or a dev manifest) and write nothing."""
    records = read_manifest(args.manifest) if args.manifest else data_io.load_split("dev")
    records = records[:args.limit] if args.limit else records
    for name in args.models:
        processor, model = load_model(MODELS[name], device)
        raw = transcribe(records, processor, model, args.batch_size)
        converted = [convert(h) for h in raw]
        style_table(name, records, {"raw": raw, "converted": converted})
        print_examples(records, raw, converted, args.examples)
        del processor, model


def write_raw_preds(model: str, split: str, preds: list[dict]) -> Path:
    """Validate and write the raw hypotheses to results/preds/<model>_seed0_<split>.raw.jsonl."""
    metrics.validate_preds(split, preds, final=True)
    path = metrics.RESULTS_DIR / "preds" / f"{model}_seed{SEED}_{split}.raw.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"id": p["id"], "hyp": p["hyp"]}, ensure_ascii=False) for p in preds)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_final(args: argparse.Namespace, device: torch.device) -> None:
    """Write predictions and a run file for every final split and every chosen model."""
    for name in args.models:
        processor, model = load_model(MODELS[name], device)
        splits, start = {}, time.perf_counter()
        for split in EVAL_SPLITS:
            records = data_io.load_split(split, final=True)
            raw = transcribe(records, processor, model, args.batch_size)
            converted = [convert(h) for h in raw]
            preds = [{"id": r["id"], "hyp": h} for r, h in zip(records, converted)]
            metrics.write_preds(name, SEED, split, preds, final=True)
            write_raw_preds(name, split, [{"id": r["id"], "hyp": h} for r, h in zip(records, raw)])
            splits[split] = {**metrics.evaluate(records, converted), "raw": metrics.evaluate(records, raw)}
            print(f"{name} {split}: api_em {splits[split]['api_em']:.3f} (raw {splits[split]['raw']['api_em']:.3f})")
        config = {"checkpoint": MODELS[name], "decoding": "greedy", "language": "en",
                  "max_new_tokens": MAX_NEW_TOKENS, "batch_size": args.batch_size, "dtype": str(model.dtype),
                  "converter": "dot/underscore between words, spelled letters joined",
                  "decode_time_s": round(time.perf_counter() - start, 1)}
        n_params = sum(p.numel() for p in model.parameters())
        path = metrics.write_result(name, SEED, config, 0.0, hardware_name(device), n_params, None, splits)
        print(f"wrote {path}")
        del processor, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS),
                        help="models to run (default: both)")
    parser.add_argument("--final", action="store_true",
                        help="evaluate dev and all final splits and write predictions and run files")
    parser.add_argument("--manifest", type=Path,
                        help="dev manifest or prompt file to score instead of data/processed/dev.jsonl (pilot)")
    parser.add_argument("--limit", type=int, help="score only the first N dev clips")
    parser.add_argument("--batch-size", type=int, default=8, help="clips per forward pass")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="torch device")
    parser.add_argument("--examples", type=int, default=15, help="example pairs to print on dev")
    args = parser.parse_args()
    if args.final and (args.manifest or args.limit):
        parser.error("--final always scores the full splits; drop --manifest and --limit")
    return args


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    if args.final:
        run_final(args, device)
    else:
        run_dev(args, device)


if __name__ == "__main__":
    main()
