"""Transcribe the dev and final splits with published Kazakh-capable checkpoints, without training."""
import argparse
import platform
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor, Wav2Vec2ForCTC, WhisperForConditionalGeneration, WhisperProcessor

from src import data_io, metrics

MODELS: dict[str, dict[str, str]] = {
    "whisper_small_zs": {
        "checkpoint": "openai/whisper-small",
        "revision": "973afd24965f72e36ca33b3055d56a652f456b4d",
        "family": "whisper",
        "architecture": "Transformer encoder-decoder",
        "ksc_training": "not stated: the model card describes 680k hours of web audio and names no datasets",
    },
    "whisper_large_zs": {
        "checkpoint": "openai/whisper-large-v3",
        "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "family": "whisper",
        "architecture": "Transformer encoder-decoder",
        "ksc_training": "not stated: the model card describes 1M hours of weakly labeled and 4M hours of "
                        "pseudo-labeled web audio and names no datasets",
    },
    "xlsr_kk_zs": {
        "checkpoint": "aismlv/wav2vec2-large-xlsr-kazakh",
        "revision": "b277ef85de7bbd28302d570062feb196cc4e9107",
        "family": "ctc",
        "architecture": "wav2vec 2.0 XLSR-53 + CTC",
        "ksc_training": "yes: fine-tuned on the KSC v1.1 train split (model card); test_read is the KSC test set",
    },
    "mms_kk_zs": {
        "checkpoint": "facebook/mms-1b-all",
        "revision": "3d33597edbdaaba14a8e858e2c8caa76e3cec0cd",
        "family": "ctc",
        "adapter": "kaz",
        "architecture": "wav2vec 2.0 MMS-1B + kaz language adapter",
        "ksc_training": "not stated: the model card lists google/fleurs only",
    },
}
SEED = 0
EVAL_SPLITS = ("dev",) + data_io.FINAL_SPLITS
LANGUAGE = "kk"
WHISPER_WINDOW_S = 30.0
MAX_NEW_TOKENS = 440
N_EXAMPLES = 10


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


def load_model(name: str, device: torch.device) -> tuple[Any, torch.nn.Module]:
    """Load the processor and model of a table entry at its pinned revision, fp16 on CUDA."""
    spec = MODELS[name]
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    source = {"pretrained_model_name_or_path": spec["checkpoint"], "revision": spec["revision"]}
    if spec["family"] == "whisper":
        processor = WhisperProcessor.from_pretrained(**source)
        model = WhisperForConditionalGeneration.from_pretrained(**source, dtype=dtype)
    else:
        processor = AutoProcessor.from_pretrained(**source)
        model = Wav2Vec2ForCTC.from_pretrained(**source, dtype=dtype)
        if "adapter" in spec:
            processor.tokenizer.set_target_lang(spec["adapter"])
            model.load_adapter(spec["adapter"], revision=spec["revision"])
    return processor, model.to(device).eval()


def make_batches(durations: list[float], budget_s: float, fixed_s: float | None = None) -> list[list[int]]:
    """Group indices sorted by duration so that batch size times the longest (or fixed) clip stays within budget."""
    batches: list[list[int]] = []
    current: list[int] = []
    for i in sorted(range(len(durations)), key=lambda j: (durations[j], j)):
        cost = fixed_s or durations[i]
        if current and (len(current) + 1) * cost > budget_s:
            batches.append(current)
            current = []
        current.append(i)
    return batches + [current] if current else batches


def _decode_whisper(waves: list[np.ndarray], processor: Any, model: torch.nn.Module) -> list[str]:
    feats = processor.feature_extractor(waves, sampling_rate=data_io.SAMPLE_RATE, return_tensors="pt")
    ids = model.generate(
        feats.input_features.to(model.device, dtype=model.dtype),
        language=LANGUAGE, task="transcribe", num_beams=1, do_sample=False,
        return_timestamps=False, max_new_tokens=MAX_NEW_TOKENS,
    )
    return processor.batch_decode(ids, skip_special_tokens=True)


def _decode_ctc(waves: list[np.ndarray], processor: Any, model: torch.nn.Module) -> list[str]:
    inputs = processor(waves, sampling_rate=data_io.SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device, dtype=model.dtype) if v.is_floating_point() else v.to(model.device))
              for k, v in inputs.items()}
    return processor.batch_decode(model(**inputs).logits.argmax(dim=-1))


def transcribe(records: list[dict], processor: Any, model: torch.nn.Module, family: str,
               budget_s: float) -> list[str]:
    """Return one raw hypothesis per record, in record order."""
    fixed = WHISPER_WINDOW_S if family == "whisper" else None
    decode = _decode_whisper if family == "whisper" else _decode_ctc
    hyps = [""] * len(records)
    done = 0
    for batch in make_batches([r["duration"] for r in records], budget_s, fixed):
        with torch.inference_mode():
            texts = decode([data_io.load_audio(records[i]) for i in batch], processor, model)
        for i, text in zip(batch, texts):
            hyps[i] = text.strip()
        done += len(batch)
        print(f"  {done}/{len(records)}", end="\r", flush=True)
    print()
    return hyps


def timed_transcribe(records: list[dict], processor: Any, model: torch.nn.Module, family: str,
                     budget_s: float) -> tuple[list[str], float]:
    """Transcribe and return the hypotheses with the real-time factor (decode time / audio time)."""
    start = time.perf_counter()
    hyps = transcribe(records, processor, model, family, budget_s)
    rtf = (time.perf_counter() - start) / sum(r["duration"] for r in records)
    return hyps, rtf


def print_report(name: str, records: list[dict], hyps: list[str], rtf: float) -> None:
    """Print dev WER, CER, code-switching metrics and seeded example pairs."""
    m = metrics.evaluate(records, hyps)
    fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
    print(f"\n{name}: {m['n_utts']} utterances, RTF {rtf:.3f}")
    print("  " + "  ".join(f"{k} {fmt(m[k])}" for k in ("wer", "cer", "cs_wer", "mono_wer", "ru_word_acc")))
    for i in sorted(np.random.default_rng(SEED).permutation(len(records))[:N_EXAMPLES]):
        print(f"  [{'cs' if records[i]['cs'] else 'kz'}] ref: {records[i]['text']}\n"
              f"       hyp: {data_io.normalize_text(hyps[i])}")


def run_dev(args: argparse.Namespace, device: torch.device) -> None:
    """Score the chosen models on dev and write nothing."""
    records = data_io.load_split("dev")
    records = records[:args.limit] if args.limit else records
    for name in args.models:
        processor, model = load_model(name, device)
        hyps, rtf = timed_transcribe(records, processor, model, MODELS[name]["family"], args.batch_seconds)
        print_report(name, records, hyps, rtf)
        del processor, model


def run_config(name: str, model: torch.nn.Module, args: argparse.Namespace, rtf: dict[str, float]) -> dict:
    """Settings recorded in the run file, including what the model card says about KSC training data."""
    spec = MODELS[name]
    return {"architecture": spec["architecture"], "family": spec["family"], "adapter": spec.get("adapter"),
            "language": LANGUAGE if spec["family"] == "whisper" else None, "decoding": "greedy",
            "dtype": str(model.dtype).removeprefix("torch."), "batch_seconds": args.batch_seconds,
            "ksc_training": spec["ksc_training"], "rtf": rtf}


def run_final(args: argparse.Namespace, device: torch.device) -> None:
    """Write raw predictions and one run file per chosen model for the chosen splits."""
    for name in args.models:
        spec = MODELS[name]
        processor, model = load_model(name, device)
        splits, rtf, start = {}, {}, time.perf_counter()
        for split in args.splits:
            records = data_io.load_split(split, final=True)
            hyps, rtf[split] = timed_transcribe(records, processor, model, spec["family"], args.batch_seconds)
            metrics.write_preds(name, SEED, split, [{"id": r["id"], "hyp": h} for r, h in zip(records, hyps)],
                                final=True)
            splits[split] = metrics.evaluate(records, hyps)
            print(f"{name} {split}: wer {splits[split]['wer']:.4f}  cer {splits[split]['cer']:.4f}  RTF {rtf[split]:.3f}")
        config = {**run_config(name, model, args, rtf), "decode_time_s": round(time.perf_counter() - start, 1)}
        path = metrics.write_result(
            model=name, seed=SEED, checkpoint=spec["checkpoint"], revision=spec["revision"], config=config,
            train_time_s=0.0, hardware=hardware_name(device), n_params=sum(p.numel() for p in model.parameters()),
            best_epoch=None, splits=splits,
        )
        print(f"wrote {path}")
        del processor, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS),
                        help="models to run (default: all four)")
    parser.add_argument("--final", action="store_true",
                        help="score dev and the final splits and write predictions and run files")
    parser.add_argument("--splits", nargs="+", choices=EVAL_SPLITS, default=list(EVAL_SPLITS),
                        help="splits for --final (default: all four)")
    parser.add_argument("--limit", type=int, help="score only the first N dev utterances (dev mode only)")
    parser.add_argument("--batch-seconds", type=float, default=160.0,
                        help="audio seconds per batch, counting padding (Whisper clips count as 30 s)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"], help="torch device")
    args = parser.parse_args()
    if args.final and args.limit:
        parser.error("--final always scores full splits; drop --limit")
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
