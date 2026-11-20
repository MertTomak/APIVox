"""Score transcripts with corpus-level WER, CER and code-switching metrics, and write result files."""
import json
from pathlib import Path
from typing import Any

import jiwer
import numpy as np

from src import data_io

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SLICE_DIMENSIONS = ("source", "program", "cs")
RU_TAGS = ("ru", "mix")
CI_METRICS = ("wer", "cer", "cs_wer")


def _check_aligned(records: list[dict], hyps: list[str]) -> None:
    if len(records) != len(hyps):
        raise ValueError(f"got {len(hyps)} hypotheses for {len(records)} records")
    if not records:
        raise ValueError("cannot score an empty list of utterances")


def _normalize(text: str) -> str:
    """Apply data_io.normalize_text, failing clearly if the data_io v2 contract (B0) is missing."""
    normalize = getattr(data_io, "normalize_text", None)
    if normalize is None:
        raise RuntimeError("data_io.normalize_text is missing: metrics needs data_io v2 (B0) to score")
    return normalize(text)


def _references(records: list[dict]) -> list[str]:
    """Return the reference texts after checking each is non-empty, normalized and fully tagged."""
    refs = []
    for r in records:
        text = r["text"]
        if not text:
            raise ValueError(f"record {r['id']!r}: empty reference text cannot be scored")
        if _normalize(text) != text:
            raise ValueError(f"record {r['id']!r}: reference text is not normalized: {text!r}")
        if len(r["lang_tags"]) != len(text.split()):
            raise ValueError(f"record {r['id']!r}: {len(r['lang_tags'])} lang_tags for {len(text.split())} words")
        refs.append(text)
    return refs


def _edits(alignment: list) -> int:
    """Count substituted, deleted and inserted units in one jiwer alignment."""
    total = 0
    for chunk in alignment:
        if chunk.type in ("substitute", "delete"):
            total += chunk.ref_end_idx - chunk.ref_start_idx
        elif chunk.type == "insert":
            total += chunk.hyp_end_idx - chunk.hyp_start_idx
    return total


def _ru_hits(alignment: list, tags: list[str]) -> int:
    """Count reference words tagged ru or mix that the alignment marks as equal."""
    return sum(
        tags[i] in RU_TAGS
        for chunk in alignment if chunk.type == "equal"
        for i in range(chunk.ref_start_idx, chunk.ref_end_idx)
    )


def _counts(records: list[dict], hyps: list[str]) -> dict[str, np.ndarray]:
    """Per-utterance error and reference counts for words, characters and ru/mix words."""
    _check_aligned(records, hyps)
    refs = _references(records)
    norm = [_normalize(h) for h in hyps]
    words = jiwer.process_words(refs, norm)
    chars = jiwer.process_characters(refs, norm)
    tags = [r["lang_tags"] for r in records]
    return {
        "word_edits": np.array([_edits(a) for a in words.alignments], dtype=float),
        "words": np.array([len(ref.split()) for ref in refs], dtype=float),
        "char_edits": np.array([_edits(a) for a in chars.alignments], dtype=float),
        "chars": np.array([len(ref) for ref in refs], dtype=float),
        "ru_hits": np.array([_ru_hits(a, t) for a, t in zip(words.alignments, tags)], dtype=float),
        "ru_words": np.array([sum(t in RU_TAGS for t in ts) for ts in tags], dtype=float),
        "cs": np.array([bool(r["cs"]) for r in records]),
    }


def _ratio(num: np.ndarray, den: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    """Return sum(num) / sum(den) over the masked utterances, or None when the denominator is 0."""
    if mask is not None:
        num, den = num[mask], den[mask]
    total = den.sum()
    return float(num.sum() / total) if total else None


def _metrics(c: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        "wer": _ratio(c["word_edits"], c["words"]),
        "cer": _ratio(c["char_edits"], c["chars"]),
        "cs_wer": _ratio(c["word_edits"], c["words"], c["cs"]),
        "mono_wer": _ratio(c["word_edits"], c["words"], ~c["cs"]),
        "ru_word_acc": _ratio(c["ru_hits"], c["ru_words"]),
        "n_utts": int(len(c["words"])),
        "n_ref_words": int(c["words"].sum()),
    }


def evaluate(records: list[dict], hyps: list[str]) -> dict[str, Any]:
    """Corpus-level metrics of aligned hypotheses; a metric whose subset is empty is None."""
    return _metrics(_counts(records, hyps))


def slices(records: list[dict], hyps: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """The evaluate metrics per source, per program and per cs value ("true" / "false")."""
    _check_aligned(records, hyps)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in SLICE_DIMENSIONS:
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            key = str(r[dim]).lower() if isinstance(r[dim], bool) else str(r[dim])
            groups.setdefault(key, []).append(i)
        result[dim] = {
            key: evaluate([records[i] for i in ids], [hyps[i] for i in ids])
            for key, ids in sorted(groups.items())
        }
    return result


def _resample_indices(n_items: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(n, n_items))


def _resampled_ratio(num: np.ndarray, den: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Ratio per resample; resamples with a zero denominator give NaN."""
    den_sum = den[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den_sum > 0, num[idx].sum(axis=1) / den_sum, np.nan)


def _percentile_ci(samples: np.ndarray) -> list[float] | None:
    if np.isnan(samples).all():
        return None
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_ci(records: list[dict], hyps: list[str], n: int = 1000, seed: int = 0) -> dict[str, list[float] | None]:
    """95 % percentile CI of wer, cer and cs_wer from resampling utterances; None if a metric is undefined."""
    c = _counts(records, hyps)
    idx = _resample_indices(len(records), n, seed)
    cs_words = np.where(c["cs"], c["words"], 0.0)
    cs_edits = np.where(c["cs"], c["word_edits"], 0.0)
    samples = {
        "wer": _resampled_ratio(c["word_edits"], c["words"], idx),
        "cer": _resampled_ratio(c["char_edits"], c["chars"], idx),
        "cs_wer": _resampled_ratio(cs_edits, cs_words, idx),
    }
    return {name: _percentile_ci(samples[name]) for name in CI_METRICS}


def paired_bootstrap(
    records: list[dict], hyps_a: list[str], hyps_b: list[str], n: int = 1000, seed: int = 0
) -> float:
    """Probability that system A has lower WER than system B on shared resamples, ties counting half."""
    a, b = _counts(records, hyps_a), _counts(records, hyps_b)
    idx = _resample_indices(len(records), n, seed)
    wer_a = _resampled_ratio(a["word_edits"], a["words"], idx)
    wer_b = _resampled_ratio(b["word_edits"], b["words"], idx)
    return float(((wer_a < wer_b).sum() + 0.5 * (wer_a == wer_b).sum()) / n)


def validate_preds(split: str, preds: list[dict], final: bool = False) -> None:
    """Raise ValueError unless preds hold exactly one string hypothesis for every id of the split."""
    ids: list[str] = []
    for p in preds:
        if not isinstance(p, dict) or not isinstance(p.get("id"), str) or not isinstance(p.get("hyp"), str):
            raise ValueError(f"every prediction needs a string 'id' and a string 'hyp', got {p!r}")
        ids.append(p["id"])
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate ids in predictions: {duplicates[:5]}")
    expected = {r["id"] for r in data_io.load_split(split, final=final)}
    missing, extra = sorted(expected - set(ids)), sorted(set(ids) - expected)
    if missing or extra:
        raise ValueError(f"ids differ from split '{split}': {len(missing)} missing {missing[:5]}, {len(extra)} unknown {extra[:5]}")


def write_preds(model: str, seed: int, split: str, preds: list[dict], final: bool = False) -> Path:
    """Validate and write the raw hypotheses to results/preds/<model>_seed<seed>_<split>.jsonl."""
    validate_preds(split, preds, final=final)
    path = RESULTS_DIR / "preds" / f"{model}_seed{seed}_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"id": p["id"], "hyp": p["hyp"]}, ensure_ascii=False) for p in preds)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_result(
    *,
    model: str,
    seed: int,
    checkpoint: str | None,
    revision: str | None,
    config: dict,
    train_time_s: float,
    hardware: str,
    n_params: int,
    best_epoch: int | None,
    splits: dict[str, dict],
) -> Path:
    """Write results/<model>_seed<seed>.json; checkpoint and revision are None for our own models."""
    run = {
        "model": model,
        "seed": seed,
        "checkpoint": checkpoint,
        "revision": revision,
        "config": config,
        "train_time_s": train_time_s,
        "hardware": hardware,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "splits": splits,
    }
    path = RESULTS_DIR / f"{model}_seed{seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
