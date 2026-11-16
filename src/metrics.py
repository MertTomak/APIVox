"""Compute API exact match, error rates, confidence intervals and result files."""
import argparse
import json
from pathlib import Path
from typing import Any, Collection

import jiwer
import numpy as np

from src import data_io

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SLICE_DIMENSIONS = ("library", "engine", "style", "speaker", "api_seen_in_train")


def _check_aligned(records: list[dict], hyps: list[str]) -> None:
    if len(records) != len(hyps):
        raise ValueError(f"got {len(hyps)} hypotheses for {len(records)} records")
    if not records:
        raise ValueError("cannot score an empty list of utterances")


def _api_hits(records: list[dict], norm_hyps: list[str]) -> np.ndarray:
    hits = [r["api"] in h.split() for r, h in zip(records, norm_hyps)]
    return np.array(hits, dtype=float)


def _word_errors(records: list[dict], norm_hyps: list[str]) -> tuple[np.ndarray, np.ndarray]:
    edits, words = [], []
    for r, h in zip(records, norm_hyps):
        out = jiwer.process_words(r["written"].lower(), h.lower())
        edits.append(out.substitutions + out.deletions + out.insertions)
        words.append(out.hits + out.substitutions + out.deletions)
    return np.array(edits, dtype=float), np.array(words, dtype=float)


def _closest_token_cer(api: str, norm_hyp: str) -> float:
    """Character error rate of the API against its closest hypothesis token, capped at 1."""
    tokens = norm_hyp.split()
    if not tokens:
        return 1.0
    return min(1.0, min(jiwer.cer(api, t) for t in tokens))


def _is_code_like(token: str) -> bool:
    has_letter = any(c.isalpha() for c in token)
    return has_letter and ("." in token or "_" in token or any(c.isupper() for c in token[1:]))


def _has_nonexistent_api(norm_hyp: str, index: Collection[str]) -> bool:
    return any(_is_code_like(t) and t not in index for t in norm_hyp.split())


def _percentile_ci(samples: np.ndarray) -> list[float]:
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def _resample_indices(n_items: int, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_items, size=(n, n_items))


def _bootstrap(hits: np.ndarray, edits: np.ndarray, words: np.ndarray, n: int, seed: int) -> dict[str, list[float]]:
    idx = _resample_indices(len(hits), n, seed)
    em = hits[idx].mean(axis=1)
    wer = edits[idx].sum(axis=1) / words[idx].sum(axis=1)
    return {"api_em": _percentile_ci(em), "wer": _percentile_ci(wer)}


def evaluate(
    records: list[dict],
    hyps: list[str],
    index: Collection[str] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Score aligned hypotheses; nonexistent_api_rate is None when no inventory index is given."""
    _check_aligned(records, hyps)
    norm = [data_io.normalize_hyp(h) for h in hyps]
    hits = _api_hits(records, norm)
    edits, words = _word_errors(records, norm)
    refs = [r["written"].lower() for r in records]
    nonexistent = None if index is None else float(np.mean([_has_nonexistent_api(h, index) for h in norm]))
    return {
        "api_em": float(hits.mean()),
        "api_em_ci": _bootstrap(hits, edits, words, n_boot, seed)["api_em"],
        "wer": float(edits.sum() / words.sum()),
        "cer": float(jiwer.cer(refs, [h.lower() for h in norm])),
        "api_cer": float(np.mean([_closest_token_cer(r["api"], h) for r, h in zip(records, norm)])),
        "nonexistent_api_rate": nonexistent,
    }


def slices(
    records: list[dict],
    hyps: list[str],
    index: Collection[str] | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Metrics per library, engine, style, speaker and api_seen_in_train, each with its size n."""
    _check_aligned(records, hyps)
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in SLICE_DIMENSIONS:
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(records):
            value = r[dim]
            key = str(value).lower() if isinstance(value, bool) else "none" if value is None else str(value)
            groups.setdefault(key, []).append(i)
        result[dim] = {
            key: {**evaluate([records[i] for i in ids], [hyps[i] for i in ids], index, n_boot, seed), "n": len(ids)}
            for key, ids in sorted(groups.items())
        }
    return result


def bootstrap_ci(records: list[dict], hyps: list[str], n: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    """95% percentile CI of api_em and wer from resampling utterances."""
    _check_aligned(records, hyps)
    norm = [data_io.normalize_hyp(h) for h in hyps]
    edits, words = _word_errors(records, norm)
    return _bootstrap(_api_hits(records, norm), edits, words, n, seed)


def paired_bootstrap(
    records: list[dict], hyps_a: list[str], hyps_b: list[str], n: int = 1000, seed: int = 0
) -> float:
    """Probability that system A beats system B on api_em over shared resamples, ties counting half."""
    _check_aligned(records, hyps_a)
    _check_aligned(records, hyps_b)
    hits_a = _api_hits(records, [data_io.normalize_hyp(h) for h in hyps_a])
    hits_b = _api_hits(records, [data_io.normalize_hyp(h) for h in hyps_b])
    idx = _resample_indices(len(records), n, seed)
    diff = hits_a[idx].mean(axis=1) - hits_b[idx].mean(axis=1)
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / n)


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
    """Validate and write results/preds/<model>_seed<seed>_<split>.jsonl."""
    validate_preds(split, preds, final=final)
    path = RESULTS_DIR / "preds" / f"{model}_seed{seed}_{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps({"id": p["id"], "hyp": p["hyp"]}, ensure_ascii=False) for p in preds)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_result(
    model: str,
    seed: int,
    config: dict,
    train_time_s: float,
    hardware: str,
    n_params: int,
    best_epoch: int | None,
    splits: dict[str, dict],
) -> Path:
    """Write results/<model>_seed<seed>.json."""
    run = {
        "model": model,
        "seed": seed,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
