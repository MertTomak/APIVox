"""Build the benchmark and ablation tables and figures by rescoring every stored prediction file."""
import argparse
import csv
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import jiwer
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import data_io, metrics

RESULTS = metrics.RESULTS_DIR
FIGURES = RESULTS / "figures"
EXPECTED = {
    "whisper_small_zs": (0,), "whisper_large_zs": (0,), "xlsr_kk_zs": (0,), "mms_kk_zs": (0,),
    "gmm_hmm": (13, 42, 2026), "ctc": (13, 42, 2026), "whisper_small_ft": (13, 42, 2026),
    "mhubert_ft": (13, 42, 2026), "aed": (42,), "rnnt": (42,),
}
GROUPS = {
    "whisper_small_zs": "baseline", "whisper_large_zs": "baseline", "xlsr_kk_zs": "baseline", "mms_kk_zs": "baseline",
    "gmm_hmm": "baseline_trained_by_us", "ctc": "baseline_trained_by_us", "aed": "baseline_trained_by_us",
    "rnnt": "baseline_trained_by_us", "whisper_small_ft": "contribution", "mhubert_ft": "contribution",
}
LADDER = ("gmm_hmm", "ctc", "aed", "rnnt", "xlsr_kk_zs", "mms_kk_zs", "whisper_small_zs", "whisper_large_zs",
          "whisper_small_ft", "mhubert_ft")
# Fine-tuned models inherit the status of their base model card when the run config has no ksc_training note.
INHERITED_CONTAMINATION = {"whisper_small_ft": "not stated (openai/whisper-small card)",
                           "mhubert_ft": "not stated (mHuBERT-147 card)"}
ABLATIONS = ("ctc-mfcc", "ctc-noaug")
FINAL_ONLY_SPLITS = data_io.FINAL_SPLITS
REGIME_SPLITS = ("test", "test_talkshow", "test_read")
SCALARS = ("wer", "cer", "cs_wer", "mono_wer", "ru_word_acc", "kz_word_acc")
SATURATED_WER = 0.9
TOLERANCE = 1e-9
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
RUN_FILE = re.compile(r"^(?P<model>.+)_seed(?P<seed>\d+)\.json$")
BENCHMARK_COLUMNS = ["model", "group", "checkpoint", "revision", "split", "wer", "wer_std", "wer_ci", "cer", "cs_wer",
                     "mono_wer", "ru_word_acc", "kz_word_acc", "n_seeds", "train_time", "contamination", "compared_with", "p_lower_wer"]
ABLATION_COLUMNS = ["model", "split", "wer", "wer_std", "wer_ci", "cer", "cs_wer", "mono_wer", "n_seeds",
                    "delta_wer_vs_ctc"]
ERROR_SPLITS = ("test", "test_talkshow")
ERROR_BASELINE = ("whisper_small_zs", 0)
ERROR_SEED = 42
# Kazakh-specific letters and the Russian letters they are confused with.
KZ_CONFUSIONS = str.maketrans("қғңүұөәіһ", "кгнууоаих")
SUFFIX_SPAN = 4
MIN_STEM = 3
ERROR_TYPES = ("ru_mix_substitution", "ru_mix_deletion", "kz_suffix_only", "kz_letter_confusion",
               "other_substitution", "other_deletion", "insertion")
N_EXAMPLES = 20
EXAMPLES_DIR = Path(tempfile.gettempdir()) / "aralas_errors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true",
                        help="include test, test_talkshow and test_read and write the tables and figures")
    parser.add_argument("--errors", action="store_true",
                        help="with --final: count error types on test and test_talkshow for the best model and "
                             "whisper_small_zs and dump examples to the system temp directory")
    args = parser.parse_args()
    if args.errors and not args.final:
        parser.error("--errors reads the final splits; add --final")
    return args


def discover_runs() -> dict[tuple[str, int], dict]:
    """Read every results/<model>_seed<seed>.json keyed by (model, seed)."""
    runs: dict[tuple[str, int], dict] = {}
    for path in sorted(RESULTS.glob("*.json")):
        match = RUN_FILE.match(path.name)
        if match:
            runs[(match["model"], int(match["seed"]))] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def missing_report(runs: dict, splits: tuple[str, ...]) -> list[str]:
    """List the expected runs, and the splits of present runs, that are absent."""
    lines = []
    for model, seeds in EXPECTED.items():
        for seed in seeds:
            if (model, seed) not in runs:
                lines.append(f"{model} seed {seed}: no run file")
                continue
            absent = [s for s in splits if s not in runs[(model, seed)]["splits"]]
            if absent:
                lines.append(f"{model} seed {seed}: no scores for {', '.join(absent)}")
    return lines


def read_hyps(model: str, seed: int, split: str, records: list[dict], final: bool) -> list[str]:
    """Load one prediction file, check it covers the split, and return hypotheses in split order."""
    path = RESULTS / "preds" / f"{model}_seed{seed}_{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"error: {path.name} is missing but {model}_seed{seed}.json reports {split}")
    preds = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics.validate_preds(split, preds, final=final)
    by_id = {p["id"]: p["hyp"] for p in preds}
    return [by_id[r["id"]] for r in records]


def differs(stored: object, fresh: object) -> bool:
    if stored is None or fresh is None:
        return stored is not fresh
    return not math.isclose(stored, fresh, rel_tol=0, abs_tol=TOLERANCE)


def check_stored(model: str, seed: int, split: str, stored: dict, fresh: dict) -> list[str]:
    """Describe every stored metric that the rescoring does not reproduce."""
    return [f"{model} seed {seed} {split} {key}: stored {stored[key]!r}, recomputed {fresh[key]!r}"
            for key in stored if key in fresh and differs(stored[key], fresh[key])]


def score_all(runs: dict, records: dict[str, list[dict]], final: bool) -> dict[tuple[str, int, str], dict]:
    """Rescore every run on every split it reports, failing if a stored number differs."""
    scores: dict[tuple[str, int, str], dict] = {}
    problems: list[str] = []
    for (model, seed), run in runs.items():
        for split, stored in run["splits"].items():
            if split not in records:
                continue
            hyps = read_hyps(model, seed, split, records[split], final)
            fresh = metrics.evaluate(records[split], hyps)
            problems += check_stored(model, seed, split, stored, fresh)
            scores[(model, seed, split)] = {"metrics": fresh, "hyps": hyps}
    if problems:
        sys.exit("error: stored numbers differ from the numbers recomputed from results/preds/:\n  " + "\n  ".join(problems))
    return scores


def mean(values: list[float | None]) -> float | None:
    return None if any(v is None for v in values) else sum(values) / len(values)


def pstd(values: list[float | None]) -> float | None:
    m = mean(values)
    return None if m is None else math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def contamination(model: str, run: dict, split: str) -> str:
    """Whether the model saw KSC training audio of this split: yes, no or unknown."""
    if GROUPS[model] == "baseline_trained_by_us":
        return "no"
    note = run["config"].get("ksc_training") or INHERITED_CONTAMINATION.get(model)
    if note is None:
        return "unknown"
    if note.startswith("yes"):
        return "yes" if split == "test_read" else "no"
    return "unknown"


def aggregate(runs: dict, scores: dict, records: dict[str, list[dict]], model: str, split: str) -> dict | None:
    """Mean and population std over the seeds of one model and split, or None without any run."""
    seeds = sorted(s for (m, s, sp) in scores if m == model and sp == split)
    if not seeds:
        return None
    parts = [scores[(model, s, split)] for s in seeds]
    out: dict = {k: mean([p["metrics"][k] for p in parts]) for k in SCALARS}
    out["wer_std"] = pstd([p["metrics"]["wer"] for p in parts])
    out["wer_ci"] = metrics.pooled_bootstrap_ci(records[split], [p["hyps"] for p in parts])
    out["n_seeds"] = len(seeds)
    out["train_time"] = mean([runs[(model, s)]["train_time_s"] for s in seeds])
    out["run"] = runs[(model, seeds[0])]
    return out


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def fmt_ci(ci: list[float] | None) -> str:
    return "" if ci is None else f"{ci[0]:.4f}-{ci[1]:.4f}"


def benchmark_rows(agg: dict, splits: tuple[str, ...], pairs: dict) -> list[dict]:
    rows = []
    for model in LADDER:
        for split in splits:
            a = agg.get((model, split))
            if a is None:
                continue
            run = a["run"]
            rows.append({
                "model": model, "group": GROUPS[model], "checkpoint": run["checkpoint"] or "",
                "revision": run["revision"] or "", "split": split, "wer": fmt(a["wer"]), "wer_std": fmt(a["wer_std"]),
                "wer_ci": fmt_ci(a["wer_ci"]), "cer": fmt(a["cer"]), "cs_wer": fmt(a["cs_wer"]),
                "mono_wer": fmt(a["mono_wer"]), "ru_word_acc": fmt(a["ru_word_acc"]),
                "kz_word_acc": fmt(a["kz_word_acc"]), "n_seeds": a["n_seeds"],
                "train_time": f"{a['train_time']:.1f}", "contamination": contamination(model, run, split),
                "compared_with": ";".join(c for c, _ in pairs.get((model, split), [])),
                "p_lower_wer": ";".join(f"{p:.3f}" for _, p in pairs.get((model, split), [])),
            })
    return rows


def ablation_rows(agg: dict, splits: tuple[str, ...]) -> list[dict]:
    rows = []
    for model in ("ctc",) + ABLATIONS:
        for split in splits:
            a, base = agg.get((model, split)), agg.get(("ctc", split))
            if a is None or base is None:
                continue
            delta = None if model == "ctc" else a["wer"] - base["wer"]
            rows.append({"model": model, "split": split, "wer": fmt(a["wer"]), "wer_std": fmt(a["wer_std"]),
                         "wer_ci": fmt_ci(a["wer_ci"]), "cer": fmt(a["cer"]), "cs_wer": fmt(a["cs_wer"]),
                         "mono_wer": fmt(a["mono_wer"]), "n_seeds": a["n_seeds"], "delta_wer_vs_ctc": fmt(delta)})
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_pairs(present: list[str]) -> list[tuple[str, str]]:
    """Neighbours on the ladder among present models, plus whisper_small_ft against whisper_small_zs."""
    ordered = [m for m in LADDER if m in present]
    pairs = list(zip(ordered, ordered[1:]))
    extra = ("whisper_small_zs", "whisper_small_ft")
    if all(m in present for m in extra) and extra not in pairs:
        pairs.append(extra)
    return pairs


def paired_table(scores: dict, records: dict[str, list[dict]]) -> dict[tuple[str, str], list[tuple[str, float]]]:
    """For each model and test split, (other model, P(model has lower seed-averaged WER than other)) per pair."""
    def seeds_of(model: str, split: str) -> list[list[str]]:
        return [v["hyps"] for (m, _, sp), v in sorted(scores.items()) if m == model and sp == split]

    present = sorted({m for (m, _, _) in scores})
    table: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for first, second in paired_pairs(present):
        for split in REGIME_SPLITS:
            a, b = seeds_of(first, split), seeds_of(second, split)
            if a and b:
                p = metrics.pooled_paired_bootstrap(records[split], b, a)
                table.setdefault((second, split), []).append((first, p))
    return table


def _labels(ax: plt.Axes, bars) -> None:
    ax.bar_label(bars, labels=[f"{b.get_height():.2f}" if b.get_height() == b.get_height() else "" for b in bars],
                 fontsize=7, padding=1)


def _bar_panel(ax: plt.Axes, models: list[str], series: dict[str, list[float | None]], ylabel: str) -> None:
    width = 0.8 / len(series)
    for i, (name, values) in enumerate(series.items()):
        xs = [x + (i - (len(series) - 1) / 2) * width for x in range(len(models))]
        heights = [float("nan") if v is None else v for v in values]
        _labels(ax, ax.bar(xs, heights, width, label=name, color=COLORS[i % len(COLORS)]))
    ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.12 * max([v for vs in series.values() for v in vs if v is not None] or [1.0]))
    if len(series) > 1:
        ax.legend(fontsize=8)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, format="svg", metadata={"Date": None})
    plt.close(fig)


def _models_with(agg: dict, splits: tuple[str, ...]) -> list[str]:
    return [m for m in LADDER if any((m, s) in agg for s in splits)]


def _saturated(agg: dict, models: list[str], splits: tuple[str, ...]) -> bool:
    return any(agg[(m, s)]["wer"] >= SATURATED_WER for m in models for s in splits if (m, s) in agg)


def _grouped_figure(agg: dict, name: str, splits: tuple[str, ...], title: str) -> None:
    """Grouped bars of WER per model and split, with CER in a second panel where WER saturates."""
    models = _models_with(agg, splits)
    if not models:
        return
    metric_list = ["wer", "cer"] if _saturated(agg, models, splits) else ["wer"]
    fig, axes = plt.subplots(1, len(metric_list), figsize=(6.5 * len(metric_list), 4.2), squeeze=False)
    for ax, metric in zip(axes[0], metric_list):
        series = {s: [agg[(m, s)][metric] if (m, s) in agg else None for m in models] for s in splits}
        _bar_panel(ax, models, series, metric.upper())
        ax.set_title(f"{title}: {metric.upper()}", fontsize=10)
    _save(fig, name)


def figure_ladder(agg: dict) -> None:
    _grouped_figure(agg, "ladder_wer.svg", ("test",), "Model ladder on test")


def figure_regimes(agg: dict) -> None:
    _grouped_figure(agg, "regimes.svg", REGIME_SPLITS, "Speech regimes")


def figure_cs_gap(agg: dict) -> None:
    """Code-switched against monolingual WER per model on test."""
    models = [m for m in LADDER if (m, "test") in agg]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    series = {"code-switched utterances": [agg[(m, "test")]["cs_wer"] for m in models],
              "monolingual utterances": [agg[(m, "test")]["mono_wer"] for m in models]}
    _bar_panel(ax, models, series, "WER")
    ax.set_title("Code-switching gap on test: WER", fontsize=10)
    _save(fig, "cs_gap.svg")


def figure_read_vs_spont(agg: dict) -> None:
    """Read-speech against spontaneous-speech scores per model, with CER where WER saturates."""
    models = [m for m in LADDER if (m, "test") in agg and (m, "test_read") in agg]
    if not models:
        return
    metric_list = ["wer", "cer"] if _saturated(agg, models, ("test", "test_read")) else ["wer"]
    fig, axes = plt.subplots(1, len(metric_list), figsize=(5.5 * len(metric_list), 4.8), squeeze=False)
    for ax, metric in zip(axes[0], metric_list):
        xs, ys = [agg[(m, "test_read")][metric] for m in models], [agg[(m, "test")][metric] for m in models]
        top = 1.15 * max(xs + ys)
        ax.plot([0, top], [0, top], color="#999999", linewidth=1, linestyle="--")
        ax.scatter(xs, ys, color=COLORS[0], zorder=3)
        for m, x, y in zip(models, xs, ys):
            ax.annotate(f"{m} ({x:.2f}, {y:.2f})", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel(f"{metric.upper()} on read speech (test_read)")
        ax.set_ylabel(f"{metric.upper()} on spontaneous speech (test)")
        ax.set_title(f"Read against spontaneous: {metric.upper()}", fontsize=10)
    _save(fig, "read_vs_spont.svg")


def best_model(agg: dict) -> str:
    """The model with the lowest mean test WER among the benchmark models."""
    present = [m for m in LADDER if (m, "test") in agg]
    if not present:
        sys.exit("error: --errors needs at least one run with test scores")
    return min(present, key=lambda m: agg[(m, "test")]["wer"])


def error_seed(scores: dict, model: str) -> int:
    """Seed 42 when the model has it, otherwise its only seed (zero-shot models use seed 0)."""
    seeds = sorted({s for (m, s, sp) in scores if m == model and sp in ERROR_SPLITS})
    if not seeds:
        sys.exit(f"error: --errors found no test or test_talkshow scores for {model}")
    return ERROR_SEED if ERROR_SEED in seeds else seeds[0]


def is_suffix_only(ref: str, hyp: str) -> bool:
    """True if the words share a stem and differ only within the last SUFFIX_SPAN characters."""
    stem = len(os.path.commonprefix([ref, hyp]))
    return stem >= MIN_STEM and stem >= max(len(ref), len(hyp)) - SUFFIX_SPAN


def substitution_type(ref: str, hyp: str, tag: str) -> str:
    """Classify one substituted reference word by its language tag and its spelling."""
    if tag in metrics.RU_TAGS:
        return "ru_mix_substitution"
    if ref.translate(KZ_CONFUSIONS) == hyp.translate(KZ_CONFUSIONS):
        return "kz_letter_confusion"
    if is_suffix_only(ref, hyp):
        return "kz_suffix_only"
    return "other_substitution"


def utterance_errors(record: dict, hyp: str) -> list[tuple[str, str, str]]:
    """Every word error of one utterance as (type, reference word, hypothesis word)."""
    ref_words, hyp_words, tags = record["text"].split(), hyp.split(), record["lang_tags"]
    if not hyp_words:
        return [("ru_mix_deletion" if t in metrics.RU_TAGS else "other_deletion", w, "") for w, t in zip(ref_words, tags)]
    errors = []
    for chunk in jiwer.process_words(record["text"], hyp).alignments[0]:
        refs = range(chunk.ref_start_idx, chunk.ref_end_idx)
        if chunk.type == "substitute":
            errors += [(substitution_type(ref_words[i], hyp_words[j], tags[i]), ref_words[i], hyp_words[j])
                       for i, j in zip(refs, range(chunk.hyp_start_idx, chunk.hyp_end_idx))]
        elif chunk.type == "delete":
            errors += [("ru_mix_deletion" if tags[i] in metrics.RU_TAGS else "other_deletion", ref_words[i], "")
                       for i in refs]
        elif chunk.type == "insert":
            errors += [("insertion", "", hyp_words[j]) for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx)]
    return errors


def error_counts(records: list[dict], hyps: list[str]) -> tuple[Counter, dict[str, list[dict]]]:
    """Count error types over a split and keep the first examples of each type in record order."""
    counts: Counter = Counter()
    examples: dict[str, list[dict]] = {t: [] for t in ERROR_TYPES}
    for record, raw in zip(records, hyps):
        hyp = data_io.normalize_text(raw)
        for kind, ref_word, hyp_word in utterance_errors(record, hyp):
            counts[kind] += 1
            if len(examples[kind]) < N_EXAMPLES:
                examples[kind].append({"id": record["id"], "ref_word": ref_word, "hyp_word": hyp_word,
                                       "ref": record["text"], "hyp": hyp})
    return counts, examples


def print_error_counts(model: str, seed: int, split: str, counts: Counter, n_words: int, word_edits: float) -> None:
    """Print the count and the share of all word errors for every error type."""
    total = sum(counts.values())
    print(f"\n{model} seed {seed} {split}: {total} word errors over {n_words} reference words "
          f"(WER numerator from metrics: {word_edits:.0f})")
    for kind in ERROR_TYPES:
        share = counts[kind] / total if total else 0.0
        print(f"  {kind:<22}{counts[kind]:>7}  {share:.4f}")
    other = counts["other_substitution"] + counts["other_deletion"] + counts["insertion"]
    print(f"  {'other (all three)':<22}{other:>7}  {(other / total if total else 0.0):.4f}")


def dump_examples(model: str, seed: int, split: str, counts: Counter, examples: dict[str, list[dict]]) -> Path:
    """Write the examples of the three largest error types outside the repository."""
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    path = EXAMPLES_DIR / f"{model}_seed{seed}_{split}.jsonl"
    largest = [kind for kind, _ in counts.most_common(3)]
    lines = [json.dumps({"type": kind, **ex}, ensure_ascii=False) for kind in largest for ex in examples[kind]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_errors(agg: dict, scores: dict, records: dict[str, list[dict]]) -> None:
    """Error-type counts on test and test_talkshow for the best model and whisper_small_zs."""
    best = best_model(agg)
    targets = [(best, error_seed(scores, best))]
    if ERROR_BASELINE[0] != best and (ERROR_BASELINE[0], ERROR_BASELINE[1], "test") in scores:
        targets.append(ERROR_BASELINE)
    print(f"\nError analysis: best model by mean test WER is {best}")
    for model, seed in targets:
        for split in ERROR_SPLITS:
            if (model, seed, split) not in scores:
                print(f"\n{model} seed {seed} {split}: no predictions, skipped")
                continue
            score = scores[(model, seed, split)]
            counts, examples = error_counts(records[split], score["hyps"])
            edits = score["metrics"]["wer"] * score["metrics"]["n_ref_words"]
            print_error_counts(model, seed, split, counts, score["metrics"]["n_ref_words"], edits)
            print(f"  examples of the three largest types: {dump_examples(model, seed, split, counts, examples)}")


def main() -> None:
    args = parse_args()
    splits = ("dev",) + (FINAL_ONLY_SPLITS if args.final else ())
    runs = discover_runs()
    missing = missing_report(runs, splits)
    print(f"Missing ({len(missing)}):" if missing else "Missing: nothing")
    for line in missing:
        print(f"  {line}")
    records = {s: data_io.load_split(s, final=args.final) for s in splits}
    scores = score_all(runs, records, args.final)
    print(f"Rescored {len(scores)} run-split pairs; every stored number matches.")
    agg = {(m, s): a for m in set(EXPECTED) | set(ABLATIONS) for s in splits
           if (a := aggregate(runs, scores, records, m, s))}
    if args.errors:
        run_errors(agg, scores, records)
        return
    rows = benchmark_rows(agg, splits, paired_table(scores, records) if args.final else {})
    if not args.final:
        for r in rows:
            print(f"  {r['model']:<18}{r['split']:<8} wer {r['wer']} +- {r['wer_std']} ci {r['wer_ci']} cer {r['cer']} "
                  f"kz_acc {r['kz_word_acc']} ru_acc {r['ru_word_acc']}")
        print("Dev-only run: nothing written. Use --final for benchmark.csv, ablations.csv and the figures.")
        return
    write_csv(RESULTS / "benchmark.csv", BENCHMARK_COLUMNS, rows)
    ablations = ablation_rows(agg, splits)
    if any(r["model"] in ABLATIONS for r in ablations):
        write_csv(RESULTS / "ablations.csv", ABLATION_COLUMNS, ablations)
    else:
        print("No ablation runs yet: ablations.csv not written.")
    for figure in (figure_ladder, figure_regimes, figure_cs_gap, figure_read_vs_spont):
        figure(agg)
    print(f"Wrote benchmark.csv ({len(rows)} rows) and figures to {RESULTS}")


if __name__ == "__main__":
    main()
