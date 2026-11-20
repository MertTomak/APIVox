"""Clean data/raw into data/processed: cleaning rules C1-C9, split manifests, code-switching tags, lexicons and statistics."""
import argparse
import csv
import filecmp
import hashlib
import json
import random
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import codeswitch, data_io  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
HUMAN_QC = ROOT / "results" / "human_qc.csv"
FIXTURES = ROOT / "tests" / "fixtures"

THRESHOLDS = {
    "min_duration_s": 0.5,
    "max_duration_s": 30.0,
    "rate_max_robust_z": 4.0,
    "clip_level": 0.999,
    "max_clipped_share": 0.001,
    "silence_top_db": 40.0,
    "min_speech_ratio": 0.3,
    "c9_min_words": 4,
}
CANDIDATE_SEED = 7
CANDIDATE_SAMPLE = 60
CANDIDATE_MARGIN = 0.10
CANDIDATE_NEAR_MAX = 40
EVAL_SPLITS = ("dev",) + data_io.FINAL_SPLITS
LOG_FIELDS = ("id", "split", "rule", "measured_value", "threshold", "action")


def measure(audio_path: str) -> dict:
    """Decode one clip and measure what the audio rules need."""
    import librosa
    try:
        wave, rate = sf.read(audio_path, dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as err:
        return {"error": f"unreadable: {err}"}
    channels = wave.shape[1]
    mono = wave[:, 0]
    if rate != data_io.SAMPLE_RATE or channels != 1 or len(mono) == 0:
        return {"error": f"rate {rate}, channels {channels}, samples {len(mono)}"}
    pcm = np.round(np.clip(mono, -1, 1) * 32767).astype(np.int16)
    voiced = librosa.effects.split(mono, top_db=THRESHOLDS["silence_top_db"])
    voiced_samples = int(sum(e - s for s, e in voiced))
    return {
        "duration": len(mono) / rate,
        "clipped_share": float(np.mean(np.abs(mono) >= THRESHOLDS["clip_level"])),
        "speech_ratio": voiced_samples / len(mono),
        "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
    }


def load_raw(raw_dir: Path) -> tuple[list[dict], dict]:
    """Raw transcript rows with their split from split_plan.json."""
    plan = json.loads((raw_dir / "split_plan.json").read_text(encoding="utf-8"))
    train_eps, talk_eps = set(plan["train_episodes"]), set(plan["test_talkshow_episodes"])
    rows = []
    for path in sorted((raw_dir / "transcripts").glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            split = plan["programs"].get(row["program"])
            if split is None:
                raise ValueError(f"{row['id']}: program {row['program']} is not in split_plan.json")
            if (split == "train" and row["episode"] not in train_eps) or \
               (split == "test_talkshow" and row["episode"] not in talk_eps):
                raise ValueError(f"{row['id']}: episode {row['episode']} is not selected for {split} in the plan")
            rows.append({**row, "split": split})
    return sorted(rows, key=lambda r: r["id"]), plan


def read_human_qc(path: Path) -> dict[str, str]:
    """id -> reason for every clip with verdict n (any checker)."""
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r.get("reason", "") for r in csv.DictReader(f) if r["verdict"].strip().lower() == "n"}


class Cleaner:
    """Applies the rules in order and remembers every removal."""

    def __init__(self) -> None:
        self.log: list[dict] = []
        self.removed: dict[str, str] = {}

    def drop(self, row: dict, rule: str, value, threshold, action: str = "removed") -> None:
        self.removed[row["id"]] = rule
        self.log.append({"id": row["id"], "split": row["split"], "rule": rule,
                         "measured_value": value, "threshold": threshold, "action": action})

    def alive(self, rows: list[dict]) -> list[dict]:
        return [r for r in rows if r["id"] not in self.removed]


def letters_per_second(row: dict) -> float:
    return len(row["text"].replace(" ", "")) / row["duration"]


def rate_bounds(train_rates: list[float], max_z: float) -> tuple[float, float]:
    """Letters-per-second bounds where the robust z-score of log rate (median, 1.4826 * MAD) stays within max_z."""
    logs = np.log(np.asarray(train_rates, dtype=float))
    median = float(np.median(logs))
    scale = 1.4826 * float(np.median(np.abs(logs - median)))
    if scale == 0:
        raise ValueError("speaking rates of train have zero spread; cannot set C4 bounds")
    return float(np.exp(median - max_z * scale)), float(np.exp(median + max_z * scale))


def apply_rules(rows: list[dict], measures: dict[str, dict], qc_rejects: dict[str, str]) -> tuple[Cleaner, dict]:
    """Run C1-C9 in order and return the cleaner and the thresholds that were computed from train."""
    t = THRESHOLDS
    c = Cleaner()
    for r in rows:
        m = measures[r["id"]]
        if "error" in m:
            c.drop(r, "C1", m["error"], "readable 16 kHz mono")
        elif not r["text_orig"].strip():
            c.drop(r, "C1", "empty transcript", "non-empty transcript")
        else:
            r["duration"] = round(m["duration"], 3)
    for r in c.alive(rows):
        if not t["min_duration_s"] <= r["duration"] <= t["max_duration_s"]:
            c.drop(r, "C2", r["duration"], f"[{t['min_duration_s']}, {t['max_duration_s']}] s")
    for r in c.alive(rows):
        if not r["text"]:
            c.drop(r, "C3", repr(r["text_orig"][:60]), "non-empty after normalize_text")
    lo, hi = rate_bounds([letters_per_second(r) for r in c.alive(rows) if r["split"] == "train"], t["rate_max_robust_z"])
    computed = {"rate_low": round(lo, 4), "rate_high": round(hi, 4)}
    for r in c.alive(rows):
        rate = letters_per_second(r)
        if not lo <= rate <= hi:
            c.drop(r, "C4", round(rate, 3), f"[{lo:.3f}, {hi:.3f}] letters/s (train log-rate robust z <= {t['rate_max_robust_z']})")
    for r in c.alive(rows):
        share = measures[r["id"]]["clipped_share"]
        if share > t["max_clipped_share"]:
            c.drop(r, "C5", round(share, 5), f"<= {t['max_clipped_share']}")
    for r in c.alive(rows):
        ratio = measures[r["id"]]["speech_ratio"]
        if ratio < t["min_speech_ratio"]:
            c.drop(r, "C6", round(ratio, 4), f">= {t['min_speech_ratio']}")
    first_by_hash: dict[str, str] = {}
    for r in c.alive(rows):
        h = measures[r["id"]]["pcm_sha256"]
        if h in first_by_hash:
            c.drop(r, "C7", f"same audio as {first_by_hash[h]}", "unique decoded audio")
        else:
            first_by_hash[h] = r["id"]
    for r in c.alive(rows):
        if r["id"] in qc_rejects:
            c.drop(r, "C8", qc_rejects[r["id"]] or "n", "listening verdict y")
    eval_texts = {r["text"] for r in c.alive(rows) if r["split"] in EVAL_SPLITS}
    for r in c.alive(rows):
        if r["split"] == "train" and r["text"] in eval_texts and len(r["text"].split()) >= t["c9_min_words"]:
            c.drop(r, "C9", r["text"], f"train text equal to a dev/test text with >= {t['c9_min_words']} words")
    return c, computed


def verbatim_overlap(rows: list[dict]) -> dict[str, int]:
    """How many dev/test utterances have a text that occurs verbatim in train."""
    train = {r["text"] for r in rows if r["split"] == "train"}
    ev = [r for r in rows if r["split"] in EVAL_SPLITS and r["text"]]
    return {"all": sum(r["text"] in train for r in ev),
            "min_4_words": sum(r["text"] in train and len(r["text"].split()) >= 4 for r in ev)}


def leakage(rows: list[dict]) -> dict[str, int]:
    splits_of = {"program": defaultdict(set), "episode": defaultdict(set)}
    for r in rows:
        for key in splits_of:
            splits_of[key][r[key]].add(r["split"])
    return {f"{key}s_in_more_than_one_split": sum(len(v) > 1 for v in d.values()) for key, d in splits_of.items()}


def summarize(rows: list[dict]) -> dict:
    tags = Counter(t for r in rows for t in r["lang_tags"])
    words = sum(len(r["lang_tags"]) for r in rows)
    n_cs = sum(r["cs"] for r in rows)
    return {
        "utterances": len(rows), "hours": round(sum(r["duration"] for r in rows) / 3600, 3),
        "programs": len({r["program"] for r in rows}), "episodes": len({r["episode"] for r in rows}),
        "words": words, "cs_utterances": n_cs, "cs_rate": round(n_cs / len(rows), 4) if rows else None,
        "ru_words": tags["ru"], "mix_words": tags["mix"], "amb_words": tags["amb"], "kz_words": tags["kz"],
    }


def manifest_record(r: dict) -> dict:
    return {"id": r["id"], "split": r["split"], "audio": r["audio"], "duration": r["duration"],
            "source": r["source"], "program": r["program"], "episode": r["episode"], "text": r["text"],
            "text_orig": r["text_orig"], "lang_tags": r["lang_tags"], "cs": r["cs"]}


def build(raw_dir: Path, out_dir: Path, jobs: int, qc_path: Path = HUMAN_QC) -> tuple[list[dict], dict]:
    """Write every file of data/processed into out_dir and return the kept rows and the computed thresholds."""
    rows, plan = load_raw(raw_dir)
    lex = codeswitch.build_lexicons(raw_dir / "text" / "ksc2_all_text.jsonl.gz")
    hesitations = 0
    for r in rows:
        r["text"] = data_io.normalize_text(r["text_orig"])
        r["lang_tags"] = codeswitch.tag_text(r["text"], lex)
        r["cs"] = data_io.is_cs(r["lang_tags"])
        hesitations += sum(bool(data_io.HESITATION.fullmatch(w)) for w in r["text_orig"].lower().split())
    with_audio = [r for r in rows if r["audio"] is not None]
    paths = [str(raw_dir.parent.parent / r["audio"]) for r in with_audio]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            results = list(pool.map(measure, paths, chunksize=64))
    else:
        results = [measure(p) for p in paths]
    measures = dict(zip((r["id"] for r in with_audio), results))
    measures |= {r["id"]: {"error": f"not decodable at import: {r['decode_error']}"} for r in rows if r["audio"] is None}
    overlap_before = verbatim_overlap(rows)
    cleaner, computed = apply_rules(rows, measures, read_human_qc(qc_path))
    kept = cleaner.alive(rows)
    for r in kept:
        data_io.validate(manifest_record(r))

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    codeswitch.write_lexicons(lex, out_dir / "lexicon")
    by_split = defaultdict(list)
    for r in kept:
        by_split[r["split"]].append(manifest_record(r))
    for split in data_io.SPLITS:
        lines = [json.dumps(r, ensure_ascii=False) for r in by_split[split]]
        (out_dir / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(cleaner.log)

    removed = defaultdict(Counter)
    for entry in cleaner.log:
        removed[entry["rule"]][entry["split"]] += 1
    stats = {
        "splits": {s: summarize([r for r in kept if r["split"] == s]) for s in data_io.SPLITS},
        "sources": {s: summarize([r for r in kept if r["source"] == s]) for s in data_io.SOURCES},
        "raw_utterances": {s: sum(r["split"] == s for r in rows) for s in data_io.SPLITS},
        "removed_per_rule": {rule: dict(sorted(c.items())) for rule, c in sorted(removed.items())},
        "removed_total": len(cleaner.log),
        "leakage": {**leakage(kept),
                    "eval_texts_verbatim_in_train_before_cleaning": overlap_before,
                    "eval_texts_verbatim_in_train_after_cleaning": verbatim_overlap(kept)},
        "hesitation_tokens_removed_by_normalization": hesitations,
        "lexicon_sizes": {"kz": len(lex.kz), "ru": len(lex.ru)},
        "thresholds": THRESHOLDS, "computed_thresholds": computed,
        "split_plan_seed": plan["seed"],
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return kept, computed


def print_removals(out_dir: Path) -> None:
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    print("rule  " + "  ".join(f"{s:>13}" for s in data_io.SPLITS))
    for rule, counts in stats["removed_per_rule"].items():
        print(f"{rule:5} " + "  ".join(f"{counts.get(s, 0):13}" for s in data_io.SPLITS))
    for split, s in stats["splits"].items():
        print(f"{split:14} {s['utterances']:6} utts {s['hours']:7.2f} h  cs={s['cs_utterances']} ({s['cs_rate']})")
    print("leakage:", stats["leakage"])


def candidates(rows: list[dict], computed: dict) -> list[str]:
    """A seeded sample of kept clips plus clips within CANDIDATE_MARGIN of the C4 thresholds."""
    rng = random.Random(CANDIDATE_SEED)
    sample = rng.sample(rows, min(CANDIDATE_SAMPLE, len(rows)))
    lo, hi = computed["rate_low"], computed["rate_high"]
    near = [r for r in rows
            if letters_per_second(r) <= lo * (1 + CANDIDATE_MARGIN) or letters_per_second(r) >= hi * (1 - CANDIDATE_MARGIN)]
    near = rng.sample(near, min(CANDIDATE_NEAR_MAX, len(near)))
    chosen = {r["id"]: ("near_threshold", r) for r in near}
    chosen |= {r["id"]: ("sample", r) for r in sample}
    return [f"{i}\t{kind}\t{r['split']}\t{r['audio']}" for i, (kind, r) in sorted(chosen.items())]


def write_fixtures(kept: list[dict]) -> None:
    """Eight short train radio clips and their manifest records for the unit tests."""
    pool = sorted((r for r in kept if r["split"] == "train" and r["source"] == "radio" and 2 <= r["duration"] <= 6),
                  key=lambda r: r["id"])
    chosen = random.Random(CANDIDATE_SEED).sample(pool, 8)
    for old in FIXTURES.glob("*.opus"):
        old.unlink()
    lines = []
    for i, r in enumerate(sorted(chosen, key=lambda r: r["id"]), start=1):
        name = f"fixture_{i:02d}.opus"
        shutil.copyfile(ROOT / r["audio"], FIXTURES / name)
        lines.append(json.dumps({**manifest_record(r), "split": "dev" if i > 6 else "train",
                                 "audio": f"tests/fixtures/{name}"}, ensure_ascii=False))
    (FIXTURES / "mini.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(jobs: int) -> None:
    """Rebuild data/processed into a temporary folder and compare it file by file."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "processed"
        build(RAW_DIR, out, jobs)
        mine = sorted(p.relative_to(PROCESSED_DIR).as_posix() for p in PROCESSED_DIR.rglob("*") if p.is_file())
        fresh = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
        if mine != fresh:
            sys.exit(f"--check failed: file lists differ: {sorted(set(mine) ^ set(fresh))}")
        bad = [f for f in mine if not filecmp.cmp(PROCESSED_DIR / f, out / f, shallow=False)]
    if bad:
        sys.exit(f"--check failed: {bad}")
    print(f"--check passed: {len(mine)} files identical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="rebuild into a temporary folder and compare with data/processed")
    mode.add_argument("--candidates", action="store_true", help="print the clips for the listening check and write nothing")
    parser.add_argument("--fixtures", action="store_true", help="also rewrite tests/fixtures from the cleaned train set")
    parser.add_argument("--jobs", type=int, default=8, help="parallel processes for audio measurements")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        check(args.jobs)
        return
    if args.candidates:
        with tempfile.TemporaryDirectory() as tmp:
            kept, computed = build(RAW_DIR, Path(tmp) / "processed", args.jobs)
        print("\n".join(candidates(kept, computed)))
        return
    kept, _ = build(RAW_DIR, PROCESSED_DIR, args.jobs)
    print_removals(PROCESSED_DIR)
    if args.fixtures:
        write_fixtures(kept)


if __name__ == "__main__":
    main()
