"""Import the spontaneous part of KSC2 and the KSC read test set into data/raw, with the program-level split plan."""
import argparse
import gzip
import hashlib
import io
import json
import random
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SEED = 2026
TRAIN_HOURS = 32.0
TALKSHOW_HOURS = 6.0
SAMPLE_RATE = 16000
# libsndfile maps this compression level to about 24 kbps for 16 kHz mono speech.
OPUS_COMPRESSION_LEVEL = 0.93
OFFICIAL_SPLITS = {"Train": "train", "Dev": "dev", "Test": "test"}
SPONTANEOUS = ("podcasts", "radio", "talkshow")
SEGMENT_NAME = re.compile(r"(\d+)_(\d+)_(\d+)")
SPLIT_RULE = (
    "podcasts: the official KSC2 split, which is already disjoint by program; "
    "radio (one program): all episodes are train candidates; "
    "talkshow (one program): whole episodes in seeded random order until >= 6 h form test_talkshow, the rest is unused; "
    "read: the KSC read-speech test set (Test/crowdsourced) is test_read; "
    "train: whole episodes of radio and of the podcast train programs in seeded random order until >= 32 h."
)
ARCHIVE = {
    "corpus": "ISSAI KSC2 (Kazakh Speech Corpus 2)",
    "url": "https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2",
    "parts": {f"ISSAI_KSC2.tar.gz.part{p}": 8388608000 for p in ("aa", "ab", "ac", "ad", "ae", "af", "ag", "ah", "ai")}
    | {"ISSAI_KSC2.tar.gz.partaj": 5311650212},
    "paper": "Mussakhojayeva, Khassanov, Varol (2022). KSC2: An Industrial-Scale Open-Source Kazakh Speech Corpus. "
             "Interspeech 2022, doi:10.21437/Interspeech.2022-421",
    "licenses": {
        "huggingface_card": "license: mit (https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2)",
        "github_recipe": "CC-BY-4.0 (https://github.com/IS2AI/Kazakh_ASR)",
        "ksc_openslr_102": "Attribution 4.0 International (CC BY 4.0) (https://openslr.org/102/)",
    },
    "read_test_set": "Test/crowdsourced of KSC2 equals the test set of KSC v1.1 (Khassanov et al., 2021, arXiv:2009.10334).",
}


def flac_seconds(path: Path) -> float:
    """Duration of a FLAC file from its STREAMINFO block."""
    with open(path, "rb") as f:
        if f.read(4) != b"fLaC":
            raise ValueError(f"{path}: not a FLAC file")
        size = int.from_bytes(f.read(4)[1:4], "big")
        info = f.read(size)
    rate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
    total = ((info[13] & 0x0F) << 32) | int.from_bytes(info[14:18], "big")
    return total / rate


def parse_segment(source: str, official: str, path: Path) -> dict:
    """Ids, program and episode of one segment from its folder and file name."""
    if source == "read":
        return {"id": f"read_{path.stem}", "source": "read", "program": "read", "episode": "read",
                "orig_split": official, "flac": path}
    m = SEGMENT_NAME.fullmatch(path.stem)
    if not m:
        raise ValueError(f"{path}: name does not match <program>_<episode>_<segment>")
    program = f"{source}_{m.group(1)}"
    return {"id": f"{source}_{path.stem}", "source": source, "program": program,
            "episode": f"{program}_{m.group(2)}", "orig_split": official, "flac": path}


def scan(ksc2: Path) -> list[dict]:
    """Every segment we may use, with its duration; fails loudly on missing folders or id clashes."""
    folders = [(src, off) for src in SPONTANEOUS for off in OFFICIAL_SPLITS] + [("read", "Test")]
    segments = []
    for source, official in folders:
        folder = ksc2 / official / ("crowdsourced" if source == "read" else source)
        if not folder.is_dir():
            raise FileNotFoundError(f"expected folder is missing: {folder}")
        segments += [parse_segment(source, OFFICIAL_SPLITS[official], p) for p in sorted(folder.glob("*.flac"))]
    with ThreadPoolExecutor(32) as pool:
        for seg, sec in zip(segments, pool.map(lambda s: flac_seconds(s["flac"]), segments)):
            seg["duration"] = sec
    seen: dict[str, Path] = {}
    for seg in segments:
        if seg["id"] in seen:
            raise ValueError(f"id clash {seg['id']}: {seen[seg['id']]} and {seg['flac']}")
        seen[seg["id"]] = seg["flac"]
    return segments


def hours(segments: list[dict]) -> float:
    return sum(s["duration"] for s in segments) / 3600


def take_episodes(segments: list[dict], target_hours: float, rng: random.Random) -> list[str]:
    """Whole episodes in seeded random order until their total reaches target_hours."""
    by_episode = defaultdict(float)
    for s in segments:
        by_episode[s["episode"]] += s["duration"] / 3600
    order = sorted(by_episode)
    rng.shuffle(order)
    chosen, total = [], 0.0
    for ep in order:
        if total >= target_hours:
            break
        chosen.append(ep)
        total += by_episode[ep]
    return sorted(chosen)


def make_plan(segments: list[dict]) -> dict:
    """Apply the split rule and return the plan that clean_dataset.py reads."""
    podcast_splits = defaultdict(set)
    for s in segments:
        if s["source"] == "podcasts":
            podcast_splits[s["program"]].add(s["orig_split"])
    mixed = {p: sorted(v) for p, v in podcast_splits.items() if len(v) > 1}
    if mixed:
        raise ValueError(f"podcast programs in more than one official split: {mixed}")
    rng = random.Random(SEED)
    talkshow = take_episodes([s for s in segments if s["source"] == "talkshow"], TALKSHOW_HOURS, rng)
    pool = [s for s in segments if s["source"] == "radio"
            or (s["source"] == "podcasts" and s["orig_split"] == "train")]
    train = take_episodes(pool, TRAIN_HOURS, rng)
    programs = {"read": "test_read", "radio_01": "train"}
    programs |= {p: next(iter(v)) for p, v in podcast_splits.items()}
    programs["talkshow_01"] = "test_talkshow"
    return {"seed": SEED, "rule": SPLIT_RULE, "train_hours_target": TRAIN_HOURS,
            "talkshow_hours_target": TALKSHOW_HOURS, "programs": dict(sorted(programs.items())),
            "train_episodes": train, "test_talkshow_episodes": talkshow, "manual_changes": []}


def assign(segments: list[dict], plan: dict) -> list[dict]:
    """Segments selected by the plan, each with its split."""
    train_eps, talk_eps = set(plan["train_episodes"]), set(plan["test_talkshow_episodes"])
    chosen = []
    for s in segments:
        split = plan["programs"].get(s["program"])
        if split is None:
            raise ValueError(f"program {s['program']} has no split in the plan")
        if split == "train" and s["episode"] not in train_eps:
            continue
        if split == "test_talkshow" and s["episode"] not in talk_eps:
            continue
        chosen.append({**s, "split": split})
    return chosen


def print_plan(segments: list[dict], chosen: list[dict]) -> None:
    table = defaultdict(lambda: defaultdict(lambda: [0, 0.0, set()]))
    for s in segments:
        cell = table[s["program"]][s["orig_split"]]
        cell[0] += 1
        cell[1] += s["duration"] / 3600
        cell[2].add(s["episode"])
    print("program        official split: utterances / hours / episodes")
    for program in sorted(table):
        cells = "  ".join(f"{sp}={c[0]}/{c[1]:.2f}h/{len(c[2])}ep" for sp, c in sorted(table[program].items()))
        print(f"  {program:14} {cells}")
    print("\nour split      utterances  hours  programs  episodes")
    by_split = defaultdict(list)
    for s in chosen:
        by_split[s["split"]].append(s)
    for split in ("train", "dev", "test", "test_talkshow", "test_read"):
        rows = by_split[split]
        print(f"  {split:14} {len(rows):10} {hours(rows):6.2f} {len({r['program'] for r in rows}):9} "
              f"{len({r['episode'] for r in rows}):9}")
    total = hours(chosen)
    print(f"\ntotal {len(chosen)} segments, {total:.2f} h, about {total * 24e3 * 3600 / 8 / 1e9:.2f} GB of Opus")


def encode(job: tuple[str, str]) -> dict:
    """Read one FLAC and write it as Opus; an undecodable FLAC is reported, not written (clean_dataset rule C1)."""
    flac, out = job
    raw = Path(flac).read_bytes()
    result = {"out": out, "flac_sha256": hashlib.sha256(raw).hexdigest()}
    try:
        wave, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    except (RuntimeError, sf.LibsndfileError) as err:
        return {**result, "decode_error": str(err), "sample_rate": None, "channels": None}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, wave, rate, format="OGG", subtype="OPUS", compression_level=OPUS_COMPRESSION_LEVEL)
    pcm, _ = sf.read(out, dtype="int16")
    return {**result, "decode_error": None, "sample_rate": rate, "channels": wave.shape[1],
            "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest()}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def write_transcripts(chosen: list[dict], out_dir: Path, jobs: int) -> dict[str, str]:
    """Write Opus audio and per-source transcript files; return audio path -> decoded PCM sha256."""
    work = [(str(s["flac"]), str(out_dir / "audio" / s["source"] / f"{s['id']}.opus")) for s in chosen]
    pcm_hash, rows = {}, defaultdict(list)
    with ProcessPoolExecutor(jobs) as pool:
        results = list(pool.map(encode, work, chunksize=64))
    for s, res in zip(chosen, results):
        rel = Path(res["out"]).relative_to(out_dir).as_posix()
        if res["decode_error"] is None:
            pcm_hash[rel] = res["pcm_sha256"]
        else:
            print(f"warning: {s['flac']} is not decodable ({res['decode_error']}); kept as a row without audio")
        rows[s["source"]].append({
            "id": s["id"], "source": s["source"], "program": s["program"], "episode": s["episode"],
            "orig_split": s["orig_split"], "audio": f"data/raw/{rel}" if res["decode_error"] is None else None,
            "duration": round(s["duration"], 4), "sample_rate": res["sample_rate"], "channels": res["channels"],
            "text_orig": read_text(s["flac"].with_suffix(".txt")), "flac_sha256": res["flac_sha256"],
            "decode_error": res["decode_error"],
        })
    (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    for source, items in rows.items():
        lines = [json.dumps(r, ensure_ascii=False) for r in sorted(items, key=lambda r: r["id"])]
        (out_dir / "transcripts" / f"{source}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pcm_hash


def write_all_text(ksc2: Path, out_dir: Path) -> int:
    """Every transcript of the archive, text only, as a reproducible gzip file."""
    files = sorted(ksc2.glob("*/*/*.txt"))
    with ThreadPoolExecutor(32) as pool:
        texts = list(pool.map(read_text, files))
    lines = [json.dumps({"id": f"{p.parent.name}_{p.stem}", "source": p.parent.name,
                         "orig_split": OFFICIAL_SPLITS[p.parent.parent.name], "text": t}, ensure_ascii=False)
             for p, t in zip(files, texts)]
    path = out_dir / "text" / "ksc2_all_text.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(("\n".join(lines) + "\n").encode("utf-8"))
    return len(lines)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_checksums(out_dir: Path, pcm_hash: dict[str, str]) -> None:
    """sha256 of every text file and of the decoded PCM of every Opus file (Ogg serial numbers are random)."""
    lines = []
    for p in sorted(out_dir.rglob("*")):
        rel = p.relative_to(out_dir).as_posix()
        if p.is_dir() or rel == "checksums.txt":
            continue
        if rel in pcm_hash:
            lines.append(f"{pcm_hash[rel]}  pcm:{rel}")
        else:
            lines.append(f"{file_sha256(p)}  {rel}")
    (out_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(ksc2: Path, out_dir: Path, jobs: int, plan: dict, segments: list[dict]) -> None:
    chosen = assign(segments, plan)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "split_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pcm_hash = write_transcripts(chosen, out_dir, jobs)
    n_text = write_all_text(ksc2, out_dir)
    sources = {**ARCHIVE, "retrieved": "2026-11-18/19", "segments_imported": len(chosen),
               "transcripts_in_all_text": n_text, "opus_compression_level": OPUS_COMPRESSION_LEVEL}
    (out_dir / "SOURCES.json").write_text(json.dumps(sources, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_checksums(out_dir, pcm_hash)
    size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file()) / 1e9
    print(f"wrote {len(chosen)} segments and {n_text} transcripts to {out_dir}, {size:.2f} GB")


def check(ksc2: Path, jobs: int, segments: list[dict]) -> None:
    """Rebuild into a temporary folder and compare every checksum with data/raw/checksums.txt."""
    plan = json.loads((RAW_DIR / "split_plan.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "raw"
        build(ksc2, out, jobs, plan, segments)
        a = (RAW_DIR / "checksums.txt").read_text(encoding="utf-8").splitlines()
        b = (out / "checksums.txt").read_text(encoding="utf-8").splitlines()
    diff = sorted(set(a) ^ set(b))
    if diff:
        sys.exit(f"--check failed: {len(diff)} checksum lines differ, first: {diff[:5]}")
    print(f"--check passed: {len(a)} checksums identical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ksc2", type=Path, required=True, help="path to the extracted ISSAI_KSC2 folder")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="print the split plan and write nothing")
    mode.add_argument("--write", action="store_true", help="write data/raw from the plan")
    mode.add_argument("--check", action="store_true", help="rebuild into a temporary folder and compare checksums")
    parser.add_argument("--jobs", type=int, default=8, help="parallel encoder processes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segments = scan(args.ksc2)
    if args.check:
        check(args.ksc2, args.jobs, segments)
        return
    plan = make_plan(segments)
    print_plan(segments, assign(segments, plan))
    if args.write:
        build(args.ksc2, RAW_DIR, args.jobs, plan, segments)


if __name__ == "__main__":
    main()
