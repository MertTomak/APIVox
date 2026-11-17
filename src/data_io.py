"""Load and validate manifests, decode audio and normalize hypotheses.

CHARS is the output alphabet; CHAR2ID gives 0-based positions in CHARS. CTC models add 1 to
every id so that index 0 is the blank.
"""
import argparse
import json
import re
from pathlib import Path, PurePath

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_RATE = 16000

CHARS = " '._" + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
ID2CHAR = dict(enumerate(CHARS))

OPEN_SPLITS = ("train", "dev")
FINAL_SPLITS = ("test", "heldout", "test_kokoro", "test_human")
SPLITS = OPEN_SPLITS + FINAL_SPLITS
ENGINES = ("piper", "kokoro", "human")
STYLES = ("explicit", "natural", "terse")

FIELDS = {
    "id": (str,),
    "split": (str,),
    "audio": (str,),
    "duration": (int, float),
    "engine": (str,),
    "speaker": (str,),
    "library": (str,),
    "doc_path": (str,),
    "written": (str,),
    "spoken": (str, type(None)),
    "api": (str,),
    "style": (str, type(None)),
    "api_seen_in_train": (bool,),
}
CONTEXT_WORD = re.compile(r"[a-z']+")
NON_CHARS = re.compile(f"[^{re.escape(CHARS)}]")


def _fail(record: dict, field: str, problem: str) -> None:
    raise ValueError(f"record {record.get('id', '<no id>')!r}, field '{field}': {problem}")


def _check_keys_and_types(record: dict) -> None:
    for key in FIELDS.keys() - record.keys():
        _fail(record, key, "missing")
    for key in record.keys() - FIELDS.keys():
        _fail(record, key, "unexpected key")
    for key, types in FIELDS.items():
        value = record[key]
        if not isinstance(value, types) or (bool not in types and isinstance(value, bool)):
            _fail(record, key, f"expected {' or '.join(t.__name__ for t in types)}, got {type(value).__name__}")


def _check_api(record: dict) -> None:
    api = record["api"]
    if not api.strip():
        _fail(record, "api", "must not be empty or whitespace")
    if api[0] in ".'" or api[-1] in ".'":
        _fail(record, "api", f"{api!r} must not start or end with '.' or an apostrophe")
    if api.endswith("()"):
        _fail(record, "api", f"{api!r} must not end in '()'")


def _check_text(record: dict) -> None:
    written, api = record["written"], record["api"]
    bad = sorted({c for c in written if c not in CHAR2ID})
    if bad:
        _fail(record, "written", f"characters outside CHARS: {bad}")
    tokens = written.split(" ")
    if tokens.count(api) != 1:
        _fail(record, "api", f"{api!r} must occur exactly once as a whole token in written, found {tokens.count(api)}")
    for word in tokens:
        if word != api and not CONTEXT_WORD.fullmatch(word):
            _fail(record, "written", f"context word {word!r} is not lowercase a-z or apostrophe")


def _check_audio_path(record: dict) -> None:
    audio = PurePath(record["audio"])
    if audio.is_absolute() or ".." in audio.parts:
        _fail(record, "audio", f"{record['audio']!r} must be a repo-relative path without '..'")


def _check_engine_fields(record: dict) -> None:
    human = record["engine"] == "human"
    if human and record["spoken"] is not None:
        _fail(record, "spoken", "must be null for a human clip")
    if not human and record["spoken"] is None:
        _fail(record, "spoken", "must not be null for a synthesized clip")
    if not human and record["style"] is None:
        _fail(record, "style", "must not be null for a synthesized clip")


def validate(record: dict) -> None:
    """Raise ValueError naming the id and the field if a manifest record breaks the contract."""
    _check_keys_and_types(record)
    if record["split"] not in SPLITS:
        _fail(record, "split", f"unknown split {record['split']!r}")
    if record["engine"] not in ENGINES:
        _fail(record, "engine", f"unknown engine {record['engine']!r}")
    if record["style"] is not None and record["style"] not in STYLES:
        _fail(record, "style", f"unknown style {record['style']!r}")
    if record["duration"] <= 0:
        _fail(record, "duration", f"must be positive, got {record['duration']}")
    _check_audio_path(record)
    _check_api(record)
    _check_text(record)
    _check_engine_fields(record)


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as err:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({err})") from err
        if not isinstance(records[-1], dict):
            raise ValueError(f"{path}:{lineno}: expected a JSON object")
    return records


def load_split(name: str, final: bool = False) -> list[dict]:
    """Read and validate data/processed/<name>.jsonl in file order; final splits need final=True."""
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLITS}")
    if name in FINAL_SPLITS and not final:
        raise PermissionError(f"split {name!r} is only readable with final=True (--final)")
    path = PROCESSED_DIR / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    records = _read_jsonl(path)
    seen: set[str] = set()
    for record in records:
        validate(record)
        if record["split"] != name:
            _fail(record, "split", f"is {record['split']!r} inside {path.name}")
        if record["id"] in seen:
            _fail(record, "id", f"duplicate id in {path.name}")
        seen.add(record["id"])
    return records


def load_audio(record: dict) -> np.ndarray:
    """Decode the record's clip into a float32 mono 16 kHz waveform in [-1, 1]."""
    _check_audio_path(record)
    path = ROOT / record["audio"]
    if not path.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    if wave.shape[1] != 1:
        raise ValueError(f"{path}: expected mono audio, got {wave.shape[1]} channels")
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz, got {rate} Hz")
    return np.clip(wave[:, 0], -1.0, 1.0)


def normalize_hyp(text: str) -> str:
    """Replace non-CHARS characters with spaces, strip "'" at token edges and "." at token ends, collapse spaces, keep case."""
    tokens = (t.lstrip("'").rstrip(".'") for t in NON_CHARS.sub(" ", text).split())
    return " ".join(t for t in tokens if t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed split and print its size.")
    parser.add_argument("split", choices=SPLITS, help="split to validate")
    parser.add_argument("--final", action="store_true", help="allow reading a final evaluation split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_split(args.split, final=args.final)
    hours = sum(r["duration"] for r in records) / 3600
    print(f"{args.split}: {len(records)} valid records, {hours:.2f} h")


if __name__ == "__main__":
    main()
