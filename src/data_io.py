"""Load and validate manifests, decode audio and normalize text with the single Aralas text policy.

CHARS is the output alphabet; CHAR2ID gives 0-based positions in CHARS. CTC models add 1 to
every id so that index 0 is the blank.
"""
import argparse
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import soundfile as sf
from num2words import num2words

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLE_RATE = 16000

# Russian letters without "ё" (mapped to "е" by normalize_text), then the nine Kazakh-specific letters.
RUSSIAN_LETTERS = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
KAZAKH_LETTERS = "әғқңөұүһі"
CHARS = " " + RUSSIAN_LETTERS + KAZAKH_LETTERS
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
ID2CHAR = dict(enumerate(CHARS))

OPEN_SPLITS = ("train", "dev")
FINAL_SPLITS = ("test", "test_talkshow", "test_read")
SPLITS = OPEN_SPLITS + FINAL_SPLITS
SOURCES = ("podcasts", "radio", "talkshow", "read")
LANG_TAGS = ("kz", "ru", "mix", "amb")
CS_TAGS = ("ru", "mix")

FIELDS = {
    "id": (str,),
    "split": (str,),
    "audio": (str,),
    "duration": (int, float),
    "source": (str,),
    "program": (str,),
    "episode": (str,),
    "text": (str,),
    "text_orig": (str,),
    "lang_tags": (list,),
    "cs": (bool,),
}

# Text policy steps, applied in this order by normalize_text.
DIGITS = re.compile(r"\d+")
BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}")
DASHES = re.compile(r"[-‐-―−]")
NON_CHARS = re.compile(f"[^{re.escape(CHARS)}]")
# Hesitation tokens of the KSC2 transcripts; "а" is a real Kazakh and Russian word and is kept.
HESITATION = re.compile(r"ы+|і+|м+|мхм|хм|э+")


def _number_words(match: re.Match) -> str:
    """Spell a digit group as Kazakh number words."""
    return f" {num2words(int(match.group()), lang='kz')} "


def normalize_text(text: str) -> str:
    """The one text policy for references and hypotheses: lowercase, ё->е, digits->words, drop brackets, dashes and non-CHARS, drop hesitations."""
    text = text.lower().replace("ё", "е")
    text = DIGITS.sub(_number_words, text)
    text = BRACKETED.sub(" ", text)
    text = DASHES.sub(" ", text)
    text = NON_CHARS.sub(" ", text)
    return " ".join(w for w in text.split() if not HESITATION.fullmatch(w))


def is_cs(lang_tags: list[str]) -> bool:
    """True if the tags contain at least one Kazakh word and at least one Russian or mixed word."""
    return "kz" in lang_tags and any(t in CS_TAGS for t in lang_tags)


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


def _check_audio_path(record: dict) -> None:
    posix, windows = PurePosixPath(record["audio"]), PureWindowsPath(record["audio"])
    if posix.is_absolute() or windows.anchor or ".." in posix.parts or ".." in windows.parts:
        _fail(record, "audio", f"{record['audio']!r} must be a repo-relative path without '..'")


def _check_text(record: dict) -> None:
    text = record["text"]
    if not text:
        _fail(record, "text", "must not be empty")
    bad = sorted({c for c in text if c not in CHAR2ID})
    if bad:
        _fail(record, "text", f"characters outside CHARS: {bad}")
    if normalize_text(text) != text:
        _fail(record, "text", f"is not normalized: {text!r}")


def _check_tags(record: dict) -> None:
    tags, n_words = record["lang_tags"], len(record["text"].split())
    if len(tags) != n_words:
        _fail(record, "lang_tags", f"{len(tags)} tags for {n_words} words")
    unknown = sorted({t for t in tags if t not in LANG_TAGS})
    if unknown:
        _fail(record, "lang_tags", f"unknown tags {unknown}, expected {LANG_TAGS}")
    if record["cs"] != is_cs(tags):
        _fail(record, "cs", f"is {record['cs']} but lang_tags say {is_cs(tags)}")


def validate(record: dict) -> None:
    """Raise ValueError naming the id and the field if a manifest record breaks the contract."""
    _check_keys_and_types(record)
    if record["split"] not in SPLITS:
        _fail(record, "split", f"unknown split {record['split']!r}")
    if record["source"] not in SOURCES:
        _fail(record, "source", f"unknown source {record['source']!r}")
    if record["duration"] <= 0:
        _fail(record, "duration", f"must be positive, got {record['duration']}")
    for key in ("program", "episode"):
        if not record[key]:
            _fail(record, key, "must not be empty")
    _check_audio_path(record)
    _check_text(record)
    _check_tags(record)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed split and print its size.")
    parser.add_argument("split", choices=SPLITS, help="split to validate")
    parser.add_argument("--final", action="store_true", help="allow reading a final evaluation split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_split(args.split, final=args.final)
    hours = sum(r["duration"] for r in records) / 3600
    n_cs = sum(r["cs"] for r in records)
    print(f"{args.split}: {len(records)} valid records, {hours:.2f} h, {n_cs} code-switched")


if __name__ == "__main__":
    main()
