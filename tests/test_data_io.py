"""Tests for src/data_io.py."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import data_io


def good(**changes) -> dict:
    """A valid code-switched radio record with the given fields replaced."""
    record = {
        "id": "radio_01_04_359",
        "split": "train",
        "audio": "data/raw/audio/radio/radio_01_04_359.opus",
        "duration": 4.2,
        "source": "radio",
        "program": "radio_01",
        "episode": "radio_01_04",
        "text": "мен уже ойлана бастадым",
        "text_orig": "ыыы мен уже ойлана бастадым",
        "lang_tags": ["kz", "ru", "kz", "kz"],
        "cs": True,
    }
    return {**record, **changes}


def write_split(directory: Path, name: str, records: list[dict]) -> None:
    lines = [json.dumps({**r, "split": name}, ensure_ascii=False) for r in records]
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def processed(monkeypatch, tmp_path):
    monkeypatch.setattr(data_io, "PROCESSED_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def root(monkeypatch, tmp_path):
    monkeypatch.setattr(data_io, "ROOT", tmp_path)
    return tmp_path


# CHARS


def test_chars_are_space_russian_without_yo_then_kazakh():
    assert data_io.CHARS == " абвгдежзийклмнопрстуфхцчшщъыьэюя" + "әғқңөұүһі"
    assert len(data_io.CHARS) == 42
    assert len(set(data_io.CHARS)) == 42
    assert "ё" not in data_io.CHARS


def test_char_ids_round_trip():
    assert data_io.CHAR2ID[" "] == 0
    assert all(data_io.ID2CHAR[data_io.CHAR2ID[c]] == c for c in data_io.CHARS)


# normalize_text


@pytest.mark.parametrize("text, expected", [
    ("Ыыы мен УЖЕ ойлана бастадым", "мен уже ойлана бастадым"),
    ("ёлка", "елка"),
    ("2020 жылы", "екі мың жиырма жылы"),
    ("15-ші сынып", "он бес ші сынып"),
    ("бір-бірімен", "бір бірімен"),
    ("сол — дұрыс", "сол дұрыс"),
    ("iPhone алдым", "алдым"),
    ("иә, болды!", "иә болды"),
    ("мхм ммм ііі э хм жоқ", "жоқ"),
    ("а сен келдің бе", "а сен келдің бе"),
    ("ол (күледі) келді [noise]", "ол келді"),
    ("  көп    бос   орын  ", "көп бос орын"),
    ("ыыы", ""),
])
def test_normalize_text(text, expected):
    assert data_io.normalize_text(text) == expected


@pytest.mark.parametrize("text", [
    "Ыыы мен УЖЕ ойлана бастадым", "2020 жылы", "iPhone - ёж (х)", "", "ы м і",
])
def test_normalize_text_is_idempotent(text):
    once = data_io.normalize_text(text)
    assert data_io.normalize_text(once) == once


def test_normalized_text_uses_only_chars():
    out = data_io.normalize_text("Привет, ДОС! 3 kg — ok? Қалайсың…")
    assert set(out) <= set(data_io.CHARS)


# is_cs


@pytest.mark.parametrize("tags, expected", [
    (["kz", "ru"], True),
    (["kz", "mix"], True),
    (["kz", "kz", "amb"], False),
    (["ru", "ru"], False),
    ([], False),
])
def test_is_cs(tags, expected):
    assert data_io.is_cs(tags) is expected


# validate


def test_valid_record_passes():
    data_io.validate(good())
    data_io.validate(good(text="мен келдім", lang_tags=["kz", "kz"], cs=False, split="dev"))
    data_io.validate(good(source="read", program="read", episode="read", split="test_read"))


@pytest.mark.parametrize("field", list(data_io.FIELDS))
def test_missing_field_fails(field):
    record = good()
    del record[field]
    with pytest.raises(ValueError, match=field):
        data_io.validate(record)


def test_unexpected_field_fails():
    with pytest.raises(ValueError, match="unexpected key"):
        data_io.validate(good(speaker="h1"))


@pytest.mark.parametrize("field, value", [
    ("duration", "4.2"), ("cs", "true"), ("lang_tags", "kz ru"), ("duration", True), ("text", None),
])
def test_wrong_type_fails(field, value):
    with pytest.raises(ValueError, match=field):
        data_io.validate(good(**{field: value}))


@pytest.mark.parametrize("field, value", [
    ("split", "validation"), ("source", "parliament"), ("duration", 0), ("duration", -1.0),
    ("program", ""), ("episode", ""),
])
def test_bad_value_fails(field, value):
    with pytest.raises(ValueError, match=field):
        data_io.validate(good(**{field: value}))


@pytest.mark.parametrize("audio", ["/abs/a.opus", "../a.opus", "data/../../a.opus"])
def test_bad_audio_path_fails(audio):
    with pytest.raises(ValueError, match="audio"):
        data_io.validate(good(audio=audio))


def test_empty_text_fails():
    with pytest.raises(ValueError, match="empty"):
        data_io.validate(good(text="", lang_tags=[], cs=False))


@pytest.mark.parametrize("text", ["Мен уже ойлана бастадым", "мен ёж ойлана бастадым", "мен уже, ойлана бастадым"])
def test_text_outside_chars_fails(text):
    with pytest.raises(ValueError, match="text"):
        data_io.validate(good(text=text))


def test_unnormalized_text_fails():
    with pytest.raises(ValueError, match="not normalized"):
        data_io.validate(good(text="ыыы мен уже ойлана", lang_tags=["kz", "kz", "ru", "kz"]))


def test_tag_count_mismatch_fails():
    with pytest.raises(ValueError, match="lang_tags"):
        data_io.validate(good(lang_tags=["kz", "ru", "kz"]))


def test_unknown_tag_fails():
    with pytest.raises(ValueError, match="unknown tags"):
        data_io.validate(good(lang_tags=["kz", "en", "kz", "kz"]))


def test_cs_inconsistent_with_tags_fails():
    with pytest.raises(ValueError, match="cs"):
        data_io.validate(good(cs=False))
    with pytest.raises(ValueError, match="cs"):
        data_io.validate(good(lang_tags=["kz"] * 4, cs=True))


# load_split


@pytest.mark.parametrize("name", data_io.FINAL_SPLITS)
def test_final_split_needs_final(processed, name):
    write_split(processed, name, [good()])
    with pytest.raises(PermissionError):
        data_io.load_split(name)
    assert [r["id"] for r in data_io.load_split(name, final=True)] == ["radio_01_04_359"]


def test_unknown_split_fails(processed):
    with pytest.raises(ValueError, match="unknown split"):
        data_io.load_split("heldout")


def test_missing_manifest_fails(processed):
    with pytest.raises(FileNotFoundError):
        data_io.load_split("dev")


def test_split_field_must_match_file(processed):
    (processed / "dev.jsonl").write_text(json.dumps(good(), ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside dev.jsonl"):
        data_io.load_split("dev")


def test_duplicate_id_fails(processed):
    write_split(processed, "train", [good(), good()])
    with pytest.raises(ValueError, match="duplicate id"):
        data_io.load_split("train")


def test_invalid_json_fails(processed):
    (processed / "train.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        data_io.load_split("train")


def test_load_split_keeps_file_order(processed):
    rows = [good(id=f"radio_01_04_{i:03d}") for i in (5, 1, 3)]
    write_split(processed, "train", rows)
    assert [r["id"] for r in data_io.load_split("train")] == [r["id"] for r in rows]


# load_audio


def write_tone(path: Path, rate: int = 16000, channels: int = 1, seconds: float = 1.0, amp: float = 0.5,
               fmt: str = "WAV", subtype: str = "PCM_16") -> None:
    t = np.arange(int(rate * seconds)) / rate
    wave = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, rate, format=fmt, subtype=subtype)


def test_load_audio_reads_mono_16k(root):
    write_tone(root / "a.wav")
    wave = data_io.load_audio(good(audio="a.wav"))
    assert wave.dtype == np.float32
    assert wave.shape == (16000,)
    assert np.abs(wave).max() <= 1.0


def test_load_audio_reads_opus(root):
    write_tone(root / "a.opus", seconds=2.0, fmt="OGG", subtype="OPUS")
    wave = data_io.load_audio(good(audio="a.opus"))
    assert wave.dtype == np.float32
    assert abs(len(wave) - 32000) < 400


def test_load_audio_rejects_stereo(root):
    write_tone(root / "s.wav", channels=2)
    with pytest.raises(ValueError, match="mono"):
        data_io.load_audio(good(audio="s.wav"))


def test_load_audio_rejects_other_rate(root):
    write_tone(root / "r.wav", rate=22050)
    with pytest.raises(ValueError, match="16000"):
        data_io.load_audio(good(audio="r.wav"))


def test_load_audio_missing_file(root):
    with pytest.raises(FileNotFoundError):
        data_io.load_audio(good(audio="nope.opus"))


def test_load_audio_rejects_unsafe_path(root):
    with pytest.raises(ValueError, match="audio"):
        data_io.load_audio(good(audio="../escape.wav"))
