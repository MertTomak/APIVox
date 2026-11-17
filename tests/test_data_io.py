"""Tests for src/data_io.py."""
import json
import string
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src import data_io

FIXTURES = Path(__file__).resolve().parent / "fixtures"
MINI = [json.loads(line) for line in (FIXTURES / "mini.jsonl").read_text(encoding="utf-8").splitlines()]


def good(**changes) -> dict:
    """A valid synthesized record with the given fields replaced."""
    return {**MINI[0], **changes}


def human(**changes) -> dict:
    """A valid human record with the given fields replaced."""
    return good(**{"engine": "human", "speaker": "m1", "spoken": None, "style": None, **changes})


def write_split(directory: Path, name: str, records: list[dict]) -> None:
    lines = [json.dumps({**r, "split": name}) for r in records]
    (directory / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def processed(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(data_io, "PROCESSED_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def repo_root(monkeypatch, tmp_path) -> Path:
    monkeypatch.setattr(data_io, "ROOT", tmp_path)
    return tmp_path


# CHARS

def test_chars_exact_order_and_length():
    expected = " '._" + string.ascii_lowercase + string.ascii_uppercase + string.digits
    assert data_io.CHARS == expected
    assert len(data_io.CHARS) == 66
    assert len(set(data_io.CHARS)) == 66


def test_char_ids_round_trip():
    assert data_io.CHAR2ID[" "] == 0 and data_io.CHAR2ID["9"] == 65
    assert all(data_io.ID2CHAR[data_io.CHAR2ID[c]] == c for c in data_io.CHARS)
    assert sorted(data_io.ID2CHAR) == list(range(66))


# load_split

@pytest.mark.parametrize("name", data_io.FINAL_SPLITS)
def test_final_split_refused_without_final(processed, name):
    write_split(processed, name, MINI)
    with pytest.raises(PermissionError, match=name):
        data_io.load_split(name)


@pytest.mark.parametrize("name", data_io.FINAL_SPLITS)
def test_final_split_allowed_with_final(processed, name):
    write_split(processed, name, MINI)
    assert [r["id"] for r in data_io.load_split(name, final=True)] == [r["id"] for r in MINI]


def test_unknown_split(processed):
    with pytest.raises(ValueError, match="unknown split"):
        data_io.load_split("validation")


def test_duplicate_id(processed):
    write_split(processed, "dev", MINI + [MINI[3]])
    with pytest.raises(ValueError, match="fixture_04.*'id'.*duplicate"):
        data_io.load_split("dev")


def test_record_split_must_match_file(processed):
    (processed / "train.jsonl").write_text(json.dumps(MINI[0]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'split'"):
        data_io.load_split("train")


def test_invalid_record_rejected(processed):
    write_split(processed, "dev", MINI[:2] + [good(id="bad", duration=0.0)])
    with pytest.raises(ValueError, match="'bad'.*'duration'"):
        data_io.load_split("dev")


def test_missing_split_file(processed):
    with pytest.raises(FileNotFoundError, match="dev.jsonl"):
        data_io.load_split("dev")


def test_invalid_json_line(processed):
    lines = [json.dumps({**MINI[0], "split": "dev"}), "{not json"]
    (processed / "dev.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev.jsonl:2: invalid JSON"):
        data_io.load_split("dev")


def test_line_not_an_object(processed):
    lines = [json.dumps({**MINI[0], "split": "dev"}), json.dumps(["a", "list"])]
    (processed / "dev.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dev.jsonl:2: expected a JSON object"):
        data_io.load_split("dev")


def test_blank_lines_skipped(processed):
    lines = [json.dumps({**r, "split": "dev"}) for r in MINI[:2]]
    (processed / "dev.jsonl").write_text("\n" + lines[0] + "\n  \n\n" + lines[1] + "\n\n", encoding="utf-8")
    assert [r["id"] for r in data_io.load_split("dev")] == [r["id"] for r in MINI[:2]]


def test_order_kept(processed):
    shuffled = MINI[::-1]
    write_split(processed, "dev", shuffled)
    assert [r["id"] for r in data_io.load_split("dev")] == [r["id"] for r in shuffled]


# validate

def test_fixture_records_pass():
    assert len(MINI) == 8
    for record in MINI:
        data_io.validate(record)


def test_human_record_passes():
    data_io.validate(human())
    data_io.validate(human(style="terse"))


def test_missing_key():
    record = good()
    del record["doc_path"]
    with pytest.raises(ValueError, match="fixture_01.*'doc_path'.*missing"):
        data_io.validate(record)


def test_extra_key():
    with pytest.raises(ValueError, match="fixture_01.*'notes'.*unexpected"):
        data_io.validate(good(notes="x"))


@pytest.mark.parametrize("field, value", [
    ("id", 7),
    ("duration", "2.0"),
    ("duration", True),
    ("written", None),
    ("spoken", 3),
    ("style", 1),
    ("api_seen_in_train", 1),
])
def test_wrong_type(field, value):
    with pytest.raises(ValueError, match=f"'{field}'.*expected"):
        data_io.validate(good(**{field: value}))


@pytest.mark.parametrize("field, value", [
    ("split", "valid"),
    ("engine", "espeak"),
    ("style", "casual"),
])
def test_unknown_split_engine_or_style(field, value):
    with pytest.raises(ValueError, match=f"fixture_01.*'{field}'.*unknown"):
        data_io.validate(good(**{field: value}))


@pytest.mark.parametrize("duration", [0, 0.0, -1.5])
def test_non_positive_duration(duration):
    with pytest.raises(ValueError, match="fixture_01.*'duration'"):
        data_io.validate(good(duration=duration))


@pytest.mark.parametrize("api, written", [
    ("DataFrame.groupby", "use groupby to split the data"),
    ("DataFrame.groupby", "use DataFrame.groupby and DataFrame.groupby"),
    ("groupby", "use DataFrame.groupby to split"),
])
def test_api_not_exactly_once_as_token(api, written):
    with pytest.raises(ValueError, match="fixture_01.*'api'.*exactly once"):
        data_io.validate(good(api=api, written=written))


def test_api_with_call_parentheses():
    with pytest.raises(ValueError, match="fixture_01.*'api'.*\\(\\)"):
        data_io.validate(good(api="read_csv()", written="call read_csv() now"))


@pytest.mark.parametrize("written", ["use DataFrame.groupby, then", "use DataFrame.groupby\tnow", "naïve DataFrame.groupby"])
def test_character_outside_chars(written):
    with pytest.raises(ValueError, match="fixture_01.*'written'.*outside CHARS"):
        data_io.validate(good(written=written))


@pytest.mark.parametrize("written", ["Use DataFrame.groupby", "use DataFrame.groupby 2", "use DataFrame.groupby x.y", "use  DataFrame.groupby"])
def test_context_word_not_lowercase(written):
    with pytest.raises(ValueError, match="fixture_01.*'written'.*context word"):
        data_io.validate(good(written=written))


@pytest.mark.parametrize("api, written", [("", "use  data"), (" ", "use   data")])
def test_api_empty_or_whitespace(api, written):
    with pytest.raises(ValueError, match="fixture_01.*'api'.*empty"):
        data_io.validate(good(api=api, written=written))


@pytest.mark.parametrize("api", [".groupby", "groupby.", "'norm", "norm'"])
def test_api_with_dot_or_apostrophe_at_edge(api):
    with pytest.raises(ValueError, match="fixture_01.*'api'.*start or end"):
        data_io.validate(good(api=api, written=f"use {api} now"))


@pytest.mark.parametrize("audio", ["/data/processed/audio/dev/x.opus", "../x.opus", "data/../../x.opus"])
def test_audio_path_not_repo_relative(audio):
    with pytest.raises(ValueError, match="fixture_01.*'audio'.*repo-relative"):
        data_io.validate(good(audio=audio))


def test_spoken_null_for_synthesized_clip():
    with pytest.raises(ValueError, match="fixture_01.*'spoken'.*must not be null"):
        data_io.validate(good(spoken=None))


def test_spoken_set_for_human_clip():
    with pytest.raises(ValueError, match="fixture_01.*'spoken'.*must be null"):
        data_io.validate(human(spoken="use data frame dot group by"))


def test_style_null_for_synthesized_clip():
    with pytest.raises(ValueError, match="fixture_01.*'style'.*must not be null"):
        data_io.validate(good(style=None))


# load_audio

@pytest.mark.parametrize("record", MINI, ids=[r["id"] for r in MINI])
def test_fixture_clip_decodes(record):
    wave = data_io.load_audio(record)
    assert wave.dtype == np.float32
    assert wave.ndim == 1
    assert -1.0 <= wave.min() and wave.max() <= 1.0
    assert abs(len(wave) - record["duration"] * 16000) <= 160
    assert np.abs(wave).max() > 0.05


def test_fixture_clips_are_opus_16k_mono():
    for record in MINI:
        info = sf.info(data_io.ROOT / record["audio"])
        assert (info.format, info.subtype, info.samplerate, info.channels) == ("OGG", "OPUS", 16000, 1)
        wave = data_io.load_audio(record)
        assert wave.dtype == np.float32 and wave.ndim == 1 and len(wave) > 0


@pytest.mark.parametrize("channels, rate, message", [(2, 16000, "mono"), (1, 44100, "16000 Hz")])
def test_wrong_audio_format_raises(repo_root, channels, rate, message):
    path = repo_root / "bad.wav"
    sf.write(path, np.zeros((rate // 10, channels), dtype=np.float32), rate)
    with pytest.raises(ValueError, match=message) as err:
        data_io.load_audio(good(audio="bad.wav"))
    assert str(path) in str(err.value)


def test_load_audio_clips_to_unit_range(repo_root):
    sf.write(repo_root / "loud.wav", np.full((1600, 1), 1.5, dtype=np.float32), 16000, subtype="FLOAT")
    wave = data_io.load_audio(good(audio="loud.wav"))
    assert len(wave) == 1600
    assert wave.min() >= -1.0 and wave.max() <= 1.0


@pytest.mark.parametrize("audio", ["/etc/x.opus", "../x.opus"])
def test_load_audio_rejects_non_relative_path(audio):
    with pytest.raises(ValueError, match="'audio'.*repo-relative"):
        data_io.load_audio(good(audio=audio))


def test_missing_audio_file_raises():
    with pytest.raises(FileNotFoundError, match="nope.opus"):
        data_io.load_audio(good(audio="tests/fixtures/nope.opus"))


# normalize_hyp

@pytest.mark.parametrize("text, expected", [
    ("norm.", "norm"),
    ("use DataFrame.groupby, then", "use DataFrame.groupby then"),
    ("(read_csv)", "read_csv"),
    ('"np.linalg.norm."', "np.linalg.norm"),
    ("__init__", "__init__"),
    ("a  b\t c", "a b c"),
    ("", ""),
    ("   ", ""),
    ("Use DataFrame.GroupBy", "Use DataFrame.GroupBy"),
    ("it's [os.path.join]!", "it's os.path.join"),
    ("call it; then: stop?", "call it then stop"),
    ("ok . done", "ok done"),
    ("DataFrame.groupby—then", "DataFrame.groupby then"),
    ("DataFrame.groupby…", "DataFrame.groupby"),
    ("read-csv", "read csv"),
    ("groupby()", "groupby"),
    ("DataFrame .groupby", "DataFrame .groupby"),
    ("don't", "don't"),
    ("'norm'", "norm"),
])
def test_normalize_hyp(text, expected):
    assert data_io.normalize_hyp(text) == expected
