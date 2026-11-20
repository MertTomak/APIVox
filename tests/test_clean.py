"""Tests for scripts/clean_dataset.py."""
import csv
import filecmp
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import clean_dataset as cd  # noqa: E402

from src import data_io  # noqa: E402

OK = {"duration": 3.0, "clipped_share": 0.0, "speech_ratio": 0.9}


def row(i: int, split: str = "train", text: str = "мен бүгін үйге келдім", duration: float = 3.0, **kw) -> dict:
    program = kw.pop("program", "radio_01")
    return {"id": f"radio_01_00_{i:03d}", "split": split, "source": "radio", "program": program,
            "episode": f"{program}_00", "audio": f"data/raw/audio/radio/radio_01_00_{i:03d}.opus",
            "duration": duration, "text_orig": text, "text": data_io.normalize_text(text),
            "lang_tags": ["kz"] * len(data_io.normalize_text(text).split()), "cs": False, **kw}


def measures_for(rows, **over) -> dict:
    out = {}
    for r in rows:
        m = {**OK, "duration": r["duration"], "pcm_sha256": f"h{r['id']}"}
        m.update(over.get(r["id"], {}))
        out[r["id"]] = m
    return out


def rules_hit(cleaner) -> dict[str, str]:
    return {e["id"]: e["rule"] for e in cleaner.log}


def base_rows(n: int = 40) -> list[dict]:
    return [row(i, text="мен бүгін үйге келдім " + "сөз " * (i % 3)) for i in range(n)]


def test_clean_rows_are_kept():
    rows = base_rows()
    c, _ = cd.apply_rules(rows, measures_for(rows), {})
    assert c.log == []


def test_c1_unreadable_and_empty_transcript():
    rows = base_rows() + [row(90), row(91, text="")]
    m = measures_for(rows, **{rows[-2]["id"]: {"error": "unreadable"}})
    c, _ = cd.apply_rules(rows, m, {})
    assert rules_hit(c) == {rows[-2]["id"]: "C1", rows[-1]["id"]: "C1"}


@pytest.mark.parametrize("duration", [0.3, 31.0])
def test_c2_duration_bounds(duration):
    rows = base_rows() + [row(90, split="dev", duration=duration)]
    c, _ = cd.apply_rules(rows, measures_for(rows), {})
    assert rules_hit(c) == {rows[-1]["id"]: "C2"}


def test_c3_only_hesitations():
    rows = base_rows() + [row(90, split="dev", text="ыыы мхм ммм")]
    c, _ = cd.apply_rules(rows, measures_for(rows), {})
    assert rules_hit(c) == {rows[-1]["id"]: "C3"}


def test_c4_thresholds_come_from_train_and_apply_everywhere():
    rows = base_rows(200)
    fast = row(300, split="test", text="мен бүгін үйге келдім " * 6, duration=1.0)
    slow = row(301, split="dev", text="иә", duration=20.0)
    rows += [fast, slow]
    c, computed = cd.apply_rules(rows, measures_for(rows), {})
    assert rules_hit(c) == {fast["id"]: "C4", slow["id"]: "C4"}
    train_rates = [cd.letters_per_second(r) for r in rows if r["split"] == "train"]
    assert computed["rate_low"] < min(train_rates) and computed["rate_high"] > max(train_rates)


def test_rate_bounds_are_robust_to_outliers():
    rates = [10.0] * 50 + [11.0] * 50 + [12.0] * 50 + [500.0]
    lo, hi = cd.rate_bounds(rates, 4.0)
    assert lo < 10.0 and 12.0 < hi < 500.0
    with pytest.raises(ValueError, match="zero spread"):
        cd.rate_bounds([5.0] * 10, 4.0)


def test_c5_clipping_and_c6_silence():
    rows = base_rows() + [row(90, split="dev"), row(91, split="dev")]
    m = measures_for(rows, **{rows[-2]["id"]: {"clipped_share": 0.01}, rows[-1]["id"]: {"speech_ratio": 0.1}})
    c, _ = cd.apply_rules(rows, m, {})
    assert rules_hit(c) == {rows[-2]["id"]: "C5", rows[-1]["id"]: "C6"}


def test_c7_keeps_first_duplicate_in_id_order():
    rows = base_rows() + [row(90, split="dev", text="сен қашан келесің"), row(91, split="dev", text="сен қашан келесің")]
    m = measures_for(rows, **{rows[-1]["id"]: {"pcm_sha256": f"h{rows[-2]['id']}"}})
    c, _ = cd.apply_rules(rows, m, {})
    assert rules_hit(c) == {rows[-1]["id"]: "C7"}


def test_c8_listening_rejects():
    rows = base_rows()
    c, _ = cd.apply_rules(rows, measures_for(rows), {rows[3]["id"]: "music"})
    assert c.log[0]["rule"] == "C8" and c.log[0]["measured_value"] == "music"


def test_c9_removes_long_train_copies_of_eval_texts_only():
    shared_long = "біз бүгін үйге ерте келдік"
    shared_short = "иә рақмет"
    rows = base_rows() + [
        row(90, text=shared_long), row(91, split="test", text=shared_long),
        row(92, text=shared_short, duration=1.2), row(93, split="dev", text=shared_short, duration=1.2),
    ]
    c, _ = cd.apply_rules(rows, measures_for(rows), {})
    assert rules_hit(c) == {rows[-4]["id"]: "C9"}


def test_verbatim_overlap_counts():
    rows = [row(1, text="бір екі үш төрт"), row(2, split="test", text="бір екі үш төрт"),
            row(3, text="иә"), row(4, split="dev", text="иә"), row(5, split="dev", text="жоқ")]
    assert cd.verbatim_overlap(rows) == {"all": 2, "min_4_words": 1}


def test_leakage_detects_a_program_in_two_splits():
    rows = [row(1), row(2, split="dev"), row(3, split="dev", program="podcasts_02")]
    out = cd.leakage(rows)
    assert out["programs_in_more_than_one_split"] == 1
    assert out["episodes_in_more_than_one_split"] == 1
    assert cd.leakage([row(1), row(3, split="dev", program="podcasts_02")])["programs_in_more_than_one_split"] == 0


# End to end on a tiny raw folder.


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return (amp * np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


@pytest.fixture
def raw_root(tmp_path) -> Path:
    raw = tmp_path / "data" / "raw"
    texts = {"train": ["мен уже ойлана бастадым", "біз бүгін үйге келдік", "ол кітап оқыды", "балалар ойнап жүр"],
             "dev": ["сен қашан келесің", "ыыы мхм"], "test": ["бұл жақсы идея екен"]}
    programs = {"train": "radio_01", "dev": "podcasts_02", "test": "podcasts_09"}
    rows = []
    for split, items in texts.items():
        for i, text in enumerate(items):
            prog = programs[split]
            rid = f"{prog}_00_{i:03d}"
            audio = raw / "audio" / prog.split("_")[0] / f"{rid}.opus"
            audio.parent.mkdir(parents=True, exist_ok=True)
            freq = 200 + 40 * i + 400 * list(texts).index(split)
            sf.write(audio, tone(2.0 + i * 0.3, freq=freq), 16000, format="OGG", subtype="OPUS")
            rows.append({"id": rid, "source": prog.split("_")[0], "program": prog, "episode": f"{prog}_00",
                         "orig_split": split, "audio": f"data/raw/audio/{prog.split('_')[0]}/{rid}.opus",
                         "duration": 2.0, "sample_rate": 16000, "channels": 1, "text_orig": text,
                         "flac_sha256": "x", "decode_error": None})
    rows.append({"id": "podcasts_09_00_009", "source": "podcasts", "program": "podcasts_09",
                 "episode": "podcasts_09_00", "orig_split": "test", "audio": None, "duration": 2.0,
                 "sample_rate": None, "channels": None, "text_orig": "бұзылған файл", "flac_sha256": "y",
                 "decode_error": "Error : flac decoder lost sync."})
    (raw / "transcripts").mkdir(parents=True)
    for source in ("radio", "podcasts"):
        lines = [json.dumps(r, ensure_ascii=False) for r in rows if r["source"] == source]
        (raw / "transcripts" / f"{source}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plan = {"seed": 2026, "programs": {"radio_01": "train", "podcasts_02": "dev", "podcasts_09": "test"},
            "train_episodes": ["radio_01_00"], "test_talkshow_episodes": []}
    (raw / "split_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    all_text = [{"id": "k1", "source": "crowdsourced", "orig_split": "train", "text": t}
                for t in ["мен бастадым ойлана", "мен бастадым ойлана", "біз келдік үйге", "біз келдік үйге"]]
    all_text += [{"id": "p1", "source": "parliament", "orig_split": "train", "text": "это уже очень важно для нас"}] * 2
    (raw / "text").mkdir()
    with gzip.open(raw / "text" / "ksc2_all_text.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in all_text) + "\n")
    return raw


def test_build_end_to_end_and_is_reproducible(raw_root, tmp_path):
    out_a, out_b = tmp_path / "pa", tmp_path / "pb"
    kept, _ = cd.build(raw_root, out_a, jobs=1, qc_path=tmp_path / "none.csv")
    cd.build(raw_root, out_b, jobs=2, qc_path=tmp_path / "none.csv")
    files = sorted(p.relative_to(out_a).as_posix() for p in out_a.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(out_b).as_posix() for p in out_b.rglob("*") if p.is_file())
    assert all(filecmp.cmp(out_a / f, out_b / f, shallow=False) for f in files)
    assert {"train.jsonl", "dev.jsonl", "test.jsonl", "stats.json", "cleaning_log.csv",
            "lexicon/kz_words.txt", "lexicon/ru_words.txt"} <= set(files)
    for line in (out_a / "train.jsonl").read_text(encoding="utf-8").splitlines():
        data_io.validate(json.loads(line))
    first = json.loads((out_a / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["text"] == "мен уже ойлана бастадым"
    assert first["lang_tags"] == ["kz", "ru", "kz", "kz"] and first["cs"] is True
    with open(out_a / "cleaning_log.csv", encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    assert [(e["split"], e["rule"]) for e in log] == [("test", "C1"), ("dev", "C3")]
    assert "lost sync" in log[0]["measured_value"]
    stats = json.loads((out_a / "stats.json").read_text(encoding="utf-8"))
    assert stats["leakage"]["programs_in_more_than_one_split"] == 0
    assert stats["splits"]["train"]["cs_utterances"] == 1
    assert stats["hesitation_tokens_removed_by_normalization"] == 2


def test_load_raw_rejects_unknown_program(raw_root):
    plan = json.loads((raw_root / "split_plan.json").read_text(encoding="utf-8"))
    del plan["programs"]["podcasts_09"]
    (raw_root / "split_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="not in split_plan"):
        cd.load_raw(raw_root)


def test_human_qc_reader(tmp_path):
    path = tmp_path / "qc.csv"
    path.write_text("id,split,verdict,reason,checker,date\na,train,y,,yagiz,1\nb,dev,n,music,yagiz,1\n"
                    "c,dev,N,cut,gorkem,1\n", encoding="utf-8")
    assert cd.read_human_qc(path) == {"b": "music", "c": "cut"}
    assert cd.read_human_qc(tmp_path / "missing.csv") == {}
