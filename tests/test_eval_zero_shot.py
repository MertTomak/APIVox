"""Tests for the rule converter and the final outputs of src/eval_zero_shot.py."""
import argparse
import json

import pytest
import torch

from src import data_io, eval_zero_shot, metrics
from src.eval_zero_shot import convert, join_spelled_letters, join_symbol_words

FIXTURE = data_io.ROOT / "tests" / "fixtures" / "mini.jsonl"


@pytest.mark.parametrize("raw, expected", [
    ("Use data frame dot groupby to split the data.", "Use data frame.groupby to split the data"),
    ("when using index dot sort underscore values for datetime",
     "when using index.sort_values for datetime"),
    ("use o s dot path dot join here", "use os.path.join here"),
    ("call read underscore csv on the file", "call read_csv on the file"),
    ("Bug in merge underscore ordered, couldn't handle", "Bug in merge_ordered couldn't handle"),
    ("returns a n d array", "returns a nd array"),
    ("the median absolute error", "the median absolute error"),
])
def test_convert(raw, expected):
    assert convert(raw) == expected


def test_symbol_word_at_edges_is_kept():
    assert join_symbol_words(["dot", "product"]) == ["dot", "product"]
    assert join_symbol_words(["the", "dot"]) == ["the", "dot"]


def test_symbol_word_is_case_insensitive():
    assert join_symbol_words(["np", "Dot", "reshape"]) == ["np.reshape"]


def test_single_letter_alone_is_not_joined():
    assert join_spelled_letters(["a", "frame"]) == ["a", "frame"]


def test_letters_at_end_are_joined():
    assert join_spelled_letters(["read", "c", "s", "v"]) == ["read", "csv"]


def test_existing_code_tokens_are_untouched():
    assert convert("use DataFrame.groupby here") == "use DataFrame.groupby here"


class TinyModel(torch.nn.Linear):
    """Stand-in for a Whisper model: has parameters and a dtype."""

    def __init__(self):
        super().__init__(2, 2)
        self.dtype = torch.float32


@pytest.fixture
def fake_final(monkeypatch, tmp_path):
    """Run the final mode on fixture records with a stand-in model and results in a temp dir."""
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    raw = [f"Use {r['written'].replace('.', ' dot ')}." for r in records]
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(data_io, "load_split", lambda name, final=False: records)
    monkeypatch.setattr(eval_zero_shot, "load_model", lambda checkpoint, device: (None, TinyModel()))
    monkeypatch.setattr(eval_zero_shot, "transcribe", lambda recs, processor, model, batch_size: raw)
    args = argparse.Namespace(models=["whisper_base_zs"], batch_size=8)
    eval_zero_shot.run_final(args, torch.device("cpu"))
    return records, raw, tmp_path


def read_hyps(path):
    return [json.loads(line)["hyp"] for line in path.read_text(encoding="utf-8").splitlines()]


def test_final_writes_converted_and_raw_preds_for_every_split(fake_final):
    records, raw, out = fake_final
    for split in eval_zero_shot.EVAL_SPLITS:
        assert read_hyps(out / "preds" / f"whisper_base_zs_seed0_{split}.raw.jsonl") == raw
        assert read_hyps(out / "preds" / f"whisper_base_zs_seed0_{split}.jsonl") == [convert(h) for h in raw]


def test_raw_metrics_in_run_file_are_recomputable(fake_final):
    records, _, out = fake_final
    run = json.loads((out / "whisper_base_zs_seed0.json").read_text(encoding="utf-8"))
    for split in eval_zero_shot.EVAL_SPLITS:
        stored = run["splits"][split]
        raw_hyps = read_hyps(out / "preds" / f"whisper_base_zs_seed0_{split}.raw.jsonl")
        conv_hyps = read_hyps(out / "preds" / f"whisper_base_zs_seed0_{split}.jsonl")
        assert stored["raw"] == metrics.evaluate(records, raw_hyps)
        assert {k: v for k, v in stored.items() if k != "raw"} == metrics.evaluate(records, conv_hyps)
