"""Tests for src/eval_zero_shot.py with stand-in models, so no checkpoint is downloaded."""
import argparse
import json
import re
from types import SimpleNamespace

import pytest
import torch

from src import data_io, eval_zero_shot, metrics

FIXTURE = data_io.ROOT / "tests" / "fixtures" / "mini.jsonl"


def fixture_records():
    return [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]


class TinyModel(torch.nn.Module):
    """Stand-in CTC model: returns all-zero logits over a 3-symbol vocabulary."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def forward(self, input_values, **_):
        return SimpleNamespace(logits=torch.zeros(input_values.shape[0], input_values.shape[1] // 320, 3))


class TinyProcessor:
    """Stand-in CTC processor: pads waves and decodes every row to the same word."""

    def __call__(self, waves, sampling_rate, return_tensors, padding):
        width = max(len(w) for w in waves)
        return {"input_values": torch.stack([torch.nn.functional.pad(torch.from_numpy(w), (0, width - len(w)))
                                             for w in waves])}

    def batch_decode(self, ids):
        return ["Сәлем!"] * ids.shape[0]


def test_every_model_has_a_pinned_revision_and_a_ksc_note():
    for name, spec in eval_zero_shot.MODELS.items():
        assert re.fullmatch(r"[0-9a-f]{40}", spec["revision"]), name
        assert spec["family"] in ("whisper", "ctc")
        assert spec["ksc_training"].split(":")[0] in ("yes", "not stated")
    assert set(eval_zero_shot.MODELS) == {"whisper_small_zs", "whisper_large_zs", "xlsr_kk_zs", "mms_kk_zs"}


@pytest.mark.parametrize("fixed", [None, 30.0])
def test_batches_cover_every_clip_once_within_budget(fixed):
    durations = [5.0, 0.6, 29.0, 3.0, 3.0, 12.0, 1.0]
    batches = eval_zero_shot.make_batches(durations, 40.0, fixed)
    assert sorted(i for b in batches for i in b) == list(range(len(durations)))
    for b in batches:
        assert len(b) == 1 or len(b) * (fixed or max(durations[i] for i in b)) <= 40.0


def test_fixture_clips_run_through_a_stub_ctc_model():
    records = fixture_records()[:3]
    hyps = eval_zero_shot.transcribe(records, TinyProcessor(), TinyModel(), "ctc", 12.0)
    assert hyps == ["Сәлем!"] * 3


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Route loading, decoding and results to stand-ins; hypotheses keep case and punctuation."""
    records = fixture_records()
    raw = [r["text_orig"].upper() + "." for r in records]
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(data_io, "load_split", lambda name, final=False: records)
    monkeypatch.setattr(eval_zero_shot, "load_model", lambda name, device: (None, TinyModel()))
    monkeypatch.setattr(eval_zero_shot, "transcribe", lambda recs, processor, model, family, budget: raw)
    return records, raw, tmp_path


def args(**kw):
    base = {"models": ["xlsr_kk_zs"], "splits": list(eval_zero_shot.EVAL_SPLITS), "batch_seconds": 60.0,
            "limit": None}
    return argparse.Namespace(**{**base, **kw})


def test_final_writes_raw_predictions_and_a_complete_run_file(stubbed):
    records, raw, out = stubbed
    eval_zero_shot.run_final(args(), torch.device("cpu"))
    run = json.loads((out / "xlsr_kk_zs_seed0.json").read_text(encoding="utf-8"))
    spec = eval_zero_shot.MODELS["xlsr_kk_zs"]
    assert (run["checkpoint"], run["revision"], run["seed"]) == (spec["checkpoint"], spec["revision"], 0)
    assert run["config"]["ksc_training"].startswith("yes")
    for split in eval_zero_shot.EVAL_SPLITS:
        path = out / "preds" / f"xlsr_kk_zs_seed0_{split}.jsonl"
        hyps = [json.loads(line)["hyp"] for line in path.read_text(encoding="utf-8").splitlines()]
        assert hyps == raw
        assert run["splits"][split] == metrics.evaluate(records, hyps)
        assert run["splits"][split]["wer"] == 0.0


def test_final_can_be_limited_to_some_splits(stubbed):
    _, _, out = stubbed
    eval_zero_shot.run_final(args(splits=["test", "test_read"]), torch.device("cpu"))
    run = json.loads((out / "xlsr_kk_zs_seed0.json").read_text(encoding="utf-8"))
    assert set(run["splits"]) == {"test", "test_read"}


def test_dev_mode_writes_nothing(stubbed, capsys):
    _, _, out = stubbed
    eval_zero_shot.run_dev(args(), torch.device("cpu"))
    assert list(out.iterdir()) == []
    assert "wer 0.0000" in capsys.readouterr().out
