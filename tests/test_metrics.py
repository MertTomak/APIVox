"""Tests for src/metrics.py."""
import json

import pytest

from src import data_io, metrics

GROUPBY = "DataFrame.groupby"
WRITTEN = "use DataFrame.groupby to split the data"


def rec(api=GROUPBY, written=WRITTEN, **extra):
    base = {"id": "u", "api": api, "written": written, "library": "pandas", "engine": "piper",
            "style": "explicit", "speaker": "p1", "api_seen_in_train": True}
    return {**base, **extra}


def evaluate_one(hyp, record=None, **kwargs):
    return metrics.evaluate([record or rec()], [hyp], n_boot=50, **kwargs)


@pytest.fixture
def fake_normalize(monkeypatch):
    """Stand-in for data_io.normalize_hyp: strip trailing sentence punctuation, collapse spaces."""
    def normalize(text):
        return " ".join(t.rstrip(".,;:!?") for t in text.split())
    monkeypatch.setattr(data_io, "normalize_hyp", normalize, raising=False)


@pytest.fixture
def fake_split(monkeypatch):
    """Stand-in for data_io.load_split returning three records."""
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    monkeypatch.setattr(data_io, "load_split", lambda name, final=False: rows, raising=False)


@pytest.fixture
def results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path)
    return tmp_path


# Tests below need data_io.normalize_hyp (fake_normalize).

def test_case_error_fails_api_em_but_not_wer(fake_normalize):
    out = evaluate_one("use dataframe.groupby to split the data")
    assert out["api_em"] == 0.0
    assert out["wer"] == 0.0
    assert out["cer"] == 0.0
    assert out["api_cer"] == pytest.approx(2 / 17)


def test_exact_api_with_wrong_context_counts(fake_normalize):
    out = evaluate_one("run DataFrame.groupby to cut the data")
    assert out["api_em"] == 1.0
    assert out["wer"] == pytest.approx(2 / 6)
    assert out["api_cer"] == 0.0


def test_punctuation_after_api_still_counts(fake_normalize):
    record = rec(api="norm", written="compute the norm")
    out = evaluate_one("compute the norm.", record)
    assert out["api_em"] == 1.0
    assert out["wer"] == 0.0


def test_api_must_be_a_whole_token(fake_normalize):
    record = rec(api="sort", written="use sort to order")
    assert evaluate_one("use sorted to order", record)["api_em"] == 0.0
    assert evaluate_one("use sort to order", record)["api_em"] == 1.0


def test_api_cer_gives_partial_credit_to_near_miss(fake_normalize):
    out = evaluate_one("use data_frame.groupby to split the data")
    assert out["api_em"] == 0.0
    assert out["api_cer"] == pytest.approx(3 / 17)


def test_empty_hypothesis_is_all_errors(fake_normalize):
    out = evaluate_one("")
    assert out["api_em"] == 0.0
    assert out["wer"] == 1.0
    assert out["cer"] == 1.0
    assert out["api_cer"] == 1.0


def test_evaluate_averages_over_utterances(fake_normalize):
    records = [rec(), rec()]
    out = metrics.evaluate(records, [WRITTEN, "use dataframe.groupby to split the data"], n_boot=50)
    assert out["api_em"] == 0.5
    assert out["api_cer"] == pytest.approx(1 / 17)


def test_evaluate_rejects_misaligned_input(fake_normalize):
    with pytest.raises(ValueError, match="hypotheses"):
        metrics.evaluate([rec()], [])


def test_nonexistent_api_rate(fake_normalize):
    index = {GROUPBY}
    records = [rec()] * 5
    hyps = [
        "use DataFrame.groupby here",
        "use data_frame.groupby here",
        "use Dataframe here",
        "use DataFrame.groupBy here",
        "use dataframe here",
    ]
    out = metrics.evaluate(records, hyps, index=index, n_boot=50)
    assert out["nonexistent_api_rate"] == pytest.approx(2 / 5)
    assert evaluate_one("use DataFrame")["nonexistent_api_rate"] is None
    assert evaluate_one("use DataFrame", index=index)["nonexistent_api_rate"] == 1.0


def test_slices_group_by_every_dimension(fake_normalize):
    records = [
        rec(library="pandas", engine="piper", style="explicit", speaker="p1", api_seen_in_train=True),
        rec(library="numpy", engine="human", style=None, speaker="m1", api_seen_in_train=False),
        rec(library="numpy", engine="human", style=None, speaker="m1", api_seen_in_train=False),
    ]
    hyps = [WRITTEN, WRITTEN, "use dataframe.groupby to split the data"]
    out = metrics.slices(records, hyps, n_boot=50)
    assert set(out) == set(metrics.SLICE_DIMENSIONS)
    assert out["library"]["pandas"]["api_em"] == 1.0
    assert out["library"]["numpy"]["api_em"] == 0.5
    assert out["library"]["numpy"]["n"] == 2
    assert out["style"]["none"]["n"] == 2
    assert out["api_seen_in_train"]["true"]["n"] == 1
    assert out["api_seen_in_train"]["false"]["api_em"] == 0.5


def test_bootstrap_ci_is_exact_for_perfect_system(fake_normalize):
    records = [rec()] * 4
    ci = metrics.bootstrap_ci(records, [WRITTEN] * 4, n=100)
    assert ci == {"api_em": [1.0, 1.0], "wer": [0.0, 0.0]}


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate(fake_normalize):
    records = [rec()] * 10
    hyps = [WRITTEN] * 6 + ["use dataframe.groupby to split the data"] * 4
    first = metrics.bootstrap_ci(records, hyps, n=200, seed=3)
    assert first == metrics.bootstrap_ci(records, hyps, n=200, seed=3)
    lo, hi = first["api_em"]
    assert 0.0 <= lo <= 0.6 <= hi <= 1.0


def test_evaluate_ci_matches_bootstrap_ci(fake_normalize):
    records = [rec()] * 6
    hyps = [WRITTEN] * 3 + ["nothing"] * 3
    out = metrics.evaluate(records, hyps, n_boot=100, seed=5)
    assert out["api_em_ci"] == metrics.bootstrap_ci(records, hyps, n=100, seed=5)["api_em"]


def test_paired_bootstrap(fake_normalize):
    records = [rec()] * 8
    good, bad = [WRITTEN] * 8, ["nothing"] * 8
    assert metrics.paired_bootstrap(records, good, bad, n=100) == 1.0
    assert metrics.paired_bootstrap(records, bad, good, n=100) == 0.0
    assert metrics.paired_bootstrap(records, good, good, n=100) == 0.5


# Tests below need data_io.load_split (fake_split) and no normalization.

def test_validate_preds_accepts_exact_ids(fake_split):
    metrics.validate_preds("dev", [{"id": i, "hyp": "x"} for i in "cab"])


@pytest.mark.parametrize("preds, message", [
    ([{"id": "a", "hyp": "x"}, {"id": "a", "hyp": "x"}, {"id": "b", "hyp": "x"}], "duplicate"),
    ([{"id": "a", "hyp": "x"}, {"id": "b", "hyp": "x"}], "missing"),
    ([{"id": i, "hyp": "x"} for i in "abcd"], "unknown"),
    ([{"id": "a", "hyp": "x"}, {"id": "b", "hyp": 3}, {"id": "c", "hyp": "x"}], "string"),
    ([{"id": "a", "hyp": "x"}, {"id": "b"}, {"id": "c", "hyp": "x"}], "string"),
])
def test_validate_preds_rejects_bad_files(fake_split, preds, message):
    with pytest.raises(ValueError, match=message):
        metrics.validate_preds("dev", preds)


# Tests below need load_split (fake_split) and write to a temporary results folder.

def test_write_preds_roundtrip(fake_split, results_dir):
    preds = [{"id": "a", "hyp": "one"}, {"id": "b", "hyp": "two"}, {"id": "c", "hyp": "three"}]
    path = metrics.write_preds("ctc", 42, "dev", preds)
    assert path == results_dir / "preds" / "ctc_seed42_dev.jsonl"
    assert [json.loads(line) for line in path.read_text().splitlines()] == preds


def test_write_preds_refuses_invalid_file(fake_split, results_dir):
    with pytest.raises(ValueError):
        metrics.write_preds("ctc", 42, "dev", [{"id": "a", "hyp": "one"}])
    assert not (results_dir / "preds").exists()


# Test below needs nothing from data_io.

def test_write_result_follows_run_file_contract(results_dir):
    path = metrics.write_result("ctc", 13, {"lr": 0.001}, 12.5, "cpu", 1000, 4, {"dev": {"api_em": 0.5}})
    assert path == results_dir / "ctc_seed13.json"
    run = json.loads(path.read_text())
    assert list(run) == ["model", "seed", "config", "train_time_s", "hardware", "n_params", "best_epoch", "splits"]
    assert run["splits"]["dev"]["api_em"] == 0.5
