"""Tests for src/gmm_hmm.py on toy chains and the audio fixtures."""
import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from src import data_io, features, gmm_hmm, metrics

FIXTURES = Path(__file__).parent / "fixtures"


def librosa_mfcc(wave: np.ndarray) -> np.ndarray:
    """Stand-in for features.mfcc until it lands: 13 MFCC + delta + delta-delta, 25 ms window, 10 ms hop."""
    m = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
    return np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).T.astype(np.float32)


@pytest.fixture(autouse=True)
def mfcc_stand_in(monkeypatch):
    monkeypatch.setattr(features, "mfcc", librosa_mfcc, raising=False)


@pytest.fixture(scope="module")
def mini() -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / "mini.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def cache(tmp_path, mini):
    path = gmm_hmm.cache_subdir(tmp_path)
    with gmm_hmm.mapper(1) as run_map:
        gmm_hmm.prepare(mini, path, run_map)
    return path


def test_state_sequence_wraps_characters_in_silence():
    a, b = data_io.CHAR2ID["а"], data_io.CHAR2ID["б"]
    sil = gmm_hmm.SIL
    expected = [sil * 3, sil * 3 + 1, sil * 3 + 2, a * 3, a * 3 + 1, a * 3 + 2, b * 3, b * 3 + 1, b * 3 + 2, sil * 3, sil * 3 + 1, sil * 3 + 2]
    assert gmm_hmm.state_sequence("аб").tolist() == expected


def test_forced_alignment_of_two_units_matches_hand_path():
    # Two units of three states; the hand-set emissions reward exactly this left-to-right path.
    hand_path = [0, 0, 1, 2, 3, 4, 4, 5]
    emit = np.full((8, 6), -10.0)
    emit[np.arange(8), hand_path] = 0.0
    half = np.full(6, np.log(0.5))
    path, score = gmm_hmm.forced_align(emit, np.arange(6), half, half)
    assert path.tolist() == hand_path
    assert score == pytest.approx(7 * np.log(0.5))


def test_standard_normal_density_at_zero():
    model = gmm_hmm.initial_model(39)
    lp = gmm_hmm.component_logprobs(model.compile(), np.zeros((1, 39)), 1)
    assert lp[0, 0, 0] == pytest.approx(-0.5 * 39 * np.log(2 * np.pi))


def test_decode_recovers_text_from_hand_emissions():
    seq = gmm_hmm.state_sequence("аб")
    emit = np.full((14, gmm_hmm.N_STATES), -10.0)
    for t, s in enumerate([*seq[:3], *seq[3:6], seq[5], *seq[6:9], seq[8], *seq[9:]]):
        emit[t, s] = 0.0
    half = np.full(gmm_hmm.N_STATES, np.log(0.5))
    model = gmm_hmm.Model(np.zeros((gmm_hmm.N_STATES, 1, 2)), np.ones((gmm_hmm.N_STATES, 1, 2)), np.zeros((gmm_hmm.N_STATES, 1)), half, half)
    lm = gmm_hmm.build_lm(["аб"])
    assert gmm_hmm.decode(emit, model, lm, 1.0, 0.0, np.inf) == "аб"


def test_lm_rows_are_distributions():
    lm = gmm_hmm.build_lm(["ол келді", "мен бардым"])
    assert np.exp(lm).sum(1) == pytest.approx(np.ones(gmm_hmm.N_CHARS + 1))


def test_log_likelihood_increases_over_three_iterations(cache, mini):
    items = [(r["id"], r["text"]) for r in mini]
    _, history = gmm_hmm.train(items, cache, seed=0, stages=(1,), iters=3)
    assert len(history) == 3
    assert history[0] < history[1] < history[2]


def test_log_likelihood_increases_within_a_split_stage(cache, mini):
    items = [(r["id"], r["text"]) for r in mini]
    _, history = gmm_hmm.train(items, cache, seed=0, stages=(1, 2), iters=3)
    assert history[3] < history[4] < history[5]


def test_same_seed_gives_identical_log_likelihoods(cache, mini):
    items = [(r["id"], r["text"]) for r in mini]
    first, h1 = gmm_hmm.train(items, cache, seed=7, stages=(1, 2), iters=2)
    second, h2 = gmm_hmm.train(items, cache, seed=7, stages=(1, 2), iters=2)
    assert h1 == h2
    assert np.array_equal(first.means, second.means)


def test_overfit_utterance_is_decoded_back(cache, mini):
    rec = mini[0]
    model, _ = gmm_hmm.train([(rec["id"], rec["text"])], cache, seed=0, stages=(1, 2, 4), iters=3)
    emit = gmm_hmm.logsumexp_last(gmm_hmm.component_logprobs(model.compile(), gmm_hmm.load_feats(cache, rec["id"]), model.n_components))
    lm = gmm_hmm.build_lm([rec["text"]])
    assert gmm_hmm.decode(emit, model, lm, 1.0, 0.0, np.inf) == rec["text"]


@pytest.fixture
def end_to_end(monkeypatch, tmp_path, mini):
    """Run main() on the fixtures with every split pointing at them and a guard on final splits."""
    def load_split(name, final=False):
        if name in data_io.FINAL_SPLITS and not final:
            raise PermissionError(name)
        return [dict(r, split=name) for r in mini]

    monkeypatch.setattr(data_io, "load_split", load_split)
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(gmm_hmm, "STAGES", (1, 2))
    monkeypatch.setattr(gmm_hmm, "ITERS_PER_STAGE", 1)
    monkeypatch.setattr(gmm_hmm, "LM_WEIGHTS", (1.0,))
    monkeypatch.setattr(gmm_hmm, "PENALTIES", (0.0, 1.0))
    return ["--workers", "1", "--cache-dir", str(tmp_path / "cache")], tmp_path / "results"


def test_run_without_final_writes_nothing(end_to_end):
    argv, results = end_to_end
    gmm_hmm.main(argv)
    assert not results.exists()


def test_run_with_final_writes_all_predictions_and_the_run_file(end_to_end):
    argv, results = end_to_end
    gmm_hmm.main(argv + ["--final", "--seed", "42"])
    for split in ("dev", "test", "test_talkshow", "test_read"):
        assert (results / "preds" / f"gmm_hmm_seed42_{split}.jsonl").is_file()
    run = json.loads((results / "gmm_hmm_seed42.json").read_text(encoding="utf-8"))
    assert run["model"] == "gmm_hmm" and run["best_epoch"] is None
    assert run["n_params"] == gmm_hmm.N_STATES * 2
    assert {"lm_weight", "insertion_penalty", "selection_metric", "train_hours_used", "skipped_utterances"} <= run["config"].keys()
