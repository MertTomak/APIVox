"""Tests for src/metrics.py with hand-computed examples."""
import json

import pytest

from src import data_io, metrics

HAS_NORMALIZE = hasattr(data_io, "normalize_text")
needs_b0 = pytest.mark.skipif(not HAS_NORMALIZE, reason="needs data_io.normalize_text from B0")


def rec(text, tags=None, cs=False, source="radio", program="radio_01", id="u"):
    tags = tags or ["kz"] * len(text.split())
    return {"id": id, "text": text, "lang_tags": tags, "cs": cs, "source": source, "program": program}


@pytest.fixture(autouse=True)
def normalize_stand_in(monkeypatch):
    """Until B0 lands, stand in for normalize_text with space collapsing, which leaves normalized text unchanged."""
    if not HAS_NORMALIZE:
        monkeypatch.setattr(data_io, "normalize_text", lambda text: " ".join(text.split()), raising=False)


@pytest.fixture
def fake_split(monkeypatch):
    """Stand-in for data_io.load_split returning three records."""
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    monkeypatch.setattr(data_io, "load_split", lambda name, final=False: rows)


@pytest.fixture
def results_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "RESULTS_DIR", tmp_path)
    return tmp_path


def test_one_substitution_in_four_words_is_wer_quarter():
    out = metrics.evaluate([rec("мен бүгін үйге барамын")], ["мен бүгін үйде барамын"])
    assert out["wer"] == 0.25
    assert out["n_utts"] == 1
    assert out["n_ref_words"] == 4


def test_kazakh_suffix_error_cer():
    # "балаларға" vs "балаларда": one substituted letter out of 9 characters.
    out = metrics.evaluate([rec("балаларға")], ["балаларда"])
    assert out["cer"] == pytest.approx(1 / 9)
    assert out["wer"] == 1.0


def test_cer_counts_spaces_as_characters():
    # "ол келді" has 8 characters; dropping the space is one deletion.
    assert metrics.evaluate([rec("ол келді")], ["олкелді"])["cer"] == pytest.approx(1 / 8)


@needs_b0
def test_hesitation_in_hypothesis_is_not_an_insertion():
    out = metrics.evaluate([rec("мен барамын")], ["мен ммм барамын"])
    assert out["wer"] == 0.0


@needs_b0
def test_hypothesis_is_normalized_before_scoring():
    out = metrics.evaluate([rec("мен барамын")], ["Мен барамын!"])
    assert out["wer"] == 0.0
    assert out["cer"] == 0.0


def test_ru_word_acc_on_hand_aligned_code_switched_pair():
    # ref:  мен  вчера  магазинге  бардым   tags kz ru mix kz
    # hyp:  мин  вчера  магазинда  бардым   -> ru/mix words: вчера equal, магазинге substituted
    record = rec("мен вчера магазинге бардым", ["kz", "ru", "mix", "kz"], cs=True)
    out = metrics.evaluate([record], ["мин вчера магазинда бардым"])
    assert out["ru_word_acc"] == 0.5
    assert out["wer"] == 0.5


def test_ru_word_acc_follows_alignment_after_insertion():
    # An inserted word shifts hypothesis positions but not reference positions.
    record = rec("мен вчера магазинге бардым", ["kz", "ru", "mix", "kz"], cs=True)
    out = metrics.evaluate([record], ["ну мен вчера магазинге бардым"])
    assert out["ru_word_acc"] == 1.0
    assert out["wer"] == 0.25


def test_ru_word_acc_is_none_without_russian_words():
    assert metrics.evaluate([rec("мен барамын")], ["мен барамын"])["ru_word_acc"] is None


def test_corpus_wer_differs_from_mean_of_utterance_wers():
    # Utterance 1: 1 error / 1 word (WER 1.0); utterance 2: 0 errors / 9 words (WER 0.0).
    long = "бұл сөйлем тоғыз сөзден тұрады және оны дұрыс таныдық"
    records = [rec("иә", id="a"), rec(long, id="b")]
    out = metrics.evaluate(records, ["жоқ", long])
    assert out["wer"] == pytest.approx(1 / 10)
    assert out["wer"] != pytest.approx((1.0 + 0.0) / 2)


def test_cs_and_mono_wer_split_by_cs_flag():
    records = [
        rec("мен вчера бардым", ["kz", "ru", "kz"], cs=True, id="a"),
        rec("мен үйге барамын", id="b"),
        rec("ол келді", id="c"),
    ]
    out = metrics.evaluate(records, ["мен вчера бардым", "мен үйде барамын", "ол кетті"])
    assert out["cs_wer"] == 0.0
    assert out["mono_wer"] == pytest.approx(2 / 5)
    assert out["wer"] == pytest.approx(2 / 8)


def test_cs_wer_is_none_without_code_switched_records():
    assert metrics.evaluate([rec("мен барамын")], ["мен барамын"])["cs_wer"] is None


def test_empty_hypothesis_is_all_deletions():
    out = metrics.evaluate([rec("мен барамын")], [""])
    assert out["wer"] == 1.0
    assert out["cer"] == 1.0


def test_evaluate_requires_normalize_text_from_b0(monkeypatch):
    monkeypatch.delattr(data_io, "normalize_text", raising=False)
    with pytest.raises(RuntimeError, match="B0"):
        metrics.evaluate([rec("мен барамын")], ["мен барамын"])


def test_evaluate_rejects_misaligned_input():
    with pytest.raises(ValueError, match="hypotheses"):
        metrics.evaluate([rec("мен")], [])


def test_evaluate_rejects_unnormalized_reference():
    with pytest.raises(ValueError, match="not normalized"):
        metrics.evaluate([rec("мен  барамын", ["kz", "kz"])], ["мен барамын"])


def test_evaluate_rejects_empty_reference():
    with pytest.raises(ValueError, match="empty reference"):
        metrics.evaluate([rec("", [])], ["мен"])


def test_evaluate_rejects_tag_count_mismatch():
    with pytest.raises(ValueError, match="lang_tags"):
        metrics.evaluate([rec("мен барамын", ["kz"])], ["мен барамын"])


def test_slices_group_by_source_program_and_cs():
    records = [
        rec("мен вчера бардым", ["kz", "ru", "kz"], cs=True, source="podcasts", program="podcasts_01", id="a"),
        rec("мен үйге барамын", source="radio", program="radio_01", id="b"),
        rec("ол келді", source="radio", program="radio_02", id="c"),
    ]
    out = metrics.slices(records, ["мен вчера бардым", "мен үйде барамын", "ол келді"])
    assert set(out) == {"source", "program", "cs"}
    assert out["source"]["podcasts"]["wer"] == 0.0
    assert out["source"]["radio"]["wer"] == pytest.approx(1 / 5)
    assert out["source"]["radio"]["n_utts"] == 2
    assert out["program"]["radio_01"]["wer"] == pytest.approx(1 / 3)
    assert out["cs"]["true"]["n_utts"] == 1
    assert out["cs"]["false"]["wer"] == pytest.approx(1 / 5)


def test_bootstrap_ci_is_exact_for_perfect_system():
    records = [rec("мен вчера бардым", ["kz", "ru", "kz"], cs=True, id=str(i)) for i in range(4)]
    ci = metrics.bootstrap_ci(records, ["мен вчера бардым"] * 4, n=100)
    assert ci == {"wer": [0.0, 0.0], "cer": [0.0, 0.0], "cs_wer": [0.0, 0.0]}


def test_bootstrap_ci_is_seeded_and_brackets_the_estimate():
    records = [rec("мен үйге барамын", id=str(i)) for i in range(10)]
    hyps = ["мен үйге барамын"] * 6 + ["мен үйде барамын"] * 4
    first = metrics.bootstrap_ci(records, hyps, n=200, seed=3)
    assert first == metrics.bootstrap_ci(records, hyps, n=200, seed=3)
    lo, hi = first["wer"]
    assert 0.0 <= lo <= metrics.evaluate(records, hyps)["wer"] <= hi <= 1.0
    assert first["cs_wer"] is None


def test_paired_bootstrap():
    records = [rec("мен үйге барамын", id=str(i)) for i in range(8)]
    good, bad = ["мен үйге барамын"] * 8, ["жоқ"] * 8
    assert metrics.paired_bootstrap(records, good, bad, n=100) == 1.0
    assert metrics.paired_bootstrap(records, bad, good, n=100) == 0.0
    assert metrics.paired_bootstrap(records, good, good, n=100) == 0.5


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


def test_write_preds_roundtrip(fake_split, results_dir):
    preds = [{"id": "a", "hyp": "бір"}, {"id": "b", "hyp": "екі"}, {"id": "c", "hyp": "үш"}]
    path = metrics.write_preds("ctc", 42, "dev", preds)
    assert path == results_dir / "preds" / "ctc_seed42_dev.jsonl"
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == preds


def test_write_preds_refuses_invalid_file(fake_split, results_dir):
    with pytest.raises(ValueError):
        metrics.write_preds("ctc", 42, "dev", [{"id": "a", "hyp": "бір"}])
    assert not (results_dir / "preds").exists()


def test_write_result_follows_run_file_contract(results_dir):
    path = metrics.write_result(model="ctc", seed=13, checkpoint=None, revision=None, config={"lr": 0.001},
                                train_time_s=12.5, hardware="cpu", n_params=1000, best_epoch=4,
                                splits={"dev": {"wer": 0.5}})
    assert path == results_dir / "ctc_seed13.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    assert list(run) == ["model", "seed", "checkpoint", "revision", "config", "train_time_s", "hardware",
                         "n_params", "best_epoch", "splits"]
    assert run["checkpoint"] is None and run["revision"] is None
    assert run["splits"]["dev"]["wer"] == 0.5


def test_write_result_rejects_positional_arguments(results_dir):
    with pytest.raises(TypeError):
        metrics.write_result("ctc", 13, {}, 0.0, "cpu", 1, None, {})
