"""Tests for src/models.py."""
import json
from pathlib import Path

import pytest
import torch

from src import data_io, features, models, train_scratch

FIXTURES = Path(__file__).parent / "fixtures"


def small_ctc(n_chars: int = 10, dropout: float = 0.0) -> models.CTCModel:
    torch.manual_seed(0)
    return models.build_model("ctc", in_dim=80, n_chars=n_chars, hidden=32, layers=2, dropout=dropout)


@pytest.mark.parametrize("ids, expected", [
    ([], []),
    ([0, 0, 0], []),
    ([3, 3, 3], [3]),
    ([1, 1, 0, 1], [1, 1]),
    ([1, 0, 0, 2, 2, 0, 2], [1, 2, 2]),
    ([0, 5, 0, 5, 5, 6, 0], [5, 5, 6]),
])
def test_ctc_collapse(ids, expected):
    assert models.ctc_collapse(ids) == expected


def test_ctc_collapse_with_other_blank():
    assert models.ctc_collapse([4, 4, 7, 4, 2, 2, 7], blank=7) == [4, 4, 2]


def test_encode_decode_roundtrip():
    text = data_io.CHARS[:6]
    ids = models.encode_text(text)
    assert min(ids) == 1
    assert models.decode_ids(ids) == text


def test_encode_rejects_unknown_character():
    with pytest.raises(ValueError, match="outside data_io.CHARS"):
        models.encode_text("€")


@pytest.mark.parametrize("target, expected", [([1, 2, 3], 3), ([1, 1], 3), ([2, 2, 2, 3], 6)])
def test_ctc_min_frames(target, expected):
    assert models.ctc_min_frames(target) == expected


def test_subsampled_lengths_match_conv_output():
    sub = models.Subsampler(80, 8)
    for n in range(1, 30):
        x, lengths = sub(torch.zeros(1, n, 80), torch.tensor([n]))
        assert x.shape[1] == lengths.item() == models.subsampled_lengths(torch.tensor(n)).item()


def test_output_shapes_and_probabilities():
    model = small_ctc().eval()
    feats, lengths = torch.randn(3, 101, 80), torch.tensor([101, 60, 7])
    log_probs, out_lengths = model(feats, lengths)
    assert log_probs.shape == (3, 26, 11)
    assert out_lengths.tolist() == [26, 15, 2]
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(3, 26))


def test_padding_does_not_change_output():
    model = small_ctc().eval()
    short = torch.randn(1, 37, 80)
    batch = torch.zeros(2, 100, 80)
    batch[0] = torch.randn(100, 80)
    batch[1, :37] = short[0]
    alone, n_alone = model(short, torch.tensor([37]))
    together, n_together = model(batch, torch.tensor([100, 37]))
    assert n_alone.item() == n_together[1].item() == 10
    torch.testing.assert_close(together[1, :10], alone[0], atol=1e-5, rtol=1e-5)


def synthetic_batch(n_chars: int = 10) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator().manual_seed(1)
    feats = torch.randn(4, 120, 80, generator=gen)
    lengths = torch.tensor([120, 100, 90, 80])
    targets = torch.randint(1, n_chars + 1, (4, 6), generator=gen)
    return feats, lengths, targets, torch.full((4,), 6)


def test_one_step_lowers_loss():
    model = small_ctc()
    batch = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.loss(*batch)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        assert model.loss(*batch).item() < before.item()


def test_memorizes_a_tiny_set():
    model = small_ctc()
    feats, lengths, targets, target_lengths = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(250):
        optimizer.zero_grad()
        loss = model.loss(feats, lengths, targets, target_lengths)
        loss.backward()
        optimizer.step()
    model.eval()
    assert loss.item() < 0.05
    assert model.decode(feats, lengths) == targets.tolist()


def fixture_batch() -> tuple[torch.Tensor, ...]:
    lines = (FIXTURES / "mini.jsonl").read_text(encoding="utf-8").splitlines()
    records = [r for r in map(json.loads, lines) if r["split"] == "train"]
    items = [(torch.from_numpy(features.logmel(data_io.load_audio(r))), models.encode_text(r["text"]))
             for r in records]
    return train_scratch.collate(items)


@pytest.mark.parametrize("head", models.HEADS)
def test_one_step_lowers_loss_on_fixture(head):
    torch.manual_seed(0)
    model = models.build_model(head, in_dim=80, max_len=300, hidden=32, layers=2, dropout=0.0)
    batch = fixture_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.loss(*batch)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        assert model.loss(*batch).item() < before.item()


def test_build_model_rejects_unknown_head():
    with pytest.raises(ValueError, match="unknown head"):
        models.build_model("transformer", in_dim=80, n_chars=10)


@pytest.mark.parametrize("budget_s, epoch_s, expected", [(None, 99.0, 40), (600.0, 100.0, 6), (50.0, 100.0, 1),
                                                         (1e6, 1.0, 40)])
def test_planned_epochs_fit_the_budget(budget_s, epoch_s, expected):
    assert train_scratch.planned_epochs(40, budget_s, epoch_s) == expected


def test_shortened_cosine_reaches_zero_at_the_new_total():
    schedule = train_scratch.WarmupCosine(warmup=10, total=1000)
    assert schedule(0) == pytest.approx(0.1) and schedule(9) == pytest.approx(1.0)
    schedule.total = 110
    assert schedule(60) == pytest.approx(0.5)
    assert schedule(110) == pytest.approx(0.0, abs=1e-12) and schedule(500) == pytest.approx(0.0, abs=1e-12)


def small_aed(n_chars: int = 10, max_len: int = 12) -> models.AEDModel:
    torch.manual_seed(0)
    return models.build_model("aed", in_dim=80, n_chars=n_chars, max_len=max_len, hidden=32, layers=2,
                              dropout=0.0)


def test_aed_needs_max_len():
    with pytest.raises(ValueError, match="max_len"):
        models.build_model("aed", in_dim=80, n_chars=10)


def test_aed_logit_shapes():
    model = small_aed()
    feats, lengths, targets, _ = synthetic_batch()
    assert model(feats, lengths, targets).shape == (4, 7, 11)


def test_attention_weights_sum_to_one_and_skip_padding():
    model = small_aed().eval()
    h, keys, mask, state = model._encode(torch.randn(2, 100, 80), torch.tensor([100, 37]))
    _, _, weights = model._step(torch.full((2,), model.sos), state, h, keys, mask)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    assert torch.all(weights[1, 10:] == 0)


def test_decoding_stops_at_eos():
    model = small_aed().eval()
    feats, lengths = torch.randn(3, 80, 80), torch.tensor([80, 60, 40])
    with torch.no_grad():
        model.output.bias.fill_(0.0)
        model.output.bias[models.EOS] = 1e4
    assert model.decode(feats, lengths) == [[], [], []]
    with torch.no_grad():
        model.output.bias[models.EOS] = -1e4
        model.output.bias[5] = 1e4
    assert model.decode(feats, lengths) == [[5] * 12] * 3


def test_aed_one_step_lowers_loss():
    model = small_aed()
    batch = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.loss(*batch)
    before.backward()
    optimizer.step()
    with torch.no_grad():
        assert model.loss(*batch).item() < before.item()


def small_rnnt(n_chars: int = 10) -> models.RNNTModel:
    torch.manual_seed(0)
    return models.build_model("rnnt", in_dim=80, n_chars=n_chars, hidden=32, layers=2, dropout=0.0)


def test_rnnt_joint_shapes():
    model = small_rnnt()
    feats, lengths, targets, _ = synthetic_batch()
    logits, out_lengths = model(feats, lengths, targets)
    assert logits.shape == (4, 30, 7, 11)
    assert out_lengths.tolist() == [30, 25, 23, 20]


def test_rnnt_tiny_input_loss_decreases():
    model = small_rnnt()
    batch = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = model.loss(*batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert all(b < a for a, b in zip(losses, losses[1:]))


def test_rnnt_loss_ignores_target_padding():
    model = small_rnnt()
    feats, lengths, targets, target_lengths = synthetic_batch()
    target_lengths = torch.tensor([6, 4, 3, 2])
    padded = targets.clone()
    for b, n in enumerate(target_lengths.tolist()):
        padded[b, n:] = 0
    wider = torch.cat([padded, torch.zeros(4, 3, dtype=torch.long)], dim=1)
    torch.testing.assert_close(model.loss(feats, lengths, padded, target_lengths),
                               model.loss(feats, lengths, wider, target_lengths))


def test_rnnt_decoding_emits_at_most_five_symbols_per_frame():
    model = small_rnnt().eval()
    feats, lengths = torch.randn(2, 40, 80), torch.tensor([40, 21])
    with torch.no_grad():
        model.output.bias.fill_(0.0)
        model.output.bias[models.BLANK] = 1e4
    assert model.decode(feats, lengths) == [[], []]
    with torch.no_grad():
        model.output.bias[models.BLANK] = -1e4
        model.output.bias[3] = 1e4
    assert model.decode(feats, lengths) == [[3] * 50, [3] * 30]


def test_aed_memorizes_a_tiny_set():
    model = small_aed()
    feats, lengths, targets, target_lengths = synthetic_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for _ in range(150):
        optimizer.zero_grad()
        model.loss(feats, lengths, targets, target_lengths).backward()
        optimizer.step()
    model.eval()
    assert model.decode(feats, lengths) == targets.tolist()
