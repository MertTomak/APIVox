"""Tests for src/features.py."""
import json
from pathlib import Path

import librosa
import numpy as np
import pytest

from src import features
from src.data_io import load_audio
from src.features import HOP_LENGTH, SAMPLE_RATE, mfcc

SR = features.SAMPLE_RATE


def tone(seconds: float = 1.0, hz: float = 440.0, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def noise(n: int = 24000, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, n).astype(np.float32)


def _wave(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)


@pytest.mark.parametrize("n, expected", [(400, 1), (559, 1), (560, 2), (16000, 98), (24000, 148)])
def test_frame_count_formula(n, expected):
    assert features.frame_count(n) == expected
    assert features.logmel(noise(n)).shape[0] == expected


def test_logmel_shape_and_dtype():
    feats = features.logmel(noise())
    assert feats.shape == (148, 80)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


@pytest.mark.parametrize("wave", [np.zeros(399, np.float32), np.zeros((2, 16000), np.float32),
                                  np.full(16000, np.nan, np.float32)])
def test_bad_waveform_raises(wave):
    with pytest.raises(ValueError):
        features.logmel(wave)


def test_cmvn_gives_zero_mean_unit_std():
    for feats in (features.logmel(noise()), features.cmvn(features.mfcc(noise()))):
        assert feats.dtype == np.float32
        np.testing.assert_allclose(feats.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(feats.std(axis=0), 1.0, atol=1e-3)


def test_filterbank_covers_every_band():
    fb = features.mel_filterbank()
    assert fb.shape == (80, features.N_FFT // 2 + 1)
    assert np.all(fb >= 0) and np.all(fb.sum(axis=1) > 0)


def test_440_hz_peaks_in_expected_mel_bin():
    centers_mel = np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 82)[1:-1]
    expected = int(np.argmin(np.abs(centers_mel - 2595 * np.log10(1 + 440 / 700))))
    raw = features.logmel(tone(), normalize=False)
    assert np.all(raw.argmax(axis=1) == expected)


def test_augment_is_deterministic_with_seeded_rng():
    wave = tone()
    a = features.augment(wave, np.random.default_rng(42))
    b = features.augment(wave, np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and np.all(np.abs(a) <= 1.0)


def test_augment_lengths_and_snr():
    wave = tone()
    allowed = {round(len(wave) / s) for s in features.SPEED_FACTORS}
    seen = set()
    for seed in range(40):
        out = features.augment(wave, np.random.default_rng(seed))
        assert len(out) in allowed
        seen.add(len(out))
        if len(out) == len(wave):
            residual = out.astype(np.float64) - wave
            snr = 10 * np.log10(np.mean(wave.astype(np.float64) ** 2) / np.mean(residual ** 2))
            assert 9.5 <= snr <= 30.5
    assert seen == allowed


def test_spec_augment_masks_whole_bands_and_spans():
    feats = np.ones((300, 80), dtype=np.float32)
    out = features.spec_augment(feats, np.random.default_rng(13))
    np.testing.assert_array_equal(feats, 1.0)
    zero_cols = np.all(out == 0, axis=0)
    zero_rows = np.all(out == 0, axis=1)
    assert zero_cols.sum() <= 2 * features.FREQ_MASK_MAX
    assert zero_rows.sum() <= 2 * features.TIME_MASK_MAX
    assert np.all((out == 1) | zero_cols[None, :] | zero_rows[:, None])


def test_spec_augment_is_deterministic_and_handles_short_input():
    feats = features.logmel(noise(4000))
    a = features.spec_augment(feats, np.random.default_rng(2026))
    b = features.spec_augment(feats, np.random.default_rng(2026))
    np.testing.assert_array_equal(a, b)
    assert a.shape == feats.shape


def test_spec_augment_rejects_1d_input():
    with pytest.raises(ValueError):
        features.spec_augment(np.zeros(10, np.float32), np.random.default_rng(0))


def test_mfcc_shape_and_dtype() -> None:
    feats = mfcc(_wave(1.0))
    assert feats.dtype == np.float32
    assert feats.ndim == 2 and feats.shape[1] == 39


def test_mfcc_frame_count() -> None:
    wave = _wave(2.0)
    assert mfcc(wave).shape[0] == 1 + len(wave) // HOP_LENGTH


def test_mfcc_short_segment() -> None:
    assert mfcc(_wave(0.5)).shape == (51, 39)


def test_mfcc_matches_gmm_hmm_stand_in_on_fixture_clip() -> None:
    first = (Path(__file__).parent / "fixtures" / "mini.jsonl").read_text(encoding="utf-8").splitlines()[0]
    wave = load_audio(json.loads(first))
    m = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
    expected = np.vstack(
        [m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]
    ).T.astype(np.float32)
    np.testing.assert_array_equal(mfcc(wave), expected)
