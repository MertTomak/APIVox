"""Tests for src/features.py."""
import json
from pathlib import Path

import librosa
import numpy as np

from src.data_io import load_audio
from src.features import HOP_LENGTH, SAMPLE_RATE, mfcc


def _wave(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal(int(seconds * SAMPLE_RATE)).astype(np.float32)


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
    first = (Path(__file__).parent / "fixtures" / "mini.jsonl").read_text().splitlines()[0]
    wave = load_audio(json.loads(first))
    m = librosa.feature.mfcc(y=wave, sr=16000, n_mfcc=13, n_fft=400, hop_length=160)
    expected = np.vstack(
        [m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]
    ).T.astype(np.float32)
    np.testing.assert_array_equal(mfcc(wave), expected)
