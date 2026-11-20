"""Compute log-mel and MFCC features and audio augmentation."""
import argparse

import librosa
import numpy as np

SAMPLE_RATE = 16000
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MFCC = 13


def mfcc(wave: np.ndarray) -> np.ndarray:
    """Return 13 MFCC plus delta and delta-delta as float32[T, 39]."""
    base = librosa.feature.mfcc(
        y=wave, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
    )
    d1 = librosa.feature.delta(base)
    d2 = librosa.feature.delta(base, order=2)
    return np.vstack([base, d1, d2]).T.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> None:
    parse_args()


if __name__ == "__main__":
    main()
