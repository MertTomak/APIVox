"""Compute log-mel and MFCC features and audio augmentation."""
import argparse

import librosa
import numpy as np

SAMPLE_RATE = 16000
WIN_LENGTH = 400
HOP_LENGTH = 160
N_MFCC = 13
N_FFT = 512
N_MELS = 80
LOG_FLOOR = 1e-10
STD_FLOOR = 1e-5
SPEED_FACTORS = (0.9, 1.0, 1.1)
SNR_DB_RANGE = (10.0, 30.0)
N_FREQ_MASKS, FREQ_MASK_MAX = 2, 15
N_TIME_MASKS, TIME_MASK_MAX = 2, 40


def frame_count(n_samples: int) -> int:
    """Number of log-mel frames T = 1 + floor((n - 400) / 160) for a clip of n >= 400 samples."""
    if n_samples < WIN_LENGTH:
        raise ValueError(f"need at least {WIN_LENGTH} samples (25 ms), got {n_samples}")
    return 1 + (n_samples - WIN_LENGTH) // HOP_LENGTH


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    """HTK mel scale m = 2595 * log10(1 + f / 700)."""
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    """Inverse HTK mel scale f = 700 * (10^(m / 2595) - 1)."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank() -> np.ndarray:
    """80 triangular filters with centers equally spaced on the mel scale from 0 to 8 kHz, shape [80, 257]."""
    edges = mel_to_hz(np.linspace(0.0, hz_to_mel(SAMPLE_RATE / 2), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, d=1.0 / SAMPLE_RATE)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    rising = (bins - lower) / (center - lower)
    falling = (upper - bins) / (upper - center)
    return np.maximum(0.0, np.minimum(rising, falling))


_MEL_FB = mel_filterbank()
_WINDOW = np.hanning(WIN_LENGTH + 1)[:-1]  # periodic Hann


def _check_wave(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave)
    if wave.ndim != 1:
        raise ValueError(f"expected a 1-D mono waveform, got shape {wave.shape}")
    if not np.all(np.isfinite(wave)):
        raise ValueError("waveform contains NaN or infinite samples")
    frame_count(len(wave))
    return wave.astype(np.float64)


def cmvn(feats: np.ndarray) -> np.ndarray:
    """Per-utterance normalization x' = (x - mean_t(x)) / max(std_t(x), 1e-5) for every dimension."""
    std = np.maximum(feats.std(axis=0), STD_FLOOR)
    return ((feats - feats.mean(axis=0)) / std).astype(np.float32)


def logmel(wave: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Log-mel features x[t, j] = log(sum_k |FFT_512(hann * frame_t)[k]|^2 * mel[j, k]), then CMVN, shape [T, 80]."""
    wave = _check_wave(wave)
    frames = np.lib.stride_tricks.sliding_window_view(wave, WIN_LENGTH)[::HOP_LENGTH]
    power = np.abs(np.fft.rfft(frames * _WINDOW, n=N_FFT)) ** 2
    feats = np.log(np.maximum(power @ _MEL_FB.T, LOG_FLOOR))
    return cmvn(feats) if normalize else feats.astype(np.float32)


def mfcc(wave: np.ndarray) -> np.ndarray:
    """Return 13 MFCC plus delta and delta-delta as float32[T, 39]."""
    base = librosa.feature.mfcc(
        y=wave, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=WIN_LENGTH, hop_length=HOP_LENGTH,
    )
    d1 = librosa.feature.delta(base)
    d2 = librosa.feature.delta(base, order=2)
    return np.vstack([base, d1, d2]).T.astype(np.float32)


def augment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Speed perturbation y[i] = x[i * s] with s in {0.9, 1.0, 1.1}, then white noise at SNR ~ U(10, 30) dB."""
    wave = _check_wave(wave)
    factor = float(rng.choice(SPEED_FACTORS))
    n_out = int(round(len(wave) / factor))
    perturbed = np.interp(np.arange(n_out) * factor, np.arange(len(wave)), wave)
    snr_db = rng.uniform(*SNR_DB_RANGE)
    noise_power = np.mean(perturbed ** 2) / 10.0 ** (snr_db / 10.0)
    noisy = perturbed + rng.normal(0.0, np.sqrt(noise_power), n_out)
    return np.clip(noisy, -1.0, 1.0).astype(np.float32)


def _mask(out: np.ndarray, axis: int, n_masks: int, max_width: int, rng: np.random.Generator) -> None:
    size = out.shape[axis]
    for _ in range(n_masks):
        width = int(rng.integers(0, min(max_width, size) + 1))
        start = int(rng.integers(0, size - width + 1))
        index = [slice(None), slice(None)]
        index[axis] = slice(start, start + width)
        out[tuple(index)] = 0.0  # 0 is the utterance mean after CMVN


def spec_augment(feats: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """SpecAugment: zero 2 feature bands of width f ~ U{0..15} and 2 frame spans of width t ~ U{0..40}."""
    if feats.ndim != 2:
        raise ValueError(f"expected features of shape [T, D], got shape {feats.shape}")
    out = feats.copy()
    _mask(out, 1, N_FREQ_MASKS, FREQ_MASK_MAX, rng)
    _mask(out, 0, N_TIME_MASKS, TIME_MASK_MAX, rng)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the frame count and feature shapes for a 440 Hz test tone of the given length."
    )
    parser.add_argument("--seconds", type=float, default=1.0, help="tone length in seconds (default: 1.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t = np.arange(int(args.seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    print(f"samples={len(wave)} frames={frame_count(len(wave))} "
          f"logmel={logmel(wave).shape} mfcc={mfcc(wave).shape}")


if __name__ == "__main__":
    main()
