from __future__ import annotations

import io
from typing import TypedDict

import librosa
import numpy as np

from .clap import CLAP_SR


class Scalars(TypedDict):
    bpm: float
    rms_energy: float
    spectral_centroid: float
    rhythmic_irregularity: float
    timbral_complexity: float
    dynamic_contrast: float
    harmonic_instability: float
    percussive_ratio: float
    spectral_flatness: float
    zero_crossing_rate: float


FEATURE_VERSION = 3
BASE_SCALARS = frozenset({"bpm", "rms_energy", "spectral_centroid"})
N_FFT = 2048
HOP_LENGTH = 1024


def _rhythmic_irregularity(onset_envelope: np.ndarray, sr: int) -> float:
    max_lag = min(len(onset_envelope), max(2, round(8 * sr / HOP_LENGTH)))
    if max_lag <= 2 or not np.any(onset_envelope):
        return 0.0
    autocorrelation = librosa.autocorrelate(onset_envelope, max_size=max_lag)
    zero_lag = float(autocorrelation[0])
    if zero_lag <= np.finfo(np.float32).eps:
        return 0.0
    min_lag = max(1, round(0.2 * sr / HOP_LENGTH))
    if min_lag >= len(autocorrelation):
        return 0.0
    pulse_clarity = float(np.max(autocorrelation[min_lag:])) / zero_lag
    return 1 - float(np.clip(pulse_clarity, 0, 1))


def _harmonic_instability(chroma: np.ndarray, sr: int) -> float:
    frames_per_block = max(1, round(sr / HOP_LENGTH))
    block_count = chroma.shape[1] // frames_per_block
    if block_count < 2:
        return 0.0
    blocked = chroma[:, : block_count * frames_per_block].reshape(
        chroma.shape[0], block_count, frames_per_block
    )
    blocked = np.mean(blocked, axis=2)
    norms = np.linalg.norm(blocked, axis=0)
    valid = (norms[:-1] > 0) & (norms[1:] > 0)
    if not np.any(valid):
        return 0.0
    similarities = np.sum(blocked[:, :-1] * blocked[:, 1:], axis=0) / np.maximum(
        norms[:-1] * norms[1:], np.finfo(np.float32).eps
    )
    return float(np.mean(1 - np.clip(similarities[valid], 0, 1)))


def load_audio_bytes(data: bytes, sr: int = CLAP_SR) -> np.ndarray:
    """Decode in-memory audio bytes to mono at the target rate, then discard.

    The original download from Navidrome is read straight from a buffer and never
    written to disk.
    """
    y, _ = librosa.load(io.BytesIO(data), sr=sr, mono=True)
    return y.astype(np.float32)


def extract(audio: np.ndarray, sr: int = CLAP_SR) -> Scalars:
    """Interpretable tempo, dynamics, texture, rhythm, and harmony descriptors."""
    magnitude = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    power = np.square(magnitude)
    rms = librosa.feature.rms(S=magnitude)
    log_spectrogram = librosa.amplitude_to_db(magnitude, ref=np.max)
    onset_envelope = librosa.onset.onset_strength(
        S=log_spectrogram, sr=sr, hop_length=HOP_LENGTH
    )
    tempo = librosa.beat.beat_track(
        onset_envelope=onset_envelope, sr=sr, hop_length=HOP_LENGTH
    )[0]
    contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, tuning=0.0)
    mel_power = librosa.feature.melspectrogram(S=power, sr=sr, n_mels=128)
    harmonic, percussive = librosa.decompose.hpss(mel_power)
    harmonic_energy = float(np.sum(harmonic))
    percussive_energy = float(np.sum(percussive))
    rms_values = rms.ravel()
    rms_high = float(np.percentile(rms_values, 95))
    rms_low = float(np.percentile(rms_values, 10))
    return Scalars(
        bpm=float(np.atleast_1d(tempo)[0]),
        rms_energy=float(np.mean(rms_values)),
        spectral_centroid=float(
            np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=sr))
        ),
        rhythmic_irregularity=_rhythmic_irregularity(onset_envelope, sr),
        timbral_complexity=float(np.mean(np.std(contrast, axis=1))),
        dynamic_contrast=float(
            (rms_high - rms_low) / max(rms_high, np.finfo(np.float32).eps)
        ),
        harmonic_instability=_harmonic_instability(chroma, sr),
        percussive_ratio=float(
            percussive_energy
            / max(harmonic_energy + percussive_energy, np.finfo(np.float32).eps)
        ),
        spectral_flatness=float(
            np.clip(np.mean(librosa.feature.spectral_flatness(S=magnitude)), 0, 1)
        ),
        zero_crossing_rate=float(
            np.mean(
                librosa.feature.zero_crossing_rate(
                    audio, frame_length=N_FFT, hop_length=HOP_LENGTH
                )
            )
        ),
    )
