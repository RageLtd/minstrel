from __future__ import annotations

from typing import TypedDict

import librosa
import numpy as np

from .clap import CLAP_SR


class Scalars(TypedDict):
    bpm: float
    rms_energy: float
    spectral_centroid: float


def load_audio(path: str, sr: int = CLAP_SR) -> np.ndarray:
    """Decode to mono at the target rate. soundfile handles FLAC natively."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def extract(audio: np.ndarray, sr: int = CLAP_SR) -> Scalars:
    """Cheap, interpretable knobs: tempo, energy, brightness."""
    tempo = librosa.beat.beat_track(y=audio, sr=sr)[0]
    return Scalars(
        bpm=float(np.atleast_1d(tempo)[0]),
        rms_energy=float(np.mean(librosa.feature.rms(y=audio))),
        spectral_centroid=float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        ),
    )
