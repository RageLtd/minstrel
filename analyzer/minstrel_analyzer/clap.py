from __future__ import annotations

import os

import numpy as np
import torch
import transformers
from transformers import (
    ClapAudioModelWithProjection,
    ClapProcessor,
    ClapTextModelWithProjection,
)

from .device import select_device

# Loading the full checkpoint into each single-tower projection head reports the
# other tower's weights as "unexpected" — harmless, so silence it.
transformers.logging.set_verbosity_error()

# CLAP expects 48 kHz mono audio.
CLAP_SR = 48_000
EMBED_DIM = 512

# The HF feature extractor silently takes ONE random 10 s crop of anything
# longer (truncation="rand_trunc", max_length_s=10), so a whole track must be
# tiled into windows no longer than that and embedded piecewise.
SEGMENT_WINDOW_S = 10
SEGMENT_HOP_S = 5
# A leftover tail shorter than this is dropped rather than padded.
SEGMENT_MIN_TAIL_S = 3


def segment_windows(
    sample_count: int,
    sr: int = CLAP_SR,
    *,
    window_s: int = SEGMENT_WINDOW_S,
    hop_s: int = SEGMENT_HOP_S,
    min_tail_s: int = SEGMENT_MIN_TAIL_S,
) -> list[tuple[int, int]]:
    """[start, end) sample ranges tiling a track with fixed-length windows.

    Windows advance by the hop while a full window fits. A remaining tail of at
    least `min_tail_s` gets one final window aligned to the end of the track, so
    every window is full-length (no padding) and the outro is never lost.
    Audio shorter than one window is a single window.
    """
    window = window_s * sr
    hop = hop_s * sr
    if sample_count <= window:
        return [(0, sample_count)]
    windows: list[tuple[int, int]] = []
    start = 0
    while start + window <= sample_count:
        windows.append((start, start + window))
        start += hop
    if sample_count - windows[-1][1] >= min_tail_s * sr:
        windows.append((sample_count - window, sample_count))
    return windows

# Music-tuned LAION checkpoint; matches the Xenova ONNX build we'd use if the
# worker ever moves to TypeScript, so embeddings stay comparable across both.
DEFAULT_MODEL = os.environ.get(
    "MINSTREL_CLAP_MODEL", "laion/larger_clap_music_and_speech"
)


def _l2(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def pair_probability(audio_embed: np.ndarray, text_pair: np.ndarray) -> float:
    """Probability that audio matches the first of two text embeddings."""
    sims = text_pair @ audio_embed
    e = np.exp(sims - sims.max())
    return float((e / e.sum())[0])


class Clap:
    """Produces L2-normalised embeddings in CLAP's shared audio/text space.

    The audio and text projection heads map into one contrastive space, so a
    text query embedding is directly comparable to a track's audio embedding.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.device = select_device()
        self.audio_model = (
            ClapAudioModelWithProjection.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.text_model = (
            ClapTextModelWithProjection.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        self.processor = ClapProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def embed_audio(self, audio: np.ndarray) -> np.ndarray:
        """L2-normalised 512-dim embedding for one mono 48 kHz waveform."""
        return self.embed_audio_batch([audio])[0]

    @torch.no_grad()
    def embed_audio_batch(self, audio: list[np.ndarray]) -> np.ndarray:
        """L2-normalised [n, 512] embeddings for mono 48 kHz waveforms."""
        if not audio:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        inputs = self.processor(audio=audio, sampling_rate=CLAP_SR, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _l2(self.audio_model(**inputs).audio_embeds)
        return feats.cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """L2-normalised [n, 512] embeddings for text prompts."""
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        feats = _l2(self.text_model(**inputs).text_embeds)
        return feats.cpu().numpy().astype(np.float32)

    def zero_shot_pair(
        self, audio_embed: np.ndarray, positive: str, negative: str
    ) -> float:
        """Calibrated 0..1 score via a contrastive prompt pair.

        Softmax over (sim_to_positive, sim_to_negative) gives a probability that
        the track matches `positive` — e.g. "aggressive heavy music" vs
        "calm gentle music" yields an aggressiveness score.
        """
        txt = self.embed_text([positive, negative])  # [2, 512], normalised
        return pair_probability(audio_embed, txt)
