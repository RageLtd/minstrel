import numpy as np

from minstrel_analyzer import features
from minstrel_analyzer.clap import Clap, CLAP_SR, EMBED_DIM


def _noise(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(CLAP_SR * seconds))).astype(np.float32)


def test_clap_audio_embedding_shape(clap: Clap):
    emb = clap.embed_audio(_noise())
    assert emb.shape == (EMBED_DIM,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3  # L2-normalised


def test_text_and_audio_share_space(clap: Clap):
    audio = clap.embed_audio(_noise())
    text = clap.embed_text(["heavy metal", "ambient drone"])
    assert audio.shape[0] == text.shape[1] == EMBED_DIM


def test_zero_shot_pair_is_a_probability(clap: Clap):
    emb = clap.embed_audio(_noise())
    score = clap.zero_shot_pair(emb, "aggressive heavy music", "calm gentle music")
    assert 0.0 <= score <= 1.0


def test_librosa_scalars_on_sine():
    sr = CLAP_SR
    t = np.linspace(0, 2, sr * 2, endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    f = features.extract(y, sr)
    assert f["rms_energy"] > 0
    assert f["spectral_centroid"] > 0
