import numpy as np

from minstrel_analyzer import features
from minstrel_analyzer.clap import CLAP_SR, EMBED_DIM, Clap, segment_windows


def _noise(seconds: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(CLAP_SR * seconds))).astype(np.float32)


def test_clap_audio_embedding_shape(clap: Clap):
    emb = clap.embed_audio(_noise())
    assert emb.shape == (EMBED_DIM,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3  # L2-normalised


def test_clap_audio_batch_embedding_shape(clap: Clap):
    embeddings = clap.embed_audio_batch([_noise(), _noise(2.0)])
    assert embeddings.shape == (2, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-3)


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
    assert set(f) == {
        "bpm",
        "rms_energy",
        "spectral_centroid",
        "rhythmic_irregularity",
        "timbral_complexity",
        "dynamic_contrast",
        "harmonic_instability",
        "percussive_ratio",
        "spectral_flatness",
        "zero_crossing_rate",
    }
    assert all(np.isfinite(value) for value in f.values())
    assert 0 <= f["rhythmic_irregularity"] <= 1
    assert 0 <= f["dynamic_contrast"] <= 1
    assert 0 <= f["harmonic_instability"] <= 1
    assert 0 <= f["percussive_ratio"] <= 1


def test_rhythmic_irregularity_distinguishes_random_from_regular_pulses():
    sr = CLAP_SR
    duration = 12
    sample_count = sr * duration
    pulse = np.hanning(sr // 50).astype(np.float32)
    regular = np.zeros(sample_count, dtype=np.float32)
    irregular = np.zeros(sample_count, dtype=np.float32)
    for start in range(0, sample_count - len(pulse), sr // 2):
        regular[start : start + len(pulse)] = pulse
    rng = np.random.default_rng(1)
    starts = np.cumsum(rng.integers(sr // 8, sr, size=30))
    for start in starts[starts < sample_count - len(pulse)]:
        irregular[start : start + len(pulse)] = pulse

    regular_score = features.extract(regular, sr)["rhythmic_irregularity"]
    irregular_score = features.extract(irregular, sr)["rhythmic_irregularity"]

    assert regular_score < 0.3
    assert irregular_score > 0.6


def test_librosa_features_handle_very_short_audio():
    short = np.random.default_rng(2).normal(0, 0.1, 2048).astype(np.float32)

    assert all(np.isfinite(value) for value in features.extract(short).values())


def test_segment_windows_tile_long_audio_with_a_tail_aligned_final_window():
    sr = CLAP_SR
    assert segment_windows(23 * sr, sr) == [
        (0, 10 * sr),
        (5 * sr, 15 * sr),
        (10 * sr, 20 * sr),
        (13 * sr, 23 * sr),
    ]


def test_segment_windows_drop_a_tail_shorter_than_the_minimum():
    sr = CLAP_SR
    assert segment_windows(12 * sr, sr) == [(0, 10 * sr)]


def test_segment_windows_keep_short_audio_as_one_window():
    sr = CLAP_SR
    assert segment_windows(4 * sr, sr) == [(0, 4 * sr)]
    assert segment_windows(10 * sr, sr) == [(0, 10 * sr)]
