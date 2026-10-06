import io
import json

import numpy as np
import soundfile as sf

from minstrel_analyzer import analyze, features, store
from minstrel_analyzer.navidrome import Song


def _tone_bytes(sr: int = 48000, secs: float = 3.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    buf = io.BytesIO()
    sf.write(
        buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="FLAC"
    )
    return buf.getvalue()


def _song(song_id: str, size: int, **kw) -> Song:
    return Song(
        id=song_id,
        title=kw.get("title", "t"),
        artist=kw.get("artist", "a"),
        album=kw.get("album", "al"),
        size=size,
        mbid=kw.get("mbid"),
    )


class FakeNavidrome:
    def __init__(self, songs: list[Song], audio: bytes):
        self._songs = songs
        self._audio = audio
        self.downloads = 0

    def iter_songs(self):
        return iter(self._songs)

    def download(self, song_id: str) -> bytes:
        self.downloads += 1
        return self._audio


class FakeClap:
    def __init__(self):
        self.audio_batch_sizes: list[int] = []
        self.text_calls = 0

    def embed_audio_batch(self, audio: list[np.ndarray]) -> np.ndarray:
        self.audio_batch_sizes.append(len(audio))
        embeddings = np.zeros((len(audio), 512), dtype=np.float32)
        for index in range(len(audio)):
            embeddings[index, index % 512] = 1.0  # one-hot per segment
        return embeddings

    def embed_text(self, texts: list[str]) -> np.ndarray:
        self.text_calls += 1
        embeddings = np.zeros((len(texts), 512), dtype=np.float32)
        embeddings[:, 0] = 1.0
        return embeddings


def test_sync_writes_tracks_keyed_by_navidrome_id(tmp_path):
    nav = FakeNavidrome(
        [_song("nav-1", 1000, artist="Mastodon", mbid="mb-1")], _tone_bytes()
    )
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1

    row = conn.execute("SELECT navidrome_id, mbid, artist FROM tracks").fetchone()
    assert row == ("nav-1", "mb-1", "Mastodon")
    assert conn.execute("SELECT count(*) FROM track_vec").fetchone()[0] == 1
    rms = conn.execute("SELECT rms_energy FROM track_features").fetchone()[0]
    assert rms > 0
    extra = json.loads(
        conn.execute("SELECT extra_json FROM track_features").fetchone()[0]
    )
    assert extra["feature_version"] == features.FEATURE_VERSION
    assert extra["novelty"] == 0
    assert extra["rhythmic_irregularity"] >= 0
    assert store.audio_features_ready(conn)


def test_unchanged_size_is_skipped_without_download(tmp_path):
    nav = FakeNavidrome([_song("nav-1", 1000)], _tone_bytes())
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1
    assert analyze.sync(clap, conn, nav) == 0  # same size → skipped
    assert nav.downloads == 1  # second pass never downloaded
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
    assert store.audio_features_ready(conn)


def test_stale_feature_version_is_reanalyzed_without_an_audio_size_change(tmp_path):
    nav = FakeNavidrome([_song("nav-1", 1000)], _tone_bytes())
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1
    conn.execute(
        "UPDATE track_features SET extra_json = ?",
        (json.dumps({"feature_version": features.FEATURE_VERSION - 1}),),
    )
    conn.commit()

    assert analyze.sync(clap, conn, nav) == 1
    assert nav.downloads == 2
    version = conn.execute(
        "SELECT json_extract(extra_json, '$.feature_version') FROM track_features"
    ).fetchone()[0]
    assert version == features.FEATURE_VERSION
    assert store.audio_features_ready(conn)


def test_novelty_is_a_corpus_relative_robust_distance(tmp_path):
    conn = store.open_db(str(tmp_path / "m.db"))
    for index, value in enumerate((0.0, 1.0, 10.0)):
        track_id = store.upsert_track(conn, navidrome_id=f"nav-{index}")
        scalars = {
            "bpm": 120.0,
            "rms_energy": 0.2,
            "spectral_centroid": 2000.0,
            **{key: value for key in store.NOVELTY_FEATURES},
        }
        store.upsert_features(conn, track_id, scalars, {})

    assert store.recompute_novelty(conn) == 3
    novelty = [
        row[0]
        for row in conn.execute(
            """
            SELECT json_extract(extra_json, '$.novelty')
              FROM track_features
             ORDER BY track_id
            """
        )
    ]
    assert novelty == [0.5, 0.0, 1.0]


def test_sync_batches_audio_and_embeds_fixed_prompts_once(tmp_path):
    nav = FakeNavidrome(
        [_song(f"nav-{index}", 1000 + index) for index in range(3)], _tone_bytes()
    )
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav, workers=2, batch_size=2) == 3

    assert clap.audio_batch_sizes == [2, 1]
    # Once for the zero-shot prompt pairs, once for the tag vocabulary — never
    # per song.
    assert clap.text_calls == 2
    assert nav.downloads == 3


def test_sync_stores_one_embedding_per_segment_and_a_track_centroid(tmp_path):
    nav = FakeNavidrome([_song("nav-1", 1000)], _tone_bytes(secs=23.0))
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1

    segments = conn.execute(
        "SELECT start_s, end_s, zs_aggressive FROM track_segments ORDER BY start_s"
    ).fetchall()
    assert [(start, end) for start, end, _ in segments] == [
        (0.0, 10.0),
        (5.0, 15.0),
        (10.0, 20.0),
        (13.0, 23.0),
    ]
    assert all(0.0 <= score <= 1.0 for _, _, score in segments)
    assert conn.execute("SELECT count(*) FROM segment_vec").fetchone()[0] == 4
    assert clap.audio_batch_sizes == [4]

    # track_vec is the renormalised mean of the segment embeddings. FakeClap
    # emits one-hot rows, so the centroid is 0.5 on each of the first four dims.
    blob = conn.execute("SELECT embedding FROM track_vec").fetchone()[0]
    centroid = np.frombuffer(blob, dtype="<f4")
    np.testing.assert_allclose(centroid[:4], 0.5, atol=1e-6)
    assert np.all(centroid[4:] == 0)

    extra = json.loads(
        conn.execute("SELECT extra_json FROM track_features").fetchone()[0]
    )
    assert extra["segment_count"] == 4
    assert extra["zs_aggressive_min"] <= extra["zs_aggressive_max"]
    track_score = conn.execute(
        "SELECT zs_aggressive FROM track_features"
    ).fetchone()[0]
    assert abs(track_score - np.mean([score for _, _, score in segments])) < 1e-9


def test_reanalysis_replaces_segments_instead_of_appending(tmp_path):
    nav = FakeNavidrome([_song("nav-1", 1000)], _tone_bytes(secs=23.0))
    clap = FakeClap()
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1
    changed = FakeNavidrome([_song("nav-1", 2000)], _tone_bytes(secs=23.0))
    assert analyze.sync(clap, conn, changed) == 1

    assert conn.execute("SELECT count(*) FROM track_segments").fetchone()[0] == 4
    assert conn.execute("SELECT count(*) FROM segment_vec").fetchone()[0] == 4
    assert conn.execute("SELECT count(*) FROM track_vec").fetchone()[0] == 1
