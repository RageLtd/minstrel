import io

import numpy as np
import soundfile as sf

from minstrel_analyzer import analyze, store
from minstrel_analyzer.clap import Clap
from minstrel_analyzer.navidrome import Song


def _tone_bytes(sr: int = 48000, secs: float = 3.0, freq: float = 220.0) -> bytes:
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="FLAC")
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


def test_sync_writes_tracks_keyed_by_navidrome_id(tmp_path, clap: Clap):
    nav = FakeNavidrome(
        [_song("nav-1", 1000, artist="Mastodon", mbid="mb-1")], _tone_bytes()
    )
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1

    row = conn.execute("SELECT navidrome_id, mbid, artist FROM tracks").fetchone()
    assert row == ("nav-1", "mb-1", "Mastodon")
    assert conn.execute("SELECT count(*) FROM track_vec").fetchone()[0] == 1
    rms = conn.execute("SELECT rms_energy FROM track_features").fetchone()[0]
    assert rms > 0


def test_unchanged_size_is_skipped_without_download(tmp_path, clap: Clap):
    nav = FakeNavidrome([_song("nav-1", 1000)], _tone_bytes())
    conn = store.open_db(str(tmp_path / "m.db"))

    assert analyze.sync(clap, conn, nav) == 1
    assert analyze.sync(clap, conn, nav) == 0  # same size → skipped
    assert nav.downloads == 1  # second pass never downloaded
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
