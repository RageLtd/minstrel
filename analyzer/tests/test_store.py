import numpy as np
import soundfile as sf

from minstrel_analyzer import analyze, store
from minstrel_analyzer.clap import Clap


def _write_tone(path: str, sr: int = 48000, secs: float = 3.0, freq: float = 220.0):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="FLAC")


def test_analyze_writes_to_shared_db(tmp_path, clap: Clap):
    flac = tmp_path / "tone.flac"
    _write_tone(str(flac))
    conn = store.open_db(str(tmp_path / "minstrel.db"))

    track_id = analyze.analyze_file(clap, conn, str(flac))
    assert isinstance(track_id, int)

    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM track_vec").fetchone()[0] == 1

    rms, agg = conn.execute(
        "SELECT rms_energy, zs_aggressive FROM track_features WHERE track_id=?",
        (track_id,),
    ).fetchone()
    assert rms > 0
    assert 0.0 <= agg <= 1.0

    # Round-trip the stored embedding through KNN: nearest to itself is itself.
    blob = conn.execute(
        "SELECT embedding FROM track_vec WHERE track_id=?", (track_id,)
    ).fetchone()[0]
    nearest = conn.execute(
        "SELECT track_id FROM track_vec WHERE embedding MATCH ? AND k=1 ORDER BY distance",
        (blob,),
    ).fetchone()[0]
    assert nearest == track_id


def test_unchanged_file_is_skipped(tmp_path, clap: Clap):
    flac = tmp_path / "tone.flac"
    _write_tone(str(flac))
    conn = store.open_db(str(tmp_path / "minstrel.db"))

    first = analyze.analyze_file(clap, conn, str(flac))
    second = analyze.analyze_file(clap, conn, str(flac))

    assert isinstance(first, int)
    assert second is None  # content hash unchanged → skipped
    assert conn.execute("SELECT count(*) FROM tracks").fetchone()[0] == 1
