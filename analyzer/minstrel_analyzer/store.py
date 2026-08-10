from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from .clap import EMBED_DIM
from .features import BASE_SCALARS, FEATURE_VERSION

# Same DDL the TypeScript orchestrator applies — kept in lockstep via one file.
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema.sql"


def open_db(path: str) -> sqlite3.Connection:
    """Open the shared store with sqlite-vec loaded and the schema applied.

    uv-managed Python is built with loadable-extension support, so this works
    where the macOS system Python's sqlite3 would not.
    """
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def existing_size(conn: sqlite3.Connection, navidrome_id: str) -> int | None:
    """Last-analyzed Navidrome size for a song, or None if unseen.

    A changed size means the file was replaced/re-encoded → re-analyze.
    """
    row = conn.execute(
        "SELECT nav_size FROM tracks WHERE navidrome_id = ?", (navidrome_id,)
    ).fetchone()
    return row[0] if row else None


def needs_analysis(
    conn: sqlite3.Connection, navidrome_id: str, nav_size: int | None
) -> bool:
    """Whether audio changed or its stored Librosa feature set is stale."""
    row = conn.execute(
        """
        SELECT t.nav_size,
               CASE WHEN json_valid(f.extra_json)
                    THEN json_extract(f.extra_json, '$.feature_version') END
          FROM tracks t
          LEFT JOIN track_features f ON f.track_id = t.id
         WHERE t.navidrome_id = ?
        """,
        (navidrome_id,),
    ).fetchone()
    return row is None or row[0] != nav_size or row[1] != FEATURE_VERSION


def upsert_track(
    conn: sqlite3.Connection,
    *,
    navidrome_id: str,
    mbid: str | None = None,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
    nav_size: int | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO tracks
          (navidrome_id, mbid, title, artist, album, nav_size, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, unixepoch())
        ON CONFLICT(navidrome_id) DO UPDATE SET
          mbid=excluded.mbid, title=excluded.title, artist=excluded.artist,
          album=excluded.album, nav_size=excluded.nav_size, analyzed_at=unixepoch()
        RETURNING id
        """,
        (navidrome_id, mbid, title, artist, album, nav_size),
    ).fetchone()
    return int(row[0])


def upsert_features(
    conn: sqlite3.Connection,
    track_id: int,
    scalars: dict[str, float],
    zero_shot: dict[str, float],
) -> None:
    extra = {
        "feature_version": FEATURE_VERSION,
        **{key: value for key, value in scalars.items() if key not in BASE_SCALARS},
    }
    conn.execute(
        """
        INSERT INTO track_features
          (track_id, bpm, rms_energy, spectral_centroid,
           zs_aggressive, zs_danceable, zs_acoustic, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(track_id) DO UPDATE SET
          bpm=excluded.bpm, rms_energy=excluded.rms_energy,
          spectral_centroid=excluded.spectral_centroid,
          zs_aggressive=excluded.zs_aggressive,
          zs_danceable=excluded.zs_danceable,
          zs_acoustic=excluded.zs_acoustic,
          extra_json=excluded.extra_json
        """,
        (
            track_id,
            scalars["bpm"],
            scalars["rms_energy"],
            scalars["spectral_centroid"],
            zero_shot.get("zs_aggressive"),
            zero_shot.get("zs_danceable"),
            zero_shot.get("zs_acoustic"),
            json.dumps(extra, separators=(",", ":"), sort_keys=True),
        ),
    )


NOVELTY_FEATURES = (
    "rhythmic_irregularity",
    "timbral_complexity",
    "dynamic_contrast",
    "harmonic_instability",
    "percussive_ratio",
    "spectral_flatness",
    "zero_crossing_rate",
)
FEATURE_READY_KEY = "audio_feature_version"


def audio_features_ready(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT value FROM analysis_meta WHERE key = ?", (FEATURE_READY_KEY,)
    ).fetchone()
    return row is not None and row[0] == FEATURE_VERSION


def mark_audio_features_stale(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM analysis_meta WHERE key = ?", (FEATURE_READY_KEY,))


def mark_audio_features_ready(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        INSERT INTO analysis_meta (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (FEATURE_READY_KEY, FEATURE_VERSION),
    )


def all_stored_features_current(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """
        SELECT count(*) AS total,
               count(CASE WHEN json_valid(extra_json)
                                AND json_extract(extra_json, '$.feature_version') = ?
                                AND json_extract(extra_json, '$.novelty') IS NOT NULL
                          THEN 1 END) AS current
          FROM track_features
        """,
        (FEATURE_VERSION,),
    ).fetchone()
    return row[0] > 0 and row[0] == row[1]


def novelty_needs_recompute(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        """
        SELECT 1
          FROM track_features
         WHERE json_valid(extra_json)
           AND json_extract(extra_json, '$.feature_version') = ?
           AND json_extract(extra_json, '$.novelty') IS NULL
         LIMIT 1
        """,
        (FEATURE_VERSION,),
    ).fetchone()
    return row is not None


def recompute_novelty(conn: sqlite3.Connection) -> int:
    """Recalibrate corpus-relative novelty for every current feature vector."""
    rows = conn.execute(
        "SELECT track_id, extra_json FROM track_features WHERE json_valid(extra_json)"
    ).fetchall()
    parsed: list[tuple[int, dict[str, float]]] = []
    vectors: list[list[float]] = []
    for track_id, raw in rows:
        extra = json.loads(raw)
        if extra.get("feature_version") != FEATURE_VERSION:
            continue
        try:
            vector = [float(extra[key]) for key in NOVELTY_FEATURES]
        except (KeyError, TypeError, ValueError):
            continue
        if not np.all(np.isfinite(vector)):
            continue
        parsed.append((int(track_id), extra))
        vectors.append(vector)

    if not vectors:
        return 0

    matrix = np.asarray(vectors, dtype=np.float64)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    std = np.std(matrix, axis=0)
    scale = np.where(mad > 1e-12, 1.4826 * mad, np.where(std > 1e-12, std, 1.0))
    robust_z = np.clip((matrix - median) / scale, -5.0, 5.0)
    distances = np.sqrt(np.mean(np.square(robust_z), axis=1))

    if len(distances) == 1 or float(np.ptp(distances)) <= 1e-12:
        percentiles = np.zeros_like(distances)
    else:
        _, inverse, counts = np.unique(
            distances, return_inverse=True, return_counts=True
        )
        starts = np.cumsum(counts) - counts
        average_ranks = starts + (counts - 1) / 2
        percentiles = average_ranks[inverse] / (len(distances) - 1)

    for (track_id, extra), novelty in zip(parsed, percentiles, strict=True):
        extra["novelty"] = float(novelty)
        conn.execute(
            "UPDATE track_features SET extra_json = ? WHERE track_id = ?",
            (json.dumps(extra, separators=(",", ":"), sort_keys=True), track_id),
        )
    return len(parsed)


def set_embedding(
    conn: sqlite3.Connection, track_id: int, embedding: np.ndarray
) -> None:
    """vec0 has no UPSERT, so replace by primary key. Stored as little-endian
    float32 — byte-identical to the orchestrator's Float32Array."""
    if embedding.shape[-1] != EMBED_DIM:
        raise ValueError(f"embedding must be {EMBED_DIM} dims, got {embedding.shape}")
    blob = np.asarray(embedding, dtype="<f4").tobytes()
    conn.execute("DELETE FROM track_vec WHERE track_id = ?", (track_id,))
    conn.execute(
        "INSERT INTO track_vec(track_id, embedding) VALUES (?, ?)", (track_id, blob)
    )
