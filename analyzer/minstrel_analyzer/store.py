from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import sqlite_vec

from .clap import EMBED_DIM

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


def existing_hash(conn: sqlite3.Connection, file_path: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM tracks WHERE file_path = ?", (file_path,)
    ).fetchone()
    return row[0] if row else None


def upsert_track(
    conn: sqlite3.Connection,
    *,
    mbid: str | None,
    file_path: str,
    content_hash: str,
    file_mtime: int,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
) -> int:
    row = conn.execute(
        """
        INSERT INTO tracks
          (mbid, file_path, content_hash, file_mtime, title, artist, album, analyzed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, unixepoch())
        ON CONFLICT(file_path) DO UPDATE SET
          mbid=excluded.mbid, content_hash=excluded.content_hash,
          file_mtime=excluded.file_mtime, title=excluded.title,
          artist=excluded.artist, album=excluded.album, analyzed_at=unixepoch()
        RETURNING id
        """,
        (mbid, file_path, content_hash, file_mtime, title, artist, album),
    ).fetchone()
    return int(row[0])


def upsert_features(
    conn: sqlite3.Connection,
    track_id: int,
    scalars: dict[str, float],
    zero_shot: dict[str, float],
) -> None:
    conn.execute(
        """
        INSERT INTO track_features
          (track_id, bpm, rms_energy, spectral_centroid,
           zs_aggressive, zs_danceable, zs_acoustic, extra_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
        ON CONFLICT(track_id) DO UPDATE SET
          bpm=excluded.bpm, rms_energy=excluded.rms_energy,
          spectral_centroid=excluded.spectral_centroid,
          zs_aggressive=excluded.zs_aggressive,
          zs_danceable=excluded.zs_danceable,
          zs_acoustic=excluded.zs_acoustic
        """,
        (
            track_id,
            scalars["bpm"],
            scalars["rms_energy"],
            scalars["spectral_centroid"],
            zero_shot.get("zs_aggressive"),
            zero_shot.get("zs_danceable"),
            zero_shot.get("zs_acoustic"),
        ),
    )


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
