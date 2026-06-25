from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
from pathlib import Path

from . import features, store, tags
from .clap import Clap

# Contrastive prompt pairs → calibrated 0..1 mood knobs the orchestrator filters on.
ZS_PROMPTS: dict[str, tuple[str, str]] = {
    "zs_aggressive": ("aggressive heavy intense music", "calm gentle soft music"),
    "zs_danceable": ("danceable rhythmic groove", "static ambient arrhythmic music"),
    "zs_acoustic": ("acoustic organic instruments", "electronic synthetic production"),
}


def _content_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def analyze_file(
    clap: Clap, conn: sqlite3.Connection, path: str, *, force: bool = False
) -> int | None:
    """Analyze one track into the store. Returns the track id, or None if the
    file is unchanged since last run (incremental skip)."""
    content_hash = _content_hash(path)
    if not force and store.existing_hash(conn, path) == content_hash:
        return None

    audio = features.load_audio(path)
    embedding = clap.embed_audio(audio)
    scalars = features.extract(audio)
    zero_shot = {
        col: clap.zero_shot_pair(embedding, pos, neg)
        for col, (pos, neg) in ZS_PROMPTS.items()
    }
    meta = tags.read_tags(path)

    track_id = store.upsert_track(
        conn,
        mbid=meta["mbid"],
        file_path=path,
        content_hash=content_hash,
        file_mtime=int(os.path.getmtime(path)),
        title=meta["title"],
        artist=meta["artist"],
        album=meta["album"],
    )
    store.upsert_features(conn, track_id, scalars, zero_shot)
    store.set_embedding(conn, track_id, embedding)
    conn.commit()
    return track_id


def analyze_dir(clap: Clap, conn: sqlite3.Connection, root: str) -> int:
    analyzed = 0
    for path in sorted(Path(root).rglob("*.flac")):
        if analyze_file(clap, conn, str(path)) is not None:
            analyzed += 1
    return analyzed


def main() -> None:
    parser = argparse.ArgumentParser(description="Minstrel audio analyzer")
    parser.add_argument("music_dir", help="root folder of FLAC files")
    parser.add_argument(
        "--db", default=os.environ.get("MINSTREL_DB", "minstrel.db")
    )
    args = parser.parse_args()

    clap = Clap()
    conn = store.open_db(args.db)
    count = analyze_dir(clap, conn, args.music_dir)
    print(f"analyzed {count} new/changed tracks into {args.db}")


if __name__ == "__main__":
    main()
