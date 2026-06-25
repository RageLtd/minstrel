from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Iterator
from typing import Protocol

from . import features, store
from .clap import Clap
from .navidrome import Navidrome, Song

# Contrastive prompt pairs → calibrated 0..1 mood knobs the orchestrator filters on.
ZS_PROMPTS: dict[str, tuple[str, str]] = {
    "zs_aggressive": ("aggressive heavy intense music", "calm gentle soft music"),
    "zs_danceable": ("danceable rhythmic groove", "static ambient arrhythmic music"),
    "zs_acoustic": ("acoustic organic instruments", "electronic synthetic production"),
}


class SongSource(Protocol):
    def iter_songs(self) -> Iterator[Song]: ...
    def download(self, song_id: str) -> bytes: ...


def analyze_song(
    clap: Clap,
    conn: sqlite3.Connection,
    source: SongSource,
    song: Song,
    *,
    force: bool = False,
) -> int | None:
    """Analyze one Navidrome song into the store. Returns the track id, or None
    if its size is unchanged since last run (incremental skip — no download)."""
    if not force and store.existing_size(conn, song["id"]) == song["size"]:
        return None

    audio = features.load_audio_bytes(source.download(song["id"]))
    embedding = clap.embed_audio(audio)
    scalars = features.extract(audio)
    zero_shot = {
        col: clap.zero_shot_pair(embedding, pos, neg)
        for col, (pos, neg) in ZS_PROMPTS.items()
    }

    track_id = store.upsert_track(
        conn,
        navidrome_id=song["id"],
        mbid=song["mbid"],
        title=song["title"],
        artist=song["artist"],
        album=song["album"],
        nav_size=song["size"],
    )
    store.upsert_features(conn, track_id, scalars, zero_shot)
    store.set_embedding(conn, track_id, embedding)
    conn.commit()
    return track_id


def sync(clap: Clap, conn: sqlite3.Connection, source: SongSource) -> int:
    analyzed = 0
    for song in source.iter_songs():
        if analyze_song(clap, conn, source, song) is not None:
            analyzed += 1
    return analyzed


def main() -> None:
    parser = argparse.ArgumentParser(description="Minstrel analyzer (Navidrome-sourced)")
    parser.add_argument("--db", default=os.environ.get("MINSTREL_DB", "minstrel.db"))
    parser.add_argument("--navidrome-url", default=os.environ.get("NAVIDROME_URL"))
    parser.add_argument("--user", default=os.environ.get("NAVIDROME_USER"))
    parser.add_argument("--password", default=os.environ.get("NAVIDROME_PASS"))
    args = parser.parse_args()
    if not (args.navidrome_url and args.user and args.password):
        parser.error("Navidrome URL/user/password required (flags or NAVIDROME_* env)")

    clap = Clap()
    conn = store.open_db(args.db)
    nav = Navidrome(args.navidrome_url, args.user, args.password)
    count = sync(clap, conn, nav)
    print(f"analyzed {count} new/changed tracks into {args.db}")


if __name__ == "__main__":
    main()
