from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import NamedTuple, Protocol, TypeVar

import numpy as np

from . import features, store, tags
from .clap import CLAP_SR, Clap, pair_probability, segment_windows
from .navidrome import Navidrome, Song

# Contrastive prompt pairs → calibrated 0..1 mood knobs the orchestrator filters on.
ZS_PROMPTS: dict[str, tuple[str, str]] = {
    "zs_aggressive": ("aggressive heavy intense music", "calm gentle soft music"),
    "zs_danceable": ("danceable rhythmic groove", "static ambient arrhythmic music"),
    "zs_acoustic": ("acoustic organic instruments", "electronic synthetic production"),
}

PREPARE_WORKERS = 8
AUDIO_BATCH_SIZE = 8
# Windows per GPU call; a song batch's windows are flattened and chunked by this.
SEGMENT_BATCH_SIZE = 32
T = TypeVar("T")


class SongSource(Protocol):
    def iter_songs(self) -> Iterator[Song]: ...
    def download(self, song_id: str) -> bytes: ...


class PreparedSong(NamedTuple):
    song: Song
    audio: np.ndarray
    scalars: features.Scalars
    windows: list[tuple[int, int]]


def _prepare_song(source: SongSource, song: Song) -> PreparedSong:
    audio = features.load_audio_bytes(source.download(song["id"]))
    return PreparedSong(
        song, audio, features.extract(audio), segment_windows(len(audio))
    )


def _embed_segments(
    clap: Clap, batch: list[PreparedSong], segment_batch_size: int
) -> list[np.ndarray]:
    """Per-song [n_windows, 512] embeddings for a batch of prepared songs."""
    windows = [
        item.audio[start:end] for item in batch for start, end in item.windows
    ]
    chunks = [
        clap.embed_audio_batch(chunk)
        for chunk in _batches(windows, segment_batch_size)
    ]
    flat = (
        np.concatenate(chunks)
        if chunks
        else np.empty((0, store.EMBED_DIM), dtype=np.float32)
    )
    per_song: list[np.ndarray] = []
    offset = 0
    for item in batch:
        per_song.append(flat[offset : offset + len(item.windows)])
        offset += len(item.windows)
    return per_song


def _iter_prepared(
    source: SongSource, songs: Iterable[Song], workers: int
) -> Iterator[PreparedSong]:
    songs_iter = iter(songs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: set[Future[PreparedSong]] = set()

        def submit_next() -> bool:
            try:
                song = next(songs_iter)
            except StopIteration:
                return False
            pending.add(pool.submit(_prepare_song, source, song))
            return True

        for _ in range(workers):
            if not submit_next():
                break

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.remove(future)
                prepared = future.result()
                submit_next()
                yield prepared


def _batches(items: Iterable[T], size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _embed_zero_shot_prompts(clap: Clap) -> dict[str, np.ndarray]:
    columns = list(ZS_PROMPTS)
    prompts = [prompt for column in columns for prompt in ZS_PROMPTS[column]]
    embeddings = clap.embed_text(prompts)
    return {
        column: embeddings[index * 2 : index * 2 + 2]
        for index, column in enumerate(columns)
    }


def _store_analysis(
    conn: sqlite3.Connection,
    prepared: PreparedSong,
    segment_embeddings: np.ndarray,
    prompt_embeddings: dict[str, np.ndarray],
) -> int:
    """Persist one track: per-segment vectors and mood scores, plus a track-level
    centroid and the mean/min/max of each mood score across segments."""
    song = prepared.song
    segments: list[store.Segment] = []
    for (start, end), embedding in zip(
        prepared.windows, segment_embeddings, strict=True
    ):
        zero_shot = {
            column: pair_probability(embedding, text_pair)
            for column, text_pair in prompt_embeddings.items()
        }
        segments.append((start / CLAP_SR, end / CLAP_SR, zero_shot))

    track_zero_shot = {
        column: float(np.mean([zero_shot[column] for _, _, zero_shot in segments]))
        for column in prompt_embeddings
    }
    spread = {
        f"{column}_{bound}": float(reduce(zero_shot[column] for _, _, zero_shot in segments))
        for column in prompt_embeddings
        for bound, reduce in (("min", min), ("max", max))
    }
    centroid = np.mean(segment_embeddings, axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)

    track_id = store.upsert_track(
        conn,
        navidrome_id=song["id"],
        mbid=song["mbid"],
        title=song["title"],
        artist=song["artist"],
        album=song["album"],
        nav_size=song["size"],
    )
    store.upsert_features(
        conn,
        track_id,
        prepared.scalars,
        track_zero_shot,
        {"segment_count": len(segments), **spread},
    )
    store.set_embedding(conn, track_id, centroid.astype(np.float32))
    store.replace_segments(conn, track_id, segments, segment_embeddings)
    conn.commit()
    return track_id


def analyze_song(
    clap: Clap,
    conn: sqlite3.Connection,
    source: SongSource,
    song: Song,
    *,
    force: bool = False,
) -> int | None:
    """Analyze one Navidrome song into the store. Returns the track id, or None
    if its audio and stored feature version are current (no download)."""
    if not force and not store.needs_analysis(conn, song["id"], song["size"]):
        return None

    store.mark_audio_features_stale(conn)
    conn.commit()
    prepared = _prepare_song(source, song)
    embeddings = _embed_segments(clap, [prepared], SEGMENT_BATCH_SIZE)[0]
    track_id = _store_analysis(
        conn, prepared, embeddings, _embed_zero_shot_prompts(clap)
    )
    store.recompute_novelty(conn)
    conn.commit()
    return track_id


def sync(
    clap: Clap,
    conn: sqlite3.Connection,
    source: SongSource,
    *,
    workers: int = PREPARE_WORKERS,
    batch_size: int = AUDIO_BATCH_SIZE,
    segment_batch_size: int = SEGMENT_BATCH_SIZE,
) -> int:
    analyzed = 0
    was_ready = store.audio_features_ready(conn)
    songs = [
        song
        for song in source.iter_songs()
        if store.needs_analysis(conn, song["id"], song["size"])
    ]
    if songs:
        store.mark_audio_features_stale(conn)
        conn.commit()
    prepared = _iter_prepared(source, songs, workers)
    prompt_embeddings: dict[str, np.ndarray] | None = None
    for batch in _batches(prepared, batch_size):
        if prompt_embeddings is None:
            prompt_embeddings = _embed_zero_shot_prompts(clap)
        embeddings = _embed_segments(clap, batch, segment_batch_size)
        for item, item_embeddings in zip(batch, embeddings, strict=True):
            _store_analysis(conn, item, item_embeddings, prompt_embeddings)
            analyzed += 1
    if analyzed > 0 or store.novelty_needs_recompute(conn):
        store.recompute_novelty(conn)
        conn.commit()
    tags.retag(conn, clap)
    if analyzed > 0 or was_ready or store.all_stored_features_current(conn):
        store.mark_audio_features_ready(conn)
        conn.commit()
    return analyzed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minstrel analyzer (Navidrome-sourced)"
    )
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
