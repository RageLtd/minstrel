from __future__ import annotations

from typing import TypedDict

from mutagen import File as MutagenFile


class TrackTags(TypedDict):
    mbid: str | None
    title: str | None
    artist: str | None
    album: str | None


def _first(tags: object, *keys: str) -> str | None:
    for key in keys:
        value = tags.get(key)  # type: ignore[attr-defined]
        if value:
            return str(value[0]) if isinstance(value, list) else str(value)
    return None


def read_tags(path: str) -> TrackTags:
    """Read MusicBrainz recording id + basics from Picard-written Vorbis tags.

    The recording MBID lives in MUSICBRAINZ_TRACKID (Picard's legacy field name,
    which is what Navidrome reads as mbz_recording_id). Returns None where a tag
    is absent, so untagged files fall back to file-path identity.
    """
    audio = MutagenFile(path)
    if audio is None or audio.tags is None:
        return TrackTags(mbid=None, title=None, artist=None, album=None)
    tags = audio.tags
    return TrackTags(
        mbid=_first(tags, "musicbrainz_trackid", "MUSICBRAINZ_TRACKID"),
        title=_first(tags, "title", "TITLE"),
        artist=_first(tags, "artist", "ARTIST"),
        album=_first(tags, "album", "ALBUM"),
    )
