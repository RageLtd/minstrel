from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Iterator
from typing import TypedDict
from urllib.parse import urlencode
from urllib.request import urlopen

API_VERSION = "1.16.1"


class Song(TypedDict):
    id: str
    title: str | None
    artist: str | None
    album: str | None
    size: int | None
    mbid: str | None


def _to_song(raw: dict) -> Song:
    return Song(
        id=str(raw["id"]),
        title=raw.get("title"),
        artist=raw.get("artist"),
        album=raw.get("album"),
        size=raw.get("size"),
        mbid=raw.get("musicBrainzId"),  # OpenSubsonic extension
    )


class Navidrome:
    """Minimal Subsonic client: enumerate songs and download originals.

    Audio is fetched via the `download` endpoint, which returns the original
    bytes — never `stream`, which may transcode and corrupt the embeddings.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        client: str = "minstrel",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.client = client

    def _auth_params(self) -> dict[str, str]:
        salt = secrets.token_hex(8)
        token = hashlib.md5(
            (self.password + salt).encode(), usedforsecurity=False
        ).hexdigest()
        return {
            "u": self.username,
            "t": token,
            "s": salt,
            "v": API_VERSION,
            "c": self.client,
            "f": "json",
        }

    def _url(self, endpoint: str, params: dict[str, object]) -> str:
        query = urlencode({**self._auth_params(), **params})
        return f"{self.base_url}/rest/{endpoint}?{query}"

    def _get_json(self, endpoint: str, **params: object) -> dict:
        with urlopen(self._url(endpoint, params)) as resp:
            return json.loads(resp.read())["subsonic-response"]

    def iter_songs(self, page: int = 500) -> Iterator[Song]:
        offset = 0
        while True:
            data = self._get_json(
                "getAlbumList2", type="alphabeticalByName", size=page, offset=offset
            )
            albums = data.get("albumList2", {}).get("album", [])
            if not albums:
                break
            for album in albums:
                detail = self._get_json("getAlbum", id=album["id"])
                for song in detail.get("album", {}).get("song", []):
                    yield _to_song(song)
            offset += len(albums)

    def download(self, song_id: str) -> bytes:
        with urlopen(self._url("download", {"id": song_id})) as resp:
            return resp.read()
