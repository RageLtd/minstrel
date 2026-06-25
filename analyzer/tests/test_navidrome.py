from minstrel_analyzer.navidrome import Navidrome, _to_song


def test_to_song_maps_musicbrainz_id_and_stringifies_id():
    song = _to_song(
        {
            "id": 5,
            "title": "T",
            "artist": "A",
            "album": "Al",
            "size": 123,
            "musicBrainzId": "mb-9",
        }
    )
    assert song["id"] == "5"
    assert song["mbid"] == "mb-9"
    assert song["size"] == 123


def test_iter_songs_pages_albums_then_stops(monkeypatch):
    nav = Navidrome("http://nav", "u", "p")
    pages = iter(
        [
            {"albumList2": {"album": [{"id": "al1"}]}},  # getAlbumList2 offset 0
            {"album": {"song": [{"id": "s1", "title": "x", "size": 1}]}},  # getAlbum al1
            {"albumList2": {"album": []}},  # getAlbumList2 offset 1 → empty, stop
        ]
    )
    monkeypatch.setattr(nav, "_get_json", lambda endpoint, **params: next(pages))

    songs = list(nav.iter_songs())
    assert [s["id"] for s in songs] == ["s1"]


def test_auth_params_have_token_and_salt():
    nav = Navidrome("http://nav", "alice", "secret")
    params = nav._auth_params()
    assert params["u"] == "alice"
    assert len(params["s"]) > 0
    assert len(params["t"]) == 32  # md5 hex digest
