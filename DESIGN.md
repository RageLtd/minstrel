# Minstrel — Design

A self-hosted "playlist by vibe" service. Ask in plain language — *"early Mastodon
but higher energy"* — and get a real playlist written back into Navidrome, drawn from
your own library, matched on how the music actually sounds.

## What it is (and isn't)

The LLM does **not** know what your tracks sound like and never touches audio. Its only
job is to **translate a vibe into a structured query**. Deterministic code does the
actual matching against precomputed audio features and embeddings.

Matching engine: **CLAP + librosa**. Essentia was dropped — no Linux-aarch64 wheel, and
CLAP's zero-shot scoring covers most of what its classifiers offered.

**Navidrome is the hub of record.** The analyzer reads from it and the orchestrator
writes to it; both key on Navidrome's stable Persistent ID (PID). No MusicBrainz join,
no reverse-mapping, no direct filesystem access from the analyzer.

## Hardware target

NVIDIA DGX Spark (GB10 Grace-Blackwell, ARM, 128 GB unified). Everything is `arm64` +
CUDA ≥ 12.9. Dev/test happens on an ARM Mac (CLAP on MPS, CPU floor). The only swap to
production is the analyzer's base image (Mac PyTorch → NGC arm64 CUDA).

## Container topology

| Service | Lang/Base | Role | GPU |
|---|---|---|---|
| `navidrome` | official image | Subsonic server; owns the catalog + serves audio | no |
| `analyzer` | Python on NGC arm64 PyTorch | enumerates Navidrome, downloads originals, embeds | **yes** |
| `embed-server` | same image as analyzer | CLAP text-embedding HTTP service for the orchestrator | yes |
| `orchestrator` | Bun | webui + chat, LLM tool-loop, search, writes playlists to Navidrome | no |
| `ollama` | official image | hosts Qwen3.5-35B-A3B for query translation | yes |
| SQLite file | — | shared volume; analyzer writes, orchestrator reads (WAL mode) | — |

`analyzer` and `embed-server` share the CLAP code; in deployment they can be one process
with two entrypoints. All containers run on one Docker host, so Navidrome audio fetches
are effectively local IO.

## Identity

- **Primary key: Navidrome song ID (PID).** Stable across rescans and file moves by
  design (default PIDs derive from tags/MBID, not file path). It's exactly what
  `createPlaylist` consumes, so the orchestrator stores what it will later hand back.
- **Secondary anchor: `mbid`** (read from Navidrome's OpenSubsonic `musicBrainzId`).
  Stored but not used for joins — cheap insurance for re-linking embeddings if PIDs ever
  rotate (PID-config change or a Navidrome rebuild) instead of re-analyzing the library.

## Data model (SQLite + sqlite-vec)

```sql
CREATE TABLE tracks (
  id           INTEGER PRIMARY KEY,
  navidrome_id TEXT NOT NULL UNIQUE,  -- the join key everywhere
  mbid         TEXT,                  -- secondary re-link anchor
  title TEXT, artist TEXT, album TEXT,
  nav_size     INTEGER,               -- Navidrome-reported size; incremental change signal
  analyzed_at  INTEGER
);
CREATE TABLE track_features (track_id PK, bpm, rms_energy, spectral_centroid,
  zs_aggressive, zs_danceable, zs_acoustic, extra_json);
CREATE VIRTUAL TABLE track_vec USING vec0(track_id PK, embedding FLOAT[512] cosine);
```

## Analyzer pipeline (Navidrome-sourced, in-memory)

1. Enumerate songs via the Subsonic API (`getAlbumList2` paged → `getAlbum`), each
   carrying id, title, artist, album, size, and `musicBrainzId`.
2. Skip any song whose `size` matches the stored `nav_size` → incremental.
3. **Download the original** via the `download` endpoint (never `stream` — that may
   transcode and would poison the embeddings), decode in memory, then discard the bytes.
4. CLAP audio embedding (GPU) + CLAP zero-shot mood scores + librosa scalars.
5. Upsert tracks + features + embedding keyed by `navidrome_id`.

## Query path (orchestrator)

```
chat msg → Ollama (search_tracks tool) → parse/validate → executeSearch:
  semantic_text → CLAP text-encode (embed-server) → KNN over track_vec
  seed_artists  → centroid of their stored embeddings → KNN
  apply scalar filters; missing seed artists fall back to semantic + are reported
→ second Ollama turn narrates the result
→ createPlaylist(name, [navidrome_id...]) straight to Navidrome
```

Query text is embedded by the **same** CLAP model as the audio (the embed-server), so
text and audio share one space.

## Status

- **Built & tested:** the whole stack — shared schema; orchestrator data spine
  (sqlite-vec KNN + filtered search); Navidrome-sourced CLAP analyzer (embeddings,
  zero-shot, librosa) plus the embed-server; cross-language storage handshake; LLM
  translation, search execution, and the conversational loop; the Navidrome
  `createPlaylist` writer; the `Bun.serve` entrypoint + SolidJS chat UI; and the
  Docker Compose for the Spark (Ollama is external).
- **Verified on the dev Mac:** all tests green (TS + Python); the orchestrator image
  builds, boots, and serves the bundled UI in production.
- **Remaining — needs the Spark:**
  1. `docker compose build` — confirm the NGC image tag supports GB10/aarch64
     (`nvcr.io` login may be required) and the deps install atop the image's torch.
  2. Copy `.env.example` → `.env`; fill credentials, `MUSIC_PATH`, `OLLAMA_URL`, model.
  3. `docker compose up`, then `docker compose run --rm analyzer` to ingest the library.
  4. Live end-to-end smoke: a typed vibe → a real playlist in Navidrome.
