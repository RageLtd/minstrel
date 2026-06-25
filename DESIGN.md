# Minstrel — Design

A self-hosted "playlist by vibe" service. Ask in plain language — *"early Mastodon
but higher energy"* — and get a real playlist written back into Navidrome, drawn from
your own library, matched on how the music actually sounds.

## What it is (and isn't)

The LLM does **not** know what your tracks sound like and never touches audio. Its only
job is to **translate a vibe into a structured query**. Deterministic code does the
actual matching against precomputed audio features and embeddings. Get that boundary
right and the rest is plumbing.

Matching engine: **CLAP + librosa**. Essentia is deliberately dropped — there is no
Linux-aarch64 wheel, so it would mean a from-source build pinned to a CPU TensorFlow,
and CLAP's zero-shot scoring covers most of what Essentia's classifiers gave us anyway.

## Hardware target

NVIDIA DGX Spark (GB10 Grace-Blackwell, 20-core ARM, 128 GB unified LPDDR5x).

- **Everything is `arm64` + CUDA ≥ 12.9** (GB10 is SM121; older CUDA emits no kernels for it).
- GPU-in-container via NVIDIA Container Toolkit — supported, this is NVIDIA's own ARM box.
- The Spark's weakness (memory bandwidth / token rate) is irrelevant here: analysis is a
  latency-insensitive batch, and the query-time LLM call is a single tool-call, not a
  long generation. Its strength (128 GB unified) lets CLAP + a capable LLM coexist.

## Container topology

| Service | Lang/Base | Role | GPU |
|---|---|---|---|
| `navidrome` | official image (arm64) | Subsonic server, streams from NAS over NFS | no |
| `analyzer` | Python on `nvcr.io/nvidia/pytorch` (arm64) | batch worker: CLAP embeds + zero-shot, librosa scalars | **yes** |
| `orchestrator` | Bun | webui + chat, LLM tool-loop, vector/feature search, Navidrome writes | no |
| `ollama` | official image (arm64+CUDA) | hosts Qwen3.5-35B-A3B for query translation | yes |
| SQLite file | — | shared volume; `analyzer` writes, `orchestrator` reads (WAL mode) | — |

Navidrome over NFS: inotify does **not** fire for NAS-side writes, so rely on scheduled
rescans (`ND_SCANSCHEDULE=@every 1h`), not the realtime watcher.

## Data model (SQLite + sqlite-vec)

```sql
-- canonical track row; mbid is the join key, navidrome_id resolved lazily
CREATE TABLE tracks (
  id            INTEGER PRIMARY KEY,
  mbid          TEXT UNIQUE,          -- MusicBrainz recording id from file tags
  file_path     TEXT NOT NULL,        -- fallback identity if mbid missing
  content_hash  TEXT NOT NULL,        -- for incremental re-analysis
  file_mtime    INTEGER NOT NULL,
  navidrome_id  TEXT,                 -- cached Subsonic id, filled on first use
  title         TEXT, artist TEXT, album TEXT,
  analyzed_at   INTEGER
);

-- scalar features + zero-shot scores we actually filter/sort on
CREATE TABLE track_features (
  track_id        INTEGER PRIMARY KEY REFERENCES tracks(id),
  bpm             REAL,   -- librosa
  rms_energy      REAL,   -- librosa: the "higher energy" knob
  spectral_centroid REAL, -- librosa: "brightness"
  zs_aggressive   REAL,   -- CLAP zero-shot 0..1
  zs_danceable    REAL,
  zs_acoustic     REAL,
  extra_json      TEXT    -- room for more zero-shot prompts without migrations
);

-- CLAP audio embeddings for nearest-neighbour "sounds like"
CREATE VIRTUAL TABLE track_vec USING vec0(
  track_id  INTEGER PRIMARY KEY,
  embedding FLOAT[512]
);
```

## Analyzer pipeline (batch, incremental)

Runs on a schedule (cron / `docker compose run`), claims the GPU only while active.

1. Enumerate library files (walk the mount; read tags for `mbid`, title/artist/album).
2. Skip any track whose `content_hash` is unchanged → incremental.
3. For each new/changed track:
   - decode + resample (CLAP wants 48 kHz mono; librosa its own rate)
   - **CLAP audio embedding** (GPU) → 512-dim vector
   - **CLAP zero-shot scores** (GPU): cosine vs a fixed prompt set
     (`"aggressive heavy music"`, `"danceable groove"`, `"acoustic intimate"`, …)
   - **librosa scalars** (CPU): bpm, rms_energy, spectral_centroid
4. Upsert `tracks` + `track_features` + `track_vec` in one transaction.

## Query path (the orchestration loop)

```
chat msg
  → Ollama (Qwen) with the search_tracks tool
  → LLM emits a structured tool call (below)
  → orchestrator executes:
      • semantic_text → CLAP text-encode → KNN over track_vec
      • seed_artists  → mean of their embeddings → KNN
      • apply scalar filters (bpm/energy/zero-shot) via metadata filter / join
      • re-rank, dedupe by artist, take `count`
  → map track_ids → navidrome_id (cached; else Subsonic search3 by mbid/title)
  → Subsonic createPlaylist(name, ids)
  → return the tracklist to the chat for display/confirmation
```

The query text **must** be embedded with CLAP's own text encoder, never Ollama's
text-embedding endpoint — text and audio must share CLAP's one space.

## The LLM tool (the language→query boundary)

The model only ever emits this. It never sees the DB or audio.

```jsonc
{
  "name": "search_tracks",
  "parameters": {
    "semantic_text":  "string  — free-text vibe, CLAP-encoded. optional",
    "seed_artists":   "string[] — artists in the library to anchor 'sounds like'. optional",
    "filters": {
      "bpm_min": "number?", "bpm_max": "number?",
      "energy_min": "number?",        // 0..1, drives 'higher energy'
      "aggressive_min": "number?",    // zero-shot 0..1
      "danceable_min": "number?",
      "acoustic_max": "number?"
    },
    "count": "number — playlist length, default 30"
  }
}
```

"Early Mastodon but higher energy" → `seed_artists:["Mastodon"]` (the *sounds-like*
half) + `energy_min` set above the seed tracks' average (the *delta* half). Whether the
delta is a hard filter or a re-rank bias is an open decision (below).

## Track identity & Navidrome writes

- **Join key: MusicBrainz recording id** (`mbid`), read from file tags — works only if
  the library is Picard-tagged. Audit tag coverage first; fall back to `file_path` (or an
  AcoustID fingerprint) where missing.
- Subsonic `createPlaylist` takes **Navidrome's own song ids** (MD5/UUID), *not* MBIDs.
  So: keep `mbid` in our DB, map `mbid → navidrome_id` at write time (cache it, since both
  sides derive from the same files), then post the playlist.

## Open decisions

1. **"Higher energy" semantics** — hard `energy_min` filter vs re-rank bias toward higher
   `rms_energy`. Bias is probably gentler/better; needs a feel test.
2. **Conversational refinement** — one-shot query, or multi-turn ("more like the back half,
   less screamy")? Affects whether the orchestrator keeps per-session state.
3. **Seed resolution** — when a named artist isn't in the library, fall back to pure
   `semantic_text`, or tell the user?
4. **Playlist write** — curated static list via `createPlaylist` (planned) vs Navidrome
   smart-playlist `.nsp`. Static fits "these N specific tracks" better.

## Build order

0. **De-risk first:** prove CLAP runs on the Blackwell under the NGC arm64 PyTorch image —
   embed + zero-shot ten tracks, confirm SM121 kernels fire. Highest unknown in the stack.
1. Analyzer over a small folder → populate SQLite (embeddings + features) end to end.
2. Orchestrator search path: CLAP text-encode + sqlite-vec KNN + scalar filters, no LLM yet.
3. Wire Ollama + the `search_tracks` tool; close the language→query loop.
4. Navidrome `mbid → id` mapping + `createPlaylist`; close the last mile.
5. Chat webui on top; scheduled incremental analyzer runs.
