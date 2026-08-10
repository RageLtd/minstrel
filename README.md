# Minstrel

A self-hosted "playlist by vibe" service for your own music library. Type
*"early Mastodon but higher energy"* and get a real playlist written back into
Navidrome — drawn from your own files, matched on how the music actually sounds.

The LLM never touches audio and never sees the library. Its only job is to
translate a vibe into a structured query; deterministic code does the matching
against precomputed CLAP embeddings and librosa features. It cannot invent a
track that doesn't exist, because it never names one.

See [DESIGN.md](./DESIGN.md) for the architecture rationale.

## How it works

Two halves share one SQLite file.

**Ingest** (Python, GPU) — the analyzer enumerates your library through
Navidrome's Subsonic API, downloads each original file (never `stream`, which may
transcode and poison the embeddings), decodes it in memory, and stores a
512-dimensional CLAP audio embedding, three zero-shot mood scores, and librosa
descriptors for tempo, dynamics, rhythm, timbre, and harmony. A versioned feature
set makes ordinary re-runs incremental while automatically re-analyzing tracks
when the extractor changes. Feature-aware reranking stays disabled until that
backfill and corpus-novelty calibration complete, so versions are never mixed.

**Query** (Bun, CPU) — the orchestrator serves a chat UI. Your message goes to
Ollama with exactly one tool, `search_tracks`; the model's arguments are
validated into a typed query, the semantic text is embedded by the *same* CLAP
model via the embed-server, and KNN searches over `sqlite-vec` produce an
over-fetched candidate pool. Multiple seed artists retain independent centroids,
so their neighborhoods are fused rather than averaged into a generic midpoint.
Explicit numeric constraints narrow the pool; every subjective quality is a
corpus-percentile preference that reranks it within a bounded CLAP relevance
window. Retrieval expands when constraints or artist diversity underfill, and
artist-variety quotas backfill rather than silently reducing the requested count.
One click writes the immutable result to Navidrome via `createPlaylist`.

```
chat message
  → Ollama (search_tracks tool) → validated SearchQuery
  → semantic_text  → CLAP text embedding (embed-server)
    seed_artists   → one centroid and KNN neighborhood per artist
  → adaptive KNN over track_vec + validated hard constraints
  → fuse seed neighborhoods + bounded percentile-preference reranking
  → artist diversity + cardinality backfill
  → grounded diagnostics → createPlaylist(name, [navidrome_id…]) → Navidrome
```

Navidrome is the hub of record. Everything keys on its stable Persistent ID, so
the ID the search returns is the ID the playlist writer hands back — no mapping
step, no filesystem access from the analyzer.

## Services

| Service | Stack | Role | GPU |
|---|---|---|---|
| `navidrome` | official image | Subsonic server; owns the catalog and serves audio | no |
| `analyzer` | Python on NGC PyTorch | batch ingest: enumerate, download, embed | yes |
| `embed-server` | same image | CLAP text-embedding HTTP service (port 8001, internal) | yes |
| `orchestrator` | Bun | chat UI, LLM tool-loop, search, playlist writer (port 3000) | no |
| `ollama` | **external** | hosts the translation model — not in the Compose file | yes |

Ollama is expected to already exist on your host or LAN; point `OLLAMA_URL` at it.

## Deploy

Target hardware is an NVIDIA DGX Spark (GB10, arm64, CUDA ≥ 12.9), but any
arm64 CUDA host with the NVIDIA Container Toolkit should work.

```bash
cp .env.example .env      # fill in NAVIDROME_USER/PASS, MUSIC_PATH, OLLAMA_URL
docker compose build      # nvcr.io may need `docker login nvcr.io` first
docker compose up -d
```

The analyzer is a batch job, not a daemon — it sits behind the `ingest` profile
and is not started by `up`. Once Navidrome has scanned the library, ingest it:

```bash
docker compose run --rm analyzer
```

First run downloads the CLAP checkpoint into the `hf-cache` volume, which is why
the embed-server's healthcheck allows a 300-second start period. Re-run the
analyzer whenever Navidrome picks up new music. Unchanged tracks are skipped
unless their stored feature version is stale.

Then open the orchestrator at `http://localhost:3000` and describe a vibe.

## Configuration

Everything is environment-driven; Compose reads `.env` automatically.

| Variable | Default | Purpose |
|---|---|---|
| `NAVIDROME_USER` | — | **Required.** Analyzer reads and orchestrator writes as this user |
| `NAVIDROME_PASS` | — | **Required.** |
| `MUSIC_PATH` | `/mnt/music` | Host path bind-mounted read-only into Navidrome |
| `OLLAMA_URL` | `http://host.docker.internal:11434` | Your existing Ollama server |
| `MINSTREL_MODEL` | `gemma4:26b` | Ollama tag for query translation; must already be pulled |
| `MINSTREL_THINK` | `true` | Ollama thinking channel. Keep on: reasoning-first models (GLM 5.x, qwen3.x) return empty narration without it. `false` trades reliability for latency |
| `NAVIDROME_PORT` | `4533` | Published port |
| `ORCHESTRATOR_PORT` | `3000` | Published port |
| `MINSTREL_DB` | `minstrel.db` | SQLite path (`/data/minstrel.db` in containers) |
| `MINSTREL_EMBED_URL` | `http://localhost:8001` | Where the orchestrator finds the embed-server |
| `MINSTREL_CLAP_MODEL` | `laion/larger_clap_music_and_speech` | CLAP checkpoint |
| `MINSTREL_SQLITE_PATH` | Homebrew path | macOS only — see below |

## Local development

The orchestrator runs natively on an ARM Mac; the analyzer runs on MPS with a CPU
floor. The only production swap is the analyzer's base image.

```bash
bun install
bun run dev                                     # http://localhost:3000, HMR on
```

The analyzer and embed-server are a separate `uv` project:

```bash
cd analyzer
uv sync
uv run python -m minstrel_analyzer.embed_server # port 8001
uv run python -m minstrel_analyzer.analyze --db ../minstrel.db \
  --navidrome-url http://localhost:4533 --user admin --password …
```

**macOS caveat:** Apple's bundled SQLite is compiled without loadable-extension
support, so `sqlite-vec` cannot load. `src/db.ts` points Bun at
`/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib` on darwin — install it with
`brew install sqlite`, or set `MINSTREL_SQLITE_PATH` if yours lives elsewhere.
Linux system SQLite loads extensions fine, so this is darwin-only.

## Tests

```bash
bun test                    # orchestrator
cd analyzer && uv run pytest # analyzer
```

## Layout

```
schema.sql                    single source of truth for the DDL, applied by both halves
src/
  server.ts                   Bun.serve entrypoint: routes + dependency wiring
  handlers.ts                 /api/chat and /api/playlist, dependency-injected
  loop.ts                     planner call + grounded result summary
  tools.ts                    search_tracks definition, system prompt, argument validation
  search.ts                   SearchQuery → ranked tracks
  repo.ts                     upserts + the filtered KNN
  db.ts                       SQLite + sqlite-vec setup
  ollama.ts / embedder.ts / navidrome.ts   external clients
  frontend.ts / index.css     SolidJS chat UI (bundled via HTML import)
analyzer/minstrel_analyzer/
  analyze.py                  ingest pipeline + CLI
  clap.py                     CLAP audio/text embeddings + zero-shot scoring
  features.py                 librosa descriptors for rhythm, timbre, dynamics, and harmony
  navidrome.py                Subsonic enumerate + download
  store.py                    SQLite writes
  embed_server.py             text-embedding HTTP service
```
