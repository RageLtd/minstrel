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
CREATE TABLE analysis_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE VIRTUAL TABLE track_vec USING vec0(track_id PK, embedding FLOAT[512] cosine);
-- one row per 10 s window (5 s hop); the unit of retrieval
CREATE TABLE track_segments (id PK, track_id, start_s, end_s,
  zs_aggressive, zs_danceable, zs_acoustic, tags_json, tag_version);
CREATE VIRTUAL TABLE segment_vec USING vec0(segment_id PK, embedding FLOAT[512] cosine);
```

`track_vec` holds the renormalised mean of a track's segment embeddings and only
serves seed-artist centroids. Matching runs over `segment_vec`. The HF CLAP feature
extractor otherwise takes **one random 10-second crop** of anything longer
(`truncation="rand_trunc"`), which is what originally collapsed every track to a
dice-roll — windowing is what fixes it, not a different pooling.

## Analyzer pipeline (Navidrome-sourced, in-memory)

1. Enumerate songs via the Subsonic API (`getAlbumList2` paged → `getAlbum`), each
   carrying id, title, artist, album, size, and `musicBrainzId`.
2. Skip any song whose `size` and stored feature version are current → incremental,
   with automatic backfill after the feature extractor changes.
3. **Download the original** via the `download` endpoint (never `stream` — that may
   transcode and would poison the embeddings), decode in memory, then discard the bytes.
4. Tile the waveform into 10 s windows at a 5 s hop (a tail ≥ 3 s gets one final
   end-aligned window). Per window: CLAP audio embedding (GPU, batched across
   songs) + CLAP zero-shot mood scores. Per track: the centroid embedding, the
   mean/min/max of each mood score, and librosa descriptors — tempo/energy/
   brightness, rhythmic irregularity, timbral complexity, dynamic contrast,
   harmonic instability, percussive balance, flatness, and zero-crossing rate.
   Robust median/MAD distance across the derived dimensions becomes a
   corpus-relative novelty percentile.
5. Upsert tracks + features + centroid + segments keyed by `navidrome_id`.
6. `tags.retag`: every segment whose `tag_version` is stale is scored against a
   fixed vocabulary (genre / mood / sound / feel) by within-group softmax over CLAP
   similarities, from the stored embedding — a vocabulary change never re-downloads
   audio.

## Query path (orchestrator)

```
chat msg → Ollama (search_tracks tool) → parse/validate → executeSearch:
  semantic_text → CLAP text-encode (embed-server) → KNN over segment_vec
  seed_artists  → independent per-artist centroids → one KNN neighborhood each
  each KNN collapses segments to tracks by best passage (min distance)
  fuse by best seed rank, with semantic rank as a secondary signal
  apply scalar filters; missing seed artists fall back to semantic + are reported
  softly rerank for requested novelty/rhythm/timbre/dynamics/harmony preferences
  membership gate: top 3×count candidates → track cards → Ollama /v1/systemone
    (tev1) → keep probability ≥ threshold, retrieval order preserved
  → apply focused/balanced/wide per-artist cap
→ deterministic grounded result summary (no second LLM call)
→ createPlaylist(name, [navidrome_id...]) straight to Navidrome
```

Query text is embedded by the **same** CLAP model as the audio (the embed-server), so
text and audio share one space.

## Membership decision

CLAP similarity is a candidate generator, not a judge: it cannot read "but no
screamed vocals" or weigh a track's second half against its first. The gate hands
that judgement to a **decision model** served by Ollama's `/v1/systemone`
endpoint — Together AI's `tev1` (4B) by default, with Cloudflare's `clef-flash`
(9B) and `clef` (27B) as drop-in alternatives; the request shape is the same as
TypeSafe's hosted Jev. On live probes tev1 was as decisive as clef-flash and
loads 3.5× faster, which is why it is the default. Decision models return a probability per option from one
forward pass; they never generate text, so there is nothing to parse and nothing
to hallucinate.

The model never hears audio. It judges a **track card**: title/artist/album,
duration, BPM, the mean/min/max of each zero-shot mood score, and a passage
timeline built by merging consecutive segments that share a top genre tag, each
with its aggregated mood/sound/feel tags. A batch of cards goes into one `state`
with one `noul` question per track ("does `track_k` belong on a playlist for
`playlist_request`?"), so ten candidates cost one forward pass.

Rules: the gate judges at most 3×count candidates once per search (probabilities
are memoised across adaptive retrieval passes), keeps those at or above the
threshold (default 0.5), preserves retrieval order rather than reranking by
probability, and reports `classifier` as the shortfall reason when it is the
limiter. A failing decision model is an error surfaced to the user, not a silent
fallback; an empty `MINSTREL_DECISION_MODEL` disables the gate explicitly.

## Search plan V2: intent, relevance, and cardinality

The planner must not turn ordinary descriptive language into arbitrary eligibility
thresholds. "Wake me up," "higher energy," and "more danceable" describe ranking
preferences; "over 140 BPM" and "exclude Primus" are constraints. Treating both as
SQL predicates systematically starves compound requests and makes the returned count
unpredictable.

The V2 query path follows four invariants:

1. The LLM translates language but never invents numerical search policy.
2. Subjective audio qualities affect ordering, not eligibility.
3. Hard constraints carry verifiable evidence from the original request and are never
   silently relaxed.
4. Preference-only searches return the requested count whenever enough analyzed tracks
   exist; any shortfall has a deterministic, surfaced reason.

### Planner contract

The normalized plan separates constraints, preferences, and selection policy:

```ts
interface SearchPlan {
  version: 2;
  semanticText: string;
  seedArtists?: string[];
  excludeSeedArtists?: boolean;
  constraints: Array<{
    feature: "tempoBpm" | "featurePercentile";
    operator: "min" | "max";
    value: number;
    percentileFeature?: PreferenceFeature;
    evidence: string;
  }>;
  preferences: Array<{
    feature: PreferenceFeature;
    direction: "higher" | "lower";
    strength: "subtle" | "normal" | "strong";
    evidence: string;
  }>;
  selection: {
    count: number;
    artistVariety: "focused" | "balanced" | "wide";
    strictArtistCap?: number;
  };
}
```

`PreferenceFeature` covers tempo, energy, aggression, danceability, acousticness,
novelty, rhythmic irregularity, timbral complexity, dynamic contrast, and harmonic
instability. Strength is an enum mapped to fixed deterministic weights; the model does
not emit thresholds for adjectives. BPM and count constraints are accepted only when
their numeric evidence occurs in the user request. User-specified percentile constraints
(for example, "top 20% most aggressive") are allowed; inferred subjective thresholds are
not. Invalid, contradictory, or unsupported conditions are rejected or downgraded to
preferences with a diagnostic warning.

### Corpus calibration

Subjective features are converted to empirical percentile ranks over the current
analyzed corpus:

```text
percentile(value) = average_zero_based_rank(value) / (track_count - 1)
```

For a higher preference, fit is `percentile(value)`; for lower, fit is
`1 - percentile(value)`. This makes controls comparable despite compressed zero-shot
scores, skewed Librosa distributions, and outliers. Raw BPM remains the unit for an
explicit numerical constraint, while faster/slower preferences use BPM percentile.
Novelty is already stored as a corpus percentile. Calibration is built from existing
SQLite values and requires no audio re-analysis or schema migration.

At the current library size calibration is loaded in one SQLite pass per search. A
cross-request cache is allowed only after the analyzer owns a durable corpus-revision
counter; timestamps and row counts are not sufficient invalidation keys.

### Retrieval and bounded reranking

CLAP relevance remains the candidate generator. Semantic text and each seed-artist
centroid are searched independently, then fused into one base-relevance order. The V2
scorer consumes that order through a stable `BaseCandidate` contract so seed-fusion
calibration can evolve separately from preference semantics.

Preferences combine by weighted average rather than intersection. Correlated descriptors
are grouped so several measurements of texture/change do not receive accidental multiple
votes. Their influence is bounded by the relevance gap between the requested boundary
and an outer window:

```text
c = requested track count
preference_budget = strength * (base_score(2c) - base_score(c))
final_score(track) = base_score(track) + preference_budget * (1 - preference_fit(track))
```

This permits meaningful reordering among relevant tracks without allowing an unrelated
global feature outlier to overwhelm CLAP similarity. Missing feature values are omitted
from that track's preference denominator and reported; they are not treated as either a
perfect or worst match.

### Constraints, adaptive retrieval, and diversity

Hard constraints define eligibility. Because sqlite-vec finds nearest neighbors before
joined scalar predicates apply, constrained retrieval expands deterministically: start
with `max(4 * count, 128)` candidates per query vector, double while the final selection
is underfilled, and stop only when enough eligible tracks exist or the searchable corpus
is exhausted. The semantic embedding and seed centroids are computed once and reused.

Artist variety is a first-pass quota, not an accidental result ceiling. Focused,
balanced, and wide modes first select up to six, two, and one tracks per artist
respectively, then backfill skipped candidates in final-score order until the requested
count is reached. Only an explicit `strictArtistCap` forbids that backfill.

The resulting cardinality contract is:

```text
preference-only: returned = min(requested, available analyzed tracks)
strict constraints: returned = min(requested, eligible tracks)
```

Constraints are never silently weakened. When fewer tracks are eligible, the response
states the shortfall and identifies the limiting constraints.

### Diagnostics and UI truthfulness

Search execution returns structured diagnostics alongside hits:

```ts
interface SearchDiagnostics {
  requested: number;
  returned: number;
  searchableTracks: number;
  retrievalPasses: number;
  retrievedCandidates: number;
  eligibleCandidates: number;
  preferenceReranked: boolean;
  diversityBackfilled: number;
  exhaustedCorpus: boolean;
  warnings: string[];
  shortfallReason?: "constraints" | "exclusions" | "corpus" | "strictArtistCap";
}
```

The query panel renders the understood sound, seed behavior, hard constraints, ranking
preferences, selection policy, and execution outcome as separate sections. It never
labels a preference as a minimum. Replies distinguish an empty library, missing seeds,
constraint exhaustion, strict diversity, and ordinary successful matching.

Each frontend result is bound to an immutable snapshot of the request that produced it.
Beginning another request clears the old result, and saving uses the snapshot's playlist
name and Navidrome IDs rather than mutable composer text.

### Delivery slices and acceptance tests

The implementation is staged so behavior remains reviewable:

1. Add V2 types, planner schema, source-evidence validation, execution diagnostics, and a
   temporary adapter for legacy tool arguments.
2. Add cached empirical calibration and move every inferred scalar quality into the
   unified bounded preference scorer.
3. Add adaptive KNN expansion and diversity backfill with explicit cardinality reasons.
4. Bind immutable frontend results, render V2 diagnostics, validate live intent probes,
   then remove legacy hard-filter semantics.

Tests enforce invariants rather than relying only on prompt examples: preferences never
reduce cardinality; every strict result satisfies every constraint; unsupported hard
constraints cannot survive normalization; preference influence stays inside its relevance
window; diversity backfills unless explicitly strict; and returned count equals the
cardinality contract. Golden cases include wake-up music, "Mastodon but heavier,"
"over 140 BPM," explicit artist exclusion, a user-specified feature percentile, and
contradictory bounds.

Prompt-only hardening, silent automatic relaxation, and iterative LLM replanning are
deliberately rejected. Prompts are defense in depth, silent relaxation is untruthful, and
additional model turns increase latency and nondeterminism without adding information the
deterministic executor cannot derive itself.

## Status

- **Built & tested:** the whole stack — shared schema; orchestrator data spine
  (sqlite-vec KNN + filtered search); Navidrome-sourced CLAP analyzer (embeddings,
  zero-shot, librosa) plus the embed-server; cross-language storage handshake; LLM
  translation, search execution, and the conversational loop; the Navidrome
  `createPlaylist` writer; the `Bun.serve` entrypoint + SolidJS chat UI; and the
  Docker Compose for the Spark (Ollama is external).
- **Verified on the dev Mac:** all tests green (TS + Python); the orchestrator image
  builds, boots, and serves the bundled UI in production.
- **Built & tested (segment + decision overhaul):** windowed CLAP embeddings with
  per-segment mood scores and tags, best-passage segment KNN, track cards, and the
  decision-model membership gate over Ollama `/v1/systemone` — unit-tested on
  both halves with fakes, and the gate live-verified against tev1 and clef-flash
  on hand-built cards (correct accept/reject on both a request and its inverse).
- **Remaining — needs the Spark:**
  1. `docker compose build` — confirm the NGC image tag supports GB10/aarch64
     (`nvcr.io` login may be required) and the deps install atop the image's torch.
  2. Copy `.env.example` → `.env`; fill credentials, `MUSIC_PATH`, `OLLAMA_URL`, model.
  3. Ollama ≥ 0.35.0 with `ollama pull tev1` (or `clef-flash`/`clef`; see
     ollama#18769 if `clef-flash` errors with "non-finite logit" on `/v1/systemone`).
  4. `docker compose up`, then `docker compose run --rm analyzer` — the feature
     version bump re-ingests the whole library into segments (~60 windows/track).
  5. Live end-to-end smoke: a typed vibe → a real playlist in Navidrome; tune
     `MINSTREL_DECISION_BATCH` and the gate threshold on real decisions.
