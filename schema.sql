-- Minstrel storage schema — the single source of truth shared by the TypeScript
-- orchestrator (src/db.ts) and the Python analyzer (minstrel_analyzer/store.py).
-- Both apply it idempotently. Never edit the DDL anywhere but here.

CREATE TABLE IF NOT EXISTS tracks (
  id           INTEGER PRIMARY KEY,
  navidrome_id TEXT NOT NULL UNIQUE,  -- stable Navidrome PID; the join key everywhere
  mbid         TEXT,                  -- secondary anchor for re-linking if PIDs rotate
  title        TEXT,
  artist       TEXT,
  album        TEXT,
  nav_size     INTEGER,               -- Navidrome-reported size; incremental change signal
  analyzed_at  INTEGER
);

CREATE TABLE IF NOT EXISTS track_features (
  track_id          INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
  bpm               REAL,
  rms_energy        REAL,
  spectral_centroid REAL,
  zs_aggressive     REAL,
  zs_danceable      REAL,
  zs_acoustic       REAL,
  extra_json        TEXT
);

CREATE TABLE IF NOT EXISTS analysis_meta (
  key   TEXT PRIMARY KEY,
  value INTEGER NOT NULL
);

-- Track-level centroid: the renormalised mean of the track's segment embeddings.
-- Used for seed-artist centroids and as a coarse fallback, not for matching.
CREATE VIRTUAL TABLE IF NOT EXISTS track_vec USING vec0(
  track_id  INTEGER PRIMARY KEY,
  embedding FLOAT[512] distance_metric=cosine
);

-- One row per analysed window of a track (10 s windows, 5 s hop), so a track
-- can match on any passage rather than on a single random crop.
CREATE TABLE IF NOT EXISTS track_segments (
  id            INTEGER PRIMARY KEY,
  track_id      INTEGER NOT NULL REFERENCES tracks(id) ON DELETE CASCADE,
  start_s       REAL NOT NULL,
  end_s         REAL NOT NULL,
  zs_aggressive REAL,
  zs_danceable  REAL,
  zs_acoustic   REAL,
  tags_json     TEXT,                 -- zero-shot tags per group; see tags.py
  tag_version   INTEGER               -- TAG_VERSION the tags were computed with
);
CREATE INDEX IF NOT EXISTS track_segments_track_id ON track_segments(track_id);

-- vec0 has no foreign keys; store.py/repo.ts delete these alongside segments.
CREATE VIRTUAL TABLE IF NOT EXISTS segment_vec USING vec0(
  segment_id INTEGER PRIMARY KEY,
  embedding  FLOAT[512] distance_metric=cosine
);
