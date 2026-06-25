-- Minstrel storage schema — the single source of truth shared by the TypeScript
-- orchestrator (src/db.ts) and the Python analyzer (minstrel_analyzer/store.py).
-- Both apply it idempotently. Never edit the DDL anywhere but here.

CREATE TABLE IF NOT EXISTS tracks (
  id           INTEGER PRIMARY KEY,
  mbid         TEXT UNIQUE,
  file_path    TEXT NOT NULL UNIQUE,
  content_hash TEXT NOT NULL,
  file_mtime   INTEGER NOT NULL,
  navidrome_id TEXT,
  title        TEXT,
  artist       TEXT,
  album        TEXT,
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

CREATE VIRTUAL TABLE IF NOT EXISTS track_vec USING vec0(
  track_id  INTEGER PRIMARY KEY,
  embedding FLOAT[512] distance_metric=cosine
);
