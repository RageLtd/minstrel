import type { Database } from "bun:sqlite";

// Domain operations over the Minstrel store: track upsert, feature upsert,
// embedding upsert, and the filtered similarity search the LLM tool drives.

export interface TrackInput {
  navidromeId: string;
  mbid?: string | null;
  title?: string;
  artist?: string;
  album?: string;
  navSize?: number | null;
}

export interface Features {
  bpm?: number;
  rmsEnergy?: number;
  spectralCentroid?: number;
  zsAggressive?: number;
  zsDanceable?: number;
  zsAcoustic?: number;
  extra?: Record<string, number>;
}

export type PreferenceFeature =
  | "tempo"
  | "energy"
  | "aggression"
  | "danceability"
  | "acousticness"
  | "novelty"
  | "rhythmicIrregularity"
  | "timbralComplexity"
  | "dynamicContrast"
  | "harmonicInstability";

export type FeatureValues = Partial<Record<PreferenceFeature, number>>;
export type FeatureDistributions = Record<PreferenceFeature, number[]>;

// Keep this in lockstep with analyzer/minstrel_analyzer/features.py. Queries
// ignore a partial backfill rather than comparing incompatible feature scales.
export const AUDIO_FEATURE_VERSION = 3;

export interface SearchHit {
  id: number;
  navidromeId: string;
  title: string | null;
  artist: string | null;
  album: string | null;
  distance: number;
  featureValues: FeatureValues;
}

const FEATURE_EXPRESSIONS: Record<PreferenceFeature, string> = {
  tempo: "f.bpm",
  energy: "f.rms_energy",
  aggression: "f.zs_aggressive",
  danceability: "f.zs_danceable",
  acousticness: "f.zs_acoustic",
  novelty: `CASE WHEN json_valid(f.extra_json)
                       AND json_extract(f.extra_json, '$.feature_version') = ${AUDIO_FEATURE_VERSION}
                  THEN CAST(json_extract(f.extra_json, '$.novelty') AS REAL) END`,
  rhythmicIrregularity: `CASE WHEN json_valid(f.extra_json)
                                   AND json_extract(f.extra_json, '$.feature_version') = ${AUDIO_FEATURE_VERSION}
                              THEN CAST(json_extract(f.extra_json, '$.rhythmic_irregularity') AS REAL) END`,
  timbralComplexity: `CASE WHEN json_valid(f.extra_json)
                                AND json_extract(f.extra_json, '$.feature_version') = ${AUDIO_FEATURE_VERSION}
                           THEN CAST(json_extract(f.extra_json, '$.timbral_complexity') AS REAL) END`,
  dynamicContrast: `CASE WHEN json_valid(f.extra_json)
                              AND json_extract(f.extra_json, '$.feature_version') = ${AUDIO_FEATURE_VERSION}
                         THEN CAST(json_extract(f.extra_json, '$.dynamic_contrast') AS REAL) END`,
  harmonicInstability: `CASE WHEN json_valid(f.extra_json)
                                  AND json_extract(f.extra_json, '$.feature_version') = ${AUDIO_FEATURE_VERSION}
                             THEN CAST(json_extract(f.extra_json, '$.harmonic_instability') AS REAL) END`,
};

/** Insert or update a track by its file path, returning the row id. */
export function upsertTrack(db: Database, t: TrackInput): number {
  const row = db
    .query(
      `INSERT INTO tracks (navidrome_id, mbid, title, artist, album, nav_size, analyzed_at)
       VALUES ($nid, $mbid, $title, $artist, $album, $size, unixepoch())
       ON CONFLICT(navidrome_id) DO UPDATE SET
         mbid = excluded.mbid,
         title = excluded.title,
         artist = excluded.artist,
         album = excluded.album,
         nav_size = excluded.nav_size,
         analyzed_at = unixepoch()
       RETURNING id`,
    )
    .get({
      $nid: t.navidromeId,
      $mbid: t.mbid ?? null,
      $title: t.title ?? null,
      $artist: t.artist ?? null,
      $album: t.album ?? null,
      $size: t.navSize ?? null,
    }) as { id: number };
  return row.id;
}

export function upsertFeatures(db: Database, trackId: number, f: Features): void {
  db.query(
    `INSERT INTO track_features
       (track_id, bpm, rms_energy, spectral_centroid, zs_aggressive, zs_danceable, zs_acoustic, extra_json)
     VALUES ($id, $bpm, $rms, $sc, $agg, $dance, $ac, $extra)
     ON CONFLICT(track_id) DO UPDATE SET
       bpm = excluded.bpm,
       rms_energy = excluded.rms_energy,
       spectral_centroid = excluded.spectral_centroid,
       zs_aggressive = excluded.zs_aggressive,
       zs_danceable = excluded.zs_danceable,
       zs_acoustic = excluded.zs_acoustic,
       extra_json = excluded.extra_json`,
  ).run({
    $id: trackId,
    $bpm: f.bpm ?? null,
    $rms: f.rmsEnergy ?? null,
    $sc: f.spectralCentroid ?? null,
    $agg: f.zsAggressive ?? null,
    $dance: f.zsDanceable ?? null,
    $ac: f.zsAcoustic ?? null,
    $extra: f.extra ? JSON.stringify(f.extra) : null,
  });
}

/** vec0 has no UPSERT, so replace by primary key. */
export function setEmbedding(
  db: Database,
  trackId: number,
  embedding: Float32Array,
): void {
  db.transaction(() => {
    db.query(`DELETE FROM track_vec WHERE track_id = ?`).run(trackId);
    db.query(`INSERT INTO track_vec(track_id, embedding) VALUES (?, ?)`).run(
      trackId,
      embedding,
    );
  })();
}

/**
 * k nearest tracks to a query embedding. Constraints are applied after fusion so
 * diagnostics can distinguish retrieval from eligibility.
 */
export function searchTracks(
  db: Database,
  query: Float32Array,
  k: number,
  includeFeatureValues = false,
): SearchHit[] {
  const featureColumns = includeFeatureValues
    ? `,\n           ${Object.entries(FEATURE_EXPRESSIONS)
        .map(([name, expression]) => `${expression} AS ${name}`)
        .join(",\n           ")}`
    : "";
  const sql = `
    SELECT t.id, t.navidrome_id AS navidromeId, t.title, t.artist, t.album, v.distance${featureColumns}
      FROM track_vec v
      JOIN tracks t ON t.id = v.track_id
      JOIN track_features f ON f.track_id = v.track_id
     WHERE v.embedding MATCH ? AND k = ?
     ORDER BY v.distance, t.id
     LIMIT ?`;

  type SearchRow = Omit<SearchHit, "featureValues"> &
    Record<PreferenceFeature, number | null>;
  const rows = db
    .query(sql)
    .all(query, k, k) as SearchRow[];
  return rows.map((row) => {
    const featureValues: FeatureValues = {};
    for (const name of Object.keys(FEATURE_EXPRESSIONS) as PreferenceFeature[]) {
      const value = row[name];
      if (typeof value === "number" && Number.isFinite(value)) {
        featureValues[name] = value;
      }
    }
    const {
      tempo,
      energy,
      aggression,
      danceability,
      acousticness,
      novelty,
      rhythmicIrregularity,
      timbralComplexity,
      dynamicContrast,
      harmonicInstability,
      ...hit
    } = row;
    return { ...hit, featureValues };
  });
}

export function searchableTrackCount(db: Database): number {
  const row = db
    .query(
      `SELECT count(*) AS n
         FROM track_vec v
         JOIN tracks t ON t.id = v.track_id
         JOIN track_features f ON f.track_id = v.track_id`,
    )
    .get() as { n: number };
  return row.n;
}

export function featureDistributions(db: Database): FeatureDistributions {
  const columns = Object.entries(FEATURE_EXPRESSIONS)
    .map(([name, expression]) => `${expression} AS ${name}`)
    .join(",\n             ");
  const rows = db
    .query(
      `SELECT ${columns}
         FROM track_features f
         JOIN tracks t ON t.id = f.track_id
         JOIN track_vec v ON v.track_id = f.track_id`,
    )
    .all() as Record<PreferenceFeature, number | null>[];
  return Object.fromEntries(
    (Object.keys(FEATURE_EXPRESSIONS) as PreferenceFeature[]).map((name) => [
      name,
      rows
        .map((row) => row[name])
        .filter((value): value is number =>
          typeof value === "number" && Number.isFinite(value)
        )
        .sort((a, b) => a - b),
    ]),
  ) as FeatureDistributions;
}

export function audioFeatureDataReady(db: Database): boolean {
  const row = db
    .query(
      `SELECT value
         FROM analysis_meta
        WHERE key = 'audio_feature_version'`,
    )
    .get() as { value: number } | null;
  return row?.value === AUDIO_FEATURE_VERSION;
}

/** All stored embeddings for tracks by a given artist (case-insensitive). */
export function artistEmbeddings(db: Database, artist: string): Float32Array[] {
  const rows = db
    .query(
      `SELECT v.embedding AS e
         FROM track_vec v
         JOIN tracks t ON t.id = v.track_id
        WHERE t.artist = ? COLLATE NOCASE`,
    )
    .all(artist) as { e: Uint8Array }[];
  // slice() copies into a fresh, 4-byte-aligned buffer at offset 0.
  return rows.map((r) => new Float32Array(r.e.slice().buffer));
}
