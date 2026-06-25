import type { Database } from "bun:sqlite";

// Domain operations over the Minstrel store: track upsert, feature upsert,
// embedding upsert, and the filtered similarity search the LLM tool drives.

export interface TrackInput {
  mbid: string | null;
  filePath: string;
  contentHash: string;
  fileMtime: number;
  title?: string;
  artist?: string;
  album?: string;
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

export interface SearchFilters {
  bpmMin?: number;
  bpmMax?: number;
  energyMin?: number;
  aggressiveMin?: number;
  danceableMin?: number;
  acousticMax?: number;
}

export interface SearchHit {
  id: number;
  title: string | null;
  artist: string | null;
  album: string | null;
  distance: number;
}

/** Insert or update a track by its file path, returning the row id. */
export function upsertTrack(db: Database, t: TrackInput): number {
  const row = db
    .query(
      `INSERT INTO tracks (mbid, file_path, content_hash, file_mtime, title, artist, album, analyzed_at)
       VALUES ($mbid, $path, $hash, $mtime, $title, $artist, $album, unixepoch())
       ON CONFLICT(file_path) DO UPDATE SET
         mbid = excluded.mbid,
         content_hash = excluded.content_hash,
         file_mtime = excluded.file_mtime,
         title = excluded.title,
         artist = excluded.artist,
         album = excluded.album,
         analyzed_at = unixepoch()
       RETURNING id`,
    )
    .get({
      $mbid: t.mbid,
      $path: t.filePath,
      $hash: t.contentHash,
      $mtime: t.fileMtime,
      $title: t.title ?? null,
      $artist: t.artist ?? null,
      $album: t.album ?? null,
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
 * k nearest tracks to a query embedding, then narrowed by scalar filters.
 * vec0's KNN runs before the joins, so we over-fetch and trim to `k` after
 * filtering — overfetch trades a little work for not starving filtered queries.
 */
export function searchTracks(
  db: Database,
  query: Float32Array,
  k: number,
  filters: SearchFilters = {},
  overfetch = 6,
): SearchHit[] {
  const conds: string[] = [];
  const filterParams: number[] = [];
  const add = (sql: string, val: number | undefined) => {
    if (val !== undefined) {
      conds.push(sql);
      filterParams.push(val);
    }
  };
  add("f.bpm >= ?", filters.bpmMin);
  add("f.bpm <= ?", filters.bpmMax);
  add("f.rms_energy >= ?", filters.energyMin);
  add("f.zs_aggressive >= ?", filters.aggressiveMin);
  add("f.zs_danceable >= ?", filters.danceableMin);
  add("f.zs_acoustic <= ?", filters.acousticMax);

  const fetchK = Math.min(k * overfetch, 2000);
  const sql = `
    SELECT t.id, t.title, t.artist, t.album, v.distance
      FROM track_vec v
      JOIN tracks t ON t.id = v.track_id
      JOIN track_features f ON f.track_id = v.track_id
     WHERE v.embedding MATCH ? AND k = ?
       ${conds.length ? `AND ${conds.join(" AND ")}` : ""}
     ORDER BY v.distance
     LIMIT ?`;

  return db.query(sql).all(query, fetchK, ...filterParams, k) as SearchHit[];
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
