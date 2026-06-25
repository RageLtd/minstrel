import { Database } from "bun:sqlite";
import * as sqliteVec from "sqlite-vec";
import schemaSql from "../schema.sql" with { type: "text" };

/** CLAP audio/text embeddings are 512-dimensional. */
export const EMBED_DIM = 512;

// Apple's bundled SQLite is built without loadable-extension support, so Bun must
// be pointed at a Homebrew SQLite before any Database is constructed. On Linux (the
// Spark host) the system SQLite loads extensions fine, so this is a darwin-only step.
const DARWIN_SQLITE =
  process.env.MINSTREL_SQLITE_PATH ??
  "/opt/homebrew/opt/sqlite/lib/libsqlite3.dylib";

let customSqliteConfigured = false;
function ensureExtensionCapableSqlite(): void {
  if (process.platform === "darwin" && !customSqliteConfigured) {
    Database.setCustomSQLite(DARWIN_SQLITE);
    customSqliteConfigured = true;
  }
}

// Schema is the shared ../schema.sql — the single source of truth applied by
// both this orchestrator and the Python analyzer.
export function openDb(path: string): Database {
  ensureExtensionCapableSqlite();
  const db = new Database(path, { create: true });
  db.exec("PRAGMA journal_mode = WAL;");
  db.exec("PRAGMA foreign_keys = ON;");
  sqliteVec.load(db);
  db.exec(schemaSql);
  return db;
}

/** Pack a plain number[] into the float32 BLOB sqlite-vec expects. */
export function toEmbedding(values: readonly number[]): Float32Array {
  if (values.length !== EMBED_DIM) {
    throw new Error(`embedding must be ${EMBED_DIM} dims, got ${values.length}`);
  }
  return Float32Array.from(values);
}

export interface Neighbour {
  track_id: number;
  distance: number;
}

/** k-nearest tracks to a query embedding, cosine distance ascending. */
export function searchSimilar(
  db: Database,
  query: Float32Array,
  k: number,
): Neighbour[] {
  return db
    .query(
      `SELECT track_id, distance
         FROM track_vec
        WHERE embedding MATCH ?
          AND k = ?
        ORDER BY distance`,
    )
    .all(query, k) as Neighbour[];
}

export function insertEmbedding(
  db: Database,
  trackId: number,
  embedding: Float32Array,
): void {
  db.query(`INSERT INTO track_vec(track_id, embedding) VALUES (?, ?)`).run(
    trackId,
    embedding,
  );
}
