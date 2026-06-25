import { test, expect } from "bun:test";
import {
  openDb,
  toEmbedding,
  insertEmbedding,
  searchSimilar,
  EMBED_DIM,
} from "./db";

/** Build a sparse 512-dim vector from (index, value) pairs. */
function basis(...pairs: [number, number][]): Float32Array {
  const v = new Array<number>(EMBED_DIM).fill(0);
  for (const [i, val] of pairs) v[i] = val;
  return toEmbedding(v);
}

test("sqlite-vec loads and returns cosine-nearest neighbours in order", () => {
  const db = openDb(":memory:");

  insertEmbedding(db, 1, basis([0, 1])); // A: pure dim-0
  insertEmbedding(db, 2, basis([1, 1])); // B: pure dim-1
  insertEmbedding(db, 3, basis([0, 1], [1, 0.9])); // C: mostly dim-0

  const results = searchSimilar(db, basis([0, 1]), 3); // query == A

  expect(results.map((r) => r.track_id)).toEqual([1, 3, 2]);
  expect(results[0]!.distance).toBeCloseTo(0, 5);

  db.close();
});

test("schema creates tracks + features tables", () => {
  const db = openDb(":memory:");
  const names = db
    .query(`SELECT name FROM sqlite_master WHERE type='table' ORDER BY name`)
    .all() as { name: string }[];
  const set = new Set(names.map((n) => n.name));
  expect(set.has("tracks")).toBe(true);
  expect(set.has("track_features")).toBe(true);
  db.close();
});
