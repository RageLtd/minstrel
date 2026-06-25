import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import {
  upsertTrack,
  upsertFeatures,
  setEmbedding,
  searchTracks,
} from "./repo";

function basis(...pairs: [number, number][]): Float32Array {
  const v = new Array<number>(EMBED_DIM).fill(0);
  for (const [i, val] of pairs) v[i] = val;
  return toEmbedding(v);
}

function seedTrack(
  db: ReturnType<typeof openDb>,
  path: string,
  embedding: Float32Array,
  energy: number,
): number {
  const id = upsertTrack(db, {
    mbid: null,
    filePath: path,
    contentHash: "h",
    fileMtime: 0,
    title: path,
    artist: "test",
  });
  upsertFeatures(db, id, { rmsEnergy: energy, bpm: 120 });
  setEmbedding(db, id, embedding);
  return id;
}

test("upsert by file_path is idempotent", () => {
  const db = openDb(":memory:");
  const a = upsertTrack(db, {
    mbid: null,
    filePath: "/m/song.flac",
    contentHash: "h1",
    fileMtime: 1,
  });
  const b = upsertTrack(db, {
    mbid: null,
    filePath: "/m/song.flac",
    contentHash: "h2",
    fileMtime: 2,
  });
  expect(a).toBe(b);
  const count = db.query(`SELECT count(*) AS n FROM tracks`).get() as {
    n: number;
  };
  expect(count.n).toBe(1);
  db.close();
});

test("similarity search orders by distance", () => {
  const db = openDb(":memory:");
  const t1 = seedTrack(db, "near", basis([0, 1]), 0.4);
  const t2 = seedTrack(db, "mid", basis([0, 1], [1, 0.1]), 0.9);
  const t3 = seedTrack(db, "far", basis([1, 1]), 0.9);

  const hits = searchTracks(db, basis([0, 1]), 3);
  expect(hits.map((h) => h.id)).toEqual([t1, t2, t3]);
  db.close();
});

test("energy filter narrows results after KNN", () => {
  const db = openDb(":memory:");
  seedTrack(db, "near", basis([0, 1]), 0.4); // low energy, closest
  const t2 = seedTrack(db, "mid", basis([0, 1], [1, 0.1]), 0.9);
  const t3 = seedTrack(db, "far", basis([1, 1]), 0.9);

  const hits = searchTracks(db, basis([0, 1]), 3, { energyMin: 0.8 });
  expect(hits.map((h) => h.id)).toEqual([t2, t3]);
  db.close();
});
