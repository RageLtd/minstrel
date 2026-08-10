import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import {
  upsertTrack,
  upsertFeatures,
  setEmbedding,
  searchTracks,
  featureDistributions,
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
  bpm = 120,
): number {
  const id = upsertTrack(db, {
    navidromeId: path,
    title: path,
    artist: "test",
  });
  upsertFeatures(db, id, { rmsEnergy: energy, bpm });
  setEmbedding(db, id, embedding);
  return id;
}

test("upsert by navidrome_id is idempotent", () => {
  const db = openDb(":memory:");
  const a = upsertTrack(db, { navidromeId: "nav-1", title: "first" });
  const b = upsertTrack(db, { navidromeId: "nav-1", title: "renamed" });
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

test("optional feature projection includes raw values for deterministic scoring", () => {
  const db = openDb(":memory:");
  const t1 = seedTrack(db, "near", basis([0, 1]), 0.4, 100);
  const t2 = seedTrack(db, "mid", basis([0, 1], [1, 0.1]), 0.9, 140);
  const t3 = seedTrack(db, "far", basis([1, 1]), 0.9, 160);

  const hits = searchTracks(db, basis([0, 1]), 3, true);
  expect(hits.map((h) => h.id)).toEqual([t1, t2, t3]);
  expect(hits.map((hit) => hit.featureValues.tempo)).toEqual([100, 140, 160]);
  db.close();
});

test("feature distributions retain empirical values rather than min-max thresholds", () => {
  const db = openDb(":memory:");
  seedTrack(db, "low", basis([0, 1]), 0.1);
  seedTrack(db, "mid", basis([0, 1], [1, 0.01]), 0.2);
  seedTrack(db, "high", basis([0, 1], [1, 0.02]), 0.9);

  const distributions = featureDistributions(db);

  expect(distributions.energy).toEqual([0.1, 0.2, 0.9]);
  db.close();
});

test("feature distributions reflect same-second feature updates", () => {
  const db = openDb(":memory:");
  const id = seedTrack(db, "track", basis([0, 1]), 0.1);
  expect(featureDistributions(db).energy).toEqual([0.1]);

  upsertFeatures(db, id, { rmsEnergy: 0.9, bpm: 120 });

  expect(featureDistributions(db).energy).toEqual([0.9]);
  db.close();
});
