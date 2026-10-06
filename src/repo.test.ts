import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import {
  upsertTrack,
  upsertFeatures,
  setEmbedding,
  replaceSegments,
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
  replaceSegments(db, id, [{ startS: 0, endS: 10, embedding }]);
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

test("a track matches on its best passage, not its centroid", () => {
  const db = openDb(":memory:");
  // Two-faced track: a dim-0 passage, then a dim-1 passage. Its centroid sits
  // between them, so centroid search would rank the single-passage track first.
  const twoFaced = seedTrack(db, "two-faced", basis([0, 1], [1, 1]), 0.5);
  replaceSegments(db, twoFaced, [
    { startS: 0, endS: 10, embedding: basis([0, 1]) },
    { startS: 5, endS: 15, embedding: basis([1, 1]) },
  ]);
  const steady = seedTrack(db, "steady", basis([0, 1], [1, 0.5]), 0.5);

  const hits = searchTracks(db, basis([1, 1]), 2);

  expect(hits.map((hit) => hit.id)).toEqual([twoFaced, steady]);
  expect(hits[0]!.distance).toBeCloseTo(0, 5);
  expect(hits[0]!.bestSegment).toEqual({ startS: 5, endS: 15 });
  expect(hits).toHaveLength(2); // one row per track, never per segment
  db.close();
});

test("replacing segments drops the old vectors", () => {
  const db = openDb(":memory:");
  const id = seedTrack(db, "track", basis([0, 1]), 0.5);
  replaceSegments(db, id, [
    { startS: 0, endS: 10, embedding: basis([0, 1]) },
    { startS: 5, endS: 15, embedding: basis([1, 1]) },
  ]);
  replaceSegments(db, id, [{ startS: 0, endS: 10, embedding: basis([2, 1]) }]);

  const rows = db.query(`SELECT count(*) AS n FROM track_segments`).get() as {
    n: number;
  };
  const vectors = db.query(`SELECT count(*) AS n FROM segment_vec`).get() as {
    n: number;
  };
  expect(rows.n).toBe(1);
  expect(vectors.n).toBe(1);
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
