import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import { upsertTrack, upsertFeatures, setEmbedding } from "./repo";
import { executeSearch } from "./search";
import type { SearchQuery } from "./tools";

function basis(...pairs: [number, number][]): Float32Array {
  const v = new Array<number>(EMBED_DIM).fill(0);
  for (const [i, val] of pairs) v[i] = val;
  return toEmbedding(v);
}

function seed(
  db: ReturnType<typeof openDb>,
  path: string,
  artist: string,
  emb: Float32Array,
): number {
  const id = upsertTrack(db, {
    mbid: null,
    filePath: path,
    contentHash: "h",
    fileMtime: 0,
    title: path,
    artist,
  });
  upsertFeatures(db, id, { rmsEnergy: 0.5, bpm: 120 });
  setEmbedding(db, id, emb);
  return id;
}

const noEmbed = async (): Promise<Float32Array> => {
  throw new Error("embedText should not be called");
};

test("seed artist (case-insensitive) drives the search", async () => {
  const db = openDb(":memory:");
  const mastodon = seed(db, "a", "Mastodon", basis([0, 1]));
  seed(db, "b", "Tool", basis([1, 1]));

  const query: SearchQuery = { seedArtists: ["mastodon"], filters: {}, count: 5 };
  const { hits, missingSeedArtists } = await executeSearch(db, noEmbed, query);

  expect(hits[0]!.id).toBe(mastodon);
  expect(missingSeedArtists).toEqual([]);
  db.close();
});

test("missing seed artist falls back to semantic text and is reported", async () => {
  const db = openDb(":memory:");
  seed(db, "a", "Mastodon", basis([0, 1]));
  const tool = seed(db, "b", "Tool", basis([1, 1]));

  const embedText = async (): Promise<Float32Array> => basis([1, 1]); // points at Tool
  const query: SearchQuery = {
    seedArtists: ["Ghost"],
    semanticText: "knotty prog",
    filters: {},
    count: 5,
  };
  const { hits, missingSeedArtists } = await executeSearch(db, embedText, query);

  expect(missingSeedArtists).toEqual(["Ghost"]);
  expect(hits[0]!.id).toBe(tool);
  db.close();
});

test("missing seed and no semantic text yields no hits", async () => {
  const db = openDb(":memory:");
  seed(db, "a", "Mastodon", basis([0, 1]));

  const query: SearchQuery = { seedArtists: ["Ghost"], filters: {}, count: 5 };
  const { hits, missingSeedArtists } = await executeSearch(db, noEmbed, query);

  expect(hits).toEqual([]);
  expect(missingSeedArtists).toEqual(["Ghost"]);
  db.close();
});
