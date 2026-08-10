import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import {
  AUDIO_FEATURE_VERSION,
  upsertTrack,
  upsertFeatures,
  setEmbedding,
} from "./repo";
import type { SearchHit } from "./repo";
import {
  executeSearch,
  fuseSearches,
  percentileRank,
  rerankByPreferences,
  selectWithArtistVariety,
  type BaseCandidate,
} from "./search";
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
  extra?: Record<string, number>,
): number {
  const id = upsertTrack(db, {
    navidromeId: path,
    title: path,
    artist,
  });
  upsertFeatures(db, id, {
    rmsEnergy: 0.5,
    bpm: 120,
    extra: extra ? { feature_version: AUDIO_FEATURE_VERSION, ...extra } : undefined,
  });
  setEmbedding(db, id, emb);
  return id;
}

const noEmbed = async (): Promise<Float32Array> => {
  throw new Error("embedText should not be called");
};

function fakeHit(id: number): SearchHit {
  return {
    id,
    navidromeId: String(id),
    title: String(id),
    artist: String(id),
    album: null,
    distance: 0,
    featureValues: {},
  };
}

type QueryOverrides = Omit<Partial<SearchQuery>, "selection"> & {
  selection?: Partial<SearchQuery["selection"]>;
};

function query(overrides: QueryOverrides = {}): SearchQuery {
  const { selection, ...rest } = overrides;
  return {
    version: 2,
    semanticText: "",
    constraints: [],
    preferences: [],
    selection: {
      count: 30,
      artistVariety: "balanced",
      ...selection,
    },
    warnings: [],
    ...rest,
  };
}

function markAudioFeaturesReady(db: ReturnType<typeof openDb>): void {
  db.query(
    `INSERT INTO analysis_meta (key, value)
     VALUES ('audio_feature_version', ?)
     ON CONFLICT(key) DO UPDATE SET value=excluded.value`,
  ).run(AUDIO_FEATURE_VERSION);
}

test("seed artist (case-insensitive) drives the search", async () => {
  const db = openDb(":memory:");
  const mastodon = seed(db, "a", "Mastodon", basis([0, 1]));
  seed(db, "b", "Tool", basis([1, 1]));

  const searchQuery = query({
    seedArtists: ["mastodon"],
    selection: { count: 5 },
  });
  const { hits, missingSeedArtists } = await executeSearch(db, noEmbed, searchQuery);

  expect(hits[0]!.id).toBe(mastodon);
  expect(missingSeedArtists).toEqual([]);
  db.close();
});

test("exploratory seed search excludes the seed artist case-insensitively", async () => {
  const db = openDb(":memory:");
  seed(db, "a", "Periphery", basis([0, 1]));
  seed(db, "b", "PERIPHERY", basis([0, 1], [1, 0.01]));
  const otherBand = seed(db, "c", "Protest The Hero", basis([0, 1], [1, 0.02]));

  const searchQuery = query({
    seedArtists: ["periphery"],
    excludeSeedArtists: true,
    selection: { count: 5 },
  });
  const { hits, missingSeedArtists } = await executeSearch(db, noEmbed, searchQuery);

  expect(hits.map((hit) => hit.id)).toEqual([otherBand]);
  expect(missingSeedArtists).toEqual([]);
  db.close();
});

test("missing seed artist falls back to semantic text and is reported", async () => {
  const db = openDb(":memory:");
  seed(db, "a", "Mastodon", basis([0, 1]));
  const tool = seed(db, "b", "Tool", basis([1, 1]));

  const embedText = async (): Promise<Float32Array> => basis([1, 1]); // points at Tool
  const searchQuery = query({
    seedArtists: ["Ghost"],
    semanticText: "knotty prog",
    selection: { count: 5 },
  });
  const { hits, missingSeedArtists } = await executeSearch(db, embedText, searchQuery);

  expect(missingSeedArtists).toEqual(["Ghost"]);
  expect(hits[0]!.id).toBe(tool);
  db.close();
});

test("missing seed and no semantic text yields no hits", async () => {
  const db = openDb(":memory:");
  seed(db, "a", "Mastodon", basis([0, 1]));

  const searchQuery = query({
    seedArtists: ["Ghost"],
    selection: { count: 5 },
  });
  const { hits, missingSeedArtists } = await executeSearch(db, noEmbed, searchQuery);

  expect(hits).toEqual([]);
  expect(missingSeedArtists).toEqual(["Ghost"]);
  db.close();
});

test("multiple seeds preserve their separate neighborhoods instead of averaging them", async () => {
  const db = openDb(":memory:");
  const primus = seed(db, "primus", "Primus", basis([0, 1]));
  const igorrr = seed(db, "igorrr", "Igorrr", basis([1, 1]));
  const generic = seed(db, "generic", "Generic Metal", basis([0, 1], [1, 1]));

  const searchQuery = query({
    seedArtists: ["Primus", "Igorrr"],
    selection: { count: 2 },
  });
  const { hits } = await executeSearch(db, noEmbed, searchQuery);

  expect(hits.map((hit) => hit.id)).toEqual([primus, igorrr]);
  expect(hits.map((hit) => hit.id)).not.toContain(generic);
  db.close();
});

test("semantic rank refines seed rank without rewarding absence", () => {
  const seedFirst = fakeHit(1);
  const seedSecond = fakeHit(2);
  const semanticResults = Array.from({ length: 100 }, (_, index) =>
    fakeHit(index === 99 ? seedFirst.id : index + 10),
  );

  const fused = fuseSearches(
    [[seedFirst, seedSecond]],
    semanticResults,
    semanticResults.length,
  );

  const fusedHits = fused.map(({ hit }) => hit);
  expect(fusedHits.indexOf(seedFirst)).toBeLessThan(fusedHits.indexOf(seedSecond));
});

test("empirical percentile rank handles ties without min-max distortion", () => {
  expect(percentileRank([1, 2, 2, 10], 2)).toBeCloseTo(0.5);
  expect(percentileRank([1, 2, 2, 10], 10)).toBe(1);
});

test("preference scoring cannot pull candidates beyond the outer relevance window", () => {
  const scores = [0, 1, 10, 11, 100];
  const candidates: BaseCandidate[] = scores.map((baseScore, index) => ({
    hit: {
      ...fakeHit(index + 1),
      featureValues: { novelty: index === scores.length - 1 ? 1 : 0 },
    },
    baseScore,
  }));
  const reranked = rerankByPreferences(
    candidates,
    [
      {
        feature: "novelty",
        direction: "higher",
        strength: "strong",
        evidence: "weird",
      },
    ],
    {
      tempo: [],
      energy: [],
      aggression: [],
      danceability: [],
      acousticness: [],
      novelty: [0, 0, 0, 0, 1],
      rhythmicIrregularity: [],
      timbralComplexity: [],
      dynamicContrast: [],
      harmonicInstability: [],
    },
    2,
  );

  expect(reranked.slice(0, 2).map((hit) => hit.id)).not.toContain(5);
});

test("wide variety runs before a looser explicit artist cap", () => {
  const firstA = { ...fakeHit(1), artist: "A" };
  const secondA = { ...fakeHit(2), artist: "A" };
  const firstB = { ...fakeHit(3), artist: "B" };

  const result = selectWithArtistVariety(
    [firstA, secondA, firstB],
    3,
    "wide",
    2,
  );

  expect(result.hits.map((hit) => hit.id)).toEqual([1, 3, 2]);
});

test("novelty preference promotes unusual relevant tracks without overriding relevance", async () => {
  const db = openDb(":memory:");
  const common = seed(db, "common", "Common", basis([0, 1]), { novelty: 0 });
  const unusual = seed(db, "unusual", "Unusual", basis([0, 1], [1, 0.1]), {
    novelty: 1,
  });
  seed(db, "filler-1", "Filler 1", basis([0, 1], [1, 0.2]), { novelty: 0 });
  seed(db, "filler-2", "Filler 2", basis([0, 1], [1, 0.3]), { novelty: 0 });
  seed(db, "filler-3", "Filler 3", basis([0, 1], [1, 0.4]), { novelty: 0 });
  const irrelevant = seed(db, "irrelevant", "Irrelevant", basis([1, 1]), {
    novelty: 1,
  });
  markAudioFeaturesReady(db);

  const searchQuery = query({
    semanticText: "experimental metal",
    preferences: [
      {
        feature: "novelty",
        direction: "higher",
        strength: "strong",
        evidence: "experimental",
      },
    ],
    selection: { count: 2 },
  });
  const { hits } = await executeSearch(db, async () => basis([0, 1]), searchQuery);

  expect(hits.map((hit) => hit.id)).toEqual([unusual, common]);
  expect(hits.map((hit) => hit.id)).not.toContain(irrelevant);
  db.close();
});

test("feature reranking waits for a complete feature-version backfill", async () => {
  const db = openDb(":memory:");
  const common = seed(db, "common", "Common", basis([0, 1]), {
    feature_version: AUDIO_FEATURE_VERSION - 1,
    novelty: 0,
  });
  seed(db, "unusual", "Unusual", basis([0, 1], [1, 0.1]), { novelty: 1 });

  const searchQuery = query({
    semanticText: "experimental metal",
    preferences: [
      {
        feature: "novelty",
        direction: "higher",
        strength: "strong",
        evidence: "experimental",
      },
    ],
    selection: { count: 2 },
  });
  const { hits, diagnostics } = await executeSearch(
    db,
    async () => basis([0, 1]),
    searchQuery,
  );

  expect(hits[0]!.id).toBe(common);
  expect(hits.every((hit) => Object.keys(hit.featureValues).length === 0)).toBe(true);
  expect(diagnostics.preferenceReranked).toBe(false);
  expect(diagnostics.warnings).toContain(
    "Derived audio preferences are unavailable until analysis completes.",
  );
  db.close();
});

test("balanced variety prevents a large artist catalog from dominating", async () => {
  const db = openDb(":memory:");
  seed(db, "slip-1", "Slipknot", basis([0, 1]));
  seed(db, "slip-2", "Slipknot", basis([0, 1], [1, 0.01]));
  seed(db, "slip-3", "Slipknot", basis([0, 1], [1, 0.02]));
  seed(db, "primus", "Primus", basis([0, 1], [1, 0.03]));
  seed(db, "igorrr", "Igorrr", basis([0, 1], [1, 0.04]));

  const embedText = async (): Promise<Float32Array> => basis([0, 1]);
  const searchQuery = query({
    semanticText: "heavy music",
    selection: { count: 5 },
  });
  const { hits, diagnostics } = await executeSearch(db, embedText, searchQuery);

  expect(hits.map((hit) => hit.artist)).toEqual([
    "Slipknot",
    "Slipknot",
    "Primus",
    "Igorrr",
    "Slipknot",
  ]);
  expect(diagnostics.diversityBackfilled).toBe(1);
  db.close();
});

test("wide variety backfills repeated artists to preserve requested cardinality", async () => {
  const db = openDb(":memory:");
  seed(db, "slip-1", "Slipknot", basis([0, 1]));
  seed(db, "slip-2", "Slipknot", basis([0, 1], [1, 0.01]));
  seed(db, "primus", "Primus", basis([0, 1], [1, 0.02]));
  seed(db, "igorrr", "Igorrr", basis([0, 1], [1, 0.03]));

  const embedText = async (): Promise<Float32Array> => basis([0, 1]);
  const searchQuery = query({
    semanticText: "weird eclectic heavy music",
    selection: { count: 4, artistVariety: "wide" },
  });
  const { hits, diagnostics } = await executeSearch(db, embedText, searchQuery);

  expect(hits.map((hit) => hit.artist)).toEqual([
    "Slipknot",
    "Primus",
    "Igorrr",
    "Slipknot",
  ]);
  expect(diagnostics.returned).toBe(4);
  expect(diagnostics.diversityBackfilled).toBe(1);
  db.close();
});

test("a strict artist cap may truthfully underfill", async () => {
  const db = openDb(":memory:");
  seed(db, "slip-1", "Slipknot", basis([0, 1]));
  seed(db, "slip-2", "Slipknot", basis([0, 1], [1, 0.01]));
  seed(db, "primus", "Primus", basis([0, 1], [1, 0.02]));

  const searchQuery = query({
    semanticText: "metal",
    selection: { count: 3, artistVariety: "wide", strictArtistCap: 1 },
  });
  const { hits, diagnostics } = await executeSearch(
    db,
    async () => basis([0, 1]),
    searchQuery,
  );

  expect(hits).toHaveLength(2);
  expect(diagnostics.shortfallReason).toBe("strictArtistCap");
  db.close();
});

test("soft preferences never reduce requested cardinality", async () => {
  const db = openDb(":memory:");
  for (let index = 0; index < 20; index += 1) {
    const id = seed(
      db,
      `track-${index}`,
      `Artist ${index}`,
      basis([0, 1], [1, index * 0.01]),
    );
    upsertFeatures(db, id, { bpm: 100 + index, rmsEnergy: index / 20 });
  }

  const searchQuery = query({
    semanticText: "wake-up music",
    preferences: [
      {
        feature: "energy",
        direction: "higher",
        strength: "strong",
        evidence: "wake-up",
      },
      {
        feature: "tempo",
        direction: "higher",
        strength: "normal",
        evidence: "wake-up",
      },
    ],
    selection: { count: 10 },
  });
  const { hits, diagnostics } = await executeSearch(
    db,
    async () => basis([0, 1]),
    searchQuery,
  );

  expect(hits).toHaveLength(10);
  expect(diagnostics.shortfallReason).toBeUndefined();
  expect(diagnostics.preferenceReranked).toBe(true);
  db.close();
});

test("adaptive retrieval expands until explicit constraints can fill the request", async () => {
  const db = openDb(":memory:");
  for (let index = 0; index < 140; index += 1) {
    const id = seed(
      db,
      `track-${index}`,
      `Artist ${index}`,
      basis([0, 1], [1, index * 0.01]),
    );
    upsertFeatures(db, id, {
      bpm: index < 128 ? 100 : 160,
      rmsEnergy: 0.5,
    });
  }

  const searchQuery = query({
    semanticText: "fast metal",
    constraints: [
      { feature: "tempoBpm", operator: "min", value: 150, evidence: "over 150 BPM" },
    ],
    selection: { count: 5 },
  });
  const { hits, diagnostics } = await executeSearch(
    db,
    async () => basis([0, 1]),
    searchQuery,
  );

  expect(hits).toHaveLength(5);
  expect(diagnostics.retrievalPasses).toBe(2);
  expect(diagnostics.exhaustedCorpus).toBe(true);
  db.close();
});

test("explicit feature-percentile constraints may truthfully return fewer tracks", async () => {
  const db = openDb(":memory:");
  const expected: number[] = [];
  for (let index = 0; index < 10; index += 1) {
    const id = seed(
      db,
      `track-${index}`,
      `Artist ${index}`,
      basis([0, 1], [1, index * 0.01]),
    );
    upsertFeatures(db, id, {
      bpm: 120,
      rmsEnergy: 0.5,
      zsAggressive: index / 10,
    });
    if (index >= 8) expected.push(id);
  }

  const searchQuery = query({
    semanticText: "aggressive metal",
    constraints: [
      {
        feature: "featurePercentile",
        percentileFeature: "aggression",
        operator: "min",
        value: 0.8,
        evidence: "top 20% most aggressive",
      },
    ],
    selection: { count: 5, strictArtistCap: 10 },
  });
  const { hits, diagnostics } = await executeSearch(
    db,
    async () => basis([0, 1]),
    searchQuery,
  );

  expect(hits.map((hit) => hit.id)).toEqual(expected);
  expect(diagnostics.shortfallReason).toBe("constraints");
  expect(diagnostics.returned).toBe(2);
  db.close();
});
