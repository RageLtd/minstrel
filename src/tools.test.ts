import { test, expect } from "bun:test";
import {
  SEARCH_TRACKS_TOOL,
  SYSTEM_PROMPT,
  parseSearchToolCall,
} from "./tools";

test("normalizes descriptive wake-up language into preferences, not constraints", () => {
  const userText = "I need 25 tracks to wake up and get me moving";
  const q = parseSearchToolCall(
    {
      semantic_text: "high-energy upbeat music with driving rhythms",
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "strong",
          evidence: "wake up",
        },
        {
          feature: "danceability",
          direction: "higher",
          strength: "normal",
          evidence: "get me moving",
        },
      ],
      count: 25,
    },
    userText,
  );

  expect(q.version).toBe(2);
  expect(q.constraints).toEqual([]);
  expect(q.preferences.map(({ feature }) => feature)).toEqual([
    "energy",
    "danceability",
  ]);
  expect(q.selection).toEqual({ count: 25, artistVariety: "balanced" });
});

test("decomposes 'other bands like X but higher energy'", () => {
  const userText = "Other bands like Mastodon but higher energy";
  const q = parseSearchToolCall(
    {
      seed_artists: ["  Mastodon  "],
      semantic_text: "early sludge metal",
      exclude_seed_artists: true,
      artist_variety: "wide",
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "higher energy",
        },
      ],
    },
    userText,
  );
  expect(q.seedArtists).toEqual(["Mastodon"]);
  expect(q.excludeSeedArtists).toBe(true);
  expect(q.selection.artistVariety).toBe("wide");
  expect(q.preferences[0]?.feature).toBe("energy");
});

test("planner prompt defines grounded exploratory searches", () => {
  expect(SYSTEM_PROMPT).toContain("exactly one search_tracks tool call");
  expect(SYSTEM_PROMPT).toContain("Subjective qualities always belong in preferences");
  expect(SYSTEM_PROMPT).toContain("Never turn \"wake me up,\"");
  expect(SYSTEM_PROMPT).toContain("Over 140 BPM");
});

test("tool schema requires the feature on percentile constraints", () => {
  const schema = JSON.stringify(SEARCH_TRACKS_TOOL);
  expect(schema).toContain("feature_percentile_constraints");
  expect(schema).toContain('"required":["feature","operator","value","evidence"]');
  expect(schema).toContain('"required":["value","evidence"]');
});

test("accepts an evidenced BPM constraint and drops an invented one", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "fast metal",
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "over 140 BPM" },
        { operator: "max", value: 180, evidence: "fast metal" },
      ],
    },
    "Metal over 140 BPM",
  );

  expect(q.constraints).toEqual([
    { feature: "tempoBpm", operator: "min", value: 140, evidence: "over 140 BPM" },
  ]);
  expect(q.warnings).toContain("Dropped a hard constraint without exact supporting user text.");
});

test("accepts a negated upper BPM bound without reversing it", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "mid-tempo metal",
      tempo_constraints: [
        {
          operator: "max",
          value: 140,
          evidence: "do not go over 140 BPM",
        },
      ],
    },
    "Metal, but do not go over 140 BPM",
  );

  expect(q.constraints).toEqual([
    {
      feature: "tempoBpm",
      operator: "max",
      value: 140,
      evidence: "do not go over 140 BPM",
    },
  ]);
});

test("accepts an explicit top-percentile constraint", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "aggressive metal",
      feature_percentile_constraints: [
        {
          feature: "aggression",
          operator: "min",
          value: 0.8,
          evidence: "top 20% most aggressive",
        },
      ],
    },
    "Give me the top 20% most aggressive metal",
  );

  expect(q.constraints[0]).toEqual({
    feature: "featurePercentile",
    percentileFeature: "aggression",
    operator: "min",
    value: 0.8,
    evidence: "top 20% most aggressive",
  });
});

test("converts legacy subjective thresholds into soft preferences", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "anything",
      energy_min: 5,
      acoustic_max: -2,
      count: 999,
    },
    "something energetic and electronic",
  );

  expect(q.preferences.map(({ feature, direction, strength }) => ({
    feature,
    direction,
    strength,
  }))).toEqual([
    { feature: "energy", direction: "higher", strength: "strong" },
    { feature: "acousticness", direction: "lower", strength: "subtle" },
  ]);
  expect(q.selection.count).toBe(30);
  expect(q.warnings).toContain(
    "Dropped a planner track count not supported by the user's final request.",
  );
});

test("count floors to 1", () => {
  expect(
    parseSearchToolCall({ semantic_text: "x", count: 0 }, "0 tracks").selection.count,
  ).toBe(1);
});

test("rejects a query without semantic_text", () => {
  expect(() => parseSearchToolCall({ seed_artists: ["Tool"] }, "Tool")).toThrow();
});

test("drops empty/blank seed artist names", () => {
  const q = parseSearchToolCall(
    { semantic_text: "progressive metal", seed_artists: ["", "  ", "Tool"] },
    "like Tool",
  );
  expect(q.seedArtists).toEqual(["Tool"]);
});

test("deduplicates seed artists case-insensitively and caps their count", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "progressive metal",
      seed_artists: [
        "Primus",
        " primus ",
        "Mastodon",
        "Igorrr",
        "Tool",
        "Gojira",
        "Opeth",
        "Meshuggah",
        "Periphery",
        "Slipknot",
      ],
    },
    "like Primus, Mastodon, Igorrr, Tool, Gojira, Opeth, Meshuggah, Periphery and Slipknot",
  );

  expect(q.seedArtists).toEqual([
    "Primus",
    "Mastodon",
    "Igorrr",
    "Tool",
    "Gojira",
    "Opeth",
    "Meshuggah",
    "Periphery",
  ]);
});

test("drops seed exclusion when there are no seed artists", () => {
  const q = parseSearchToolCall({
    semantic_text: "technical progressive metal",
    exclude_seed_artists: true,
  }, "technical progressive metal");
  expect(q.excludeSeedArtists).toBeUndefined();
});

test("drops an unknown artist variety", () => {
  const q = parseSearchToolCall({
    semantic_text: "metal",
    artist_variety: "maximum",
  }, "metal");
  expect(q.selection.artistVariety).toBe("balanced");
});

test("rejects contradictory BPM constraints", () => {
  expect(() =>
    parseSearchToolCall(
      {
        semantic_text: "metal",
        tempo_constraints: [
          { operator: "min", value: 160, evidence: "over 160 BPM" },
          { operator: "max", value: 100, evidence: "under 100 BPM" },
        ],
      },
      "Metal over 160 BPM but under 100 BPM",
    ),
  ).toThrow("contradictory BPM constraints");
});

test("rejects constraints whose operator reverses the user's wording", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "slow metal",
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "under 140 BPM" },
      ],
      feature_percentile_constraints: [
        {
          feature: "aggression",
          operator: "max",
          value: 0.2,
          evidence: "top 20% most aggressive",
        },
      ],
    },
    "Metal under 140 BPM from the top 20% most aggressive tracks",
  );

  expect(q.constraints).toEqual([]);
});

test("does not reuse numbers from unrelated roles", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      tempo_constraints: [
        { operator: "min", value: 2, evidence: "exactly 2 tracks" },
      ],
      count: 80,
      strict_artist_cap: {
        value: 2,
        evidence: "exactly 2 tracks",
      },
    },
    "Metal over 80 BPM, exactly 2 tracks",
  );

  expect(q.constraints).toEqual([]);
  expect(q.selection).toEqual({ count: 2, artistVariety: "balanced" });
  expect(q.warnings).toContain("Dropped a strict artist cap without explicit per-artist evidence.");
});

test("accepts a nested strict cap only with explicit per-artist evidence", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      strict_artist_cap: {
        value: 2,
        evidence: "at most 2 tracks per artist",
      },
    },
    "Metal, at most 2 tracks per artist",
  );

  expect(q.selection.strictArtistCap).toBe(2);
});

test("a per-artist cap cannot also establish total count", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      count: 2,
      strict_artist_cap: {
        value: 2,
        evidence: "at most 2 tracks per artist",
      },
    },
    "Metal, at most 2 tracks per artist",
  );

  expect(q.selection).toEqual({
    count: 30,
    artistVariety: "balanced",
    strictArtistCap: 2,
  });
});

test("preference evidence must support the requested direction", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "quiet music",
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "lower energy",
        },
        {
          feature: "tempo",
          direction: "higher",
          strength: "normal",
          evidence: "slower",
        },
        {
          feature: "acousticness",
          direction: "lower",
          strength: "normal",
          evidence: "electronic",
        },
        {
          feature: "aggression",
          direction: "higher",
          strength: "normal",
          evidence: "less heavy",
        },
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "low intensity",
        },
      ],
    },
    "Lower energy, slower, electronic music, less heavy, low intensity",
  );

  expect(q.preferences).toEqual([
    {
      feature: "acousticness",
      direction: "lower",
      strength: "normal",
      evidence: "electronic",
    },
  ]);
});

test("seed provenance uses phrase boundaries and exclusion intent names the seed role", () => {
  const substring = parseSearchToolCall(
    { semantic_text: "rock", seed_artists: ["Tool"] },
    "Toolbox-inspired rock",
  );
  const vocals = parseSearchToolCall(
    {
      semantic_text: "progressive metal without harsh vocals",
      seed_artists: ["Mastodon"],
      exclude_seed_artists: true,
    },
    "Mastodon without harsh vocals",
  );

  expect(substring.seedArtists).toBeUndefined();
  expect(vocals.seedArtists).toEqual(["Mastodon"]);
  expect(vocals.excludeSeedArtists).toBeUndefined();
});

test("negated wording cannot activate constraints, preferences, or exclusions", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      seed_artists: ["Mastodon"],
      exclude_seed_artists: true,
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "over 140 BPM" },
      ],
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "energetic",
        },
      ],
    },
    "Mastodon, but do not exclude Mastodon; metal not over 140 BPM and not energetic",
  );
  const otherArtists = parseSearchToolCall(
    {
      semantic_text: "metal",
      seed_artists: ["Mastodon"],
      exclude_seed_artists: true,
    },
    "I do not want other artists; play Mastodon",
  );

  expect(q.constraints).toEqual([]);
  expect(q.preferences).toEqual([]);
  expect(q.excludeSeedArtists).toBeUndefined();
  expect(otherArtists.excludeSeedArtists).toBeUndefined();
});

test("numeric evidence binds each threshold to its local role and feature", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "aggressive metal",
      tempo_constraints: [
        { operator: "min", value: 2, evidence: "2 tracks over 140 BPM" },
        {
          operator: "min",
          value: 2,
          evidence: "between 2 and 4 tracks at 140 BPM",
        },
        {
          operator: "max",
          value: 4,
          evidence: "between 2 and 4 tracks at 140 BPM",
        },
      ],
      feature_percentile_constraints: [
        {
          feature: "aggression",
          operator: "min",
          value: 0.7,
          evidence: "top 20% aggression and top 30% energy",
        },
        {
          feature: "energy",
          operator: "min",
          value: 0.7,
          evidence: "top 20% aggression and top 30% energy",
        },
      ],
    },
    "2 tracks over 140 BPM, between 2 and 4 tracks at 140 BPM, top 20% aggression and top 30% energy",
  );

  expect(q.constraints).toEqual([
    {
      feature: "featurePercentile",
      percentileFeature: "energy",
      operator: "min",
      value: 0.7,
      evidence: "top 20% aggression and top 30% energy",
    },
  ]);
});

test("per-artist and artist counts cannot establish total track count", () => {
  for (const userText of [
    "2 tracks from every artist",
    "2 tracks from each of the artists",
    "music from 5 bands",
    "12 other bands like Mastodon",
  ]) {
    const requested = Number(userText.match(/\d+/)?.[0]);
    const q = parseSearchToolCall(
      { semantic_text: "music", count: requested },
      userText,
    );
    expect(q.selection.count).toBe(30);
  }
});

test("negated count wording cannot establish total track count", () => {
  const q = parseSearchToolCall(
    { semantic_text: "music", count: 20 },
    "Not 20 tracks, give me 10 tracks",
  );
  const longNegation = parseSearchToolCall(
    {
      semantic_text: "metal",
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "over 140 BPM" },
      ],
    },
    "Metal that must not include anything that is over 140 BPM",
  );

  expect(q.selection.count).toBe(10);
  expect(longNegation.constraints).toEqual([]);
});

test("parser recovers omitted explicit counts and preserves counts above 100", () => {
  const omitted = parseSearchToolCall(
    { semantic_text: "metal" },
    "Give me 10 tracks",
  );
  const large = parseSearchToolCall(
    { semantic_text: "metal", count: 150 },
    "Give me 150 tracks",
  );

  expect(omitted.selection.count).toBe(10);
  expect(omitted.warnings).toContain(
    "Recovered an explicit track count omitted by the planner.",
  );
  expect(large.selection.count).toBe(150);
});

test("the last explicit count wins when the user revises cardinality", () => {
  const q = parseSearchToolCall(
    { semantic_text: "metal", count: 20 },
    "Give me 20 tracks, actually make it 10 tracks",
  );

  expect(q.selection.count).toBe(10);
  expect(q.warnings).toContain(
    "Dropped a planner track count not supported by the user's final request.",
  );
});

test("exact per-artist distribution is not weakened into a maximum", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      strict_artist_cap: {
        value: 2,
        evidence: "exactly 2 tracks per artist",
      },
    },
    "Give me exactly 2 tracks per artist",
  );

  expect(q.selection.strictArtistCap).toBeUndefined();
  expect(q.warnings).toContain(
    "Dropped a strict artist cap without explicit per-artist evidence.",
  );
});

test("trailing and coordinated negation cannot authorize positive policies", () => {
  const trailing = parseSearchToolCall(
    {
      semantic_text: "metal",
      count: 20,
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "Over 140 BPM" },
      ],
    },
    "Over 140 BPM and 20 tracks are not what I want",
  );
  const coordinated = parseSearchToolCall(
    {
      semantic_text: "quiet music",
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "energetic",
        },
      ],
    },
    "I do not want aggressive and energetic music",
  );

  expect(trailing.constraints).toEqual([]);
  expect(trailing.selection.count).toBe(30);
  expect(coordinated.preferences).toEqual([]);
});

test("inclusive requests for a seed and other bands do not exclude the seed", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "progressive metal",
      seed_artists: ["Mastodon"],
      exclude_seed_artists: true,
    },
    "Mastodon and other bands like it",
  );

  expect(q.seedArtists).toEqual(["Mastodon"]);
  expect(q.excludeSeedArtists).toBeUndefined();
});

test("avoidance and anything-but wording cannot authorize positive preferences or bounds", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "music",
      tempo_constraints: [
        { operator: "min", value: 140, evidence: "over 140 BPM" },
      ],
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "normal",
          evidence: "energetic",
        },
      ],
    },
    "Anything but metal over 140 BPM; avoid energetic music",
  );

  expect(q.constraints).toEqual([]);
  expect(q.preferences).toEqual([]);
});

test("drops invented seeds, preferences, counts, and strict caps", () => {
  const q = parseSearchToolCall(
    {
      semantic_text: "metal",
      seed_artists: ["Ghost"],
      preferences: [
        {
          feature: "energy",
          direction: "higher",
          strength: "strong",
          evidence: "metal",
        },
      ],
      count: 100,
      strict_artist_cap: 2,
      strict_artist_cap_evidence: "metal",
    },
    "play metal",
  );

  expect(q.seedArtists).toBeUndefined();
  expect(q.preferences).toEqual([]);
  expect(q.selection).toEqual({ count: 30, artistVariety: "balanced" });
});

test("rejects contradictory percentile constraints for the same feature", () => {
  expect(() =>
    parseSearchToolCall(
      {
        semantic_text: "aggressive metal",
        feature_percentile_constraints: [
          {
            feature: "aggression",
            operator: "min",
            value: 0.8,
            evidence: "top 20% most aggressive",
          },
          {
            feature: "aggression",
            operator: "max",
            value: 0.2,
            evidence: "below 20% aggression",
          },
        ],
      },
      "Use the top 20% most aggressive but below 20% aggression",
    ),
  ).toThrow("contradictory aggression constraints");
});
