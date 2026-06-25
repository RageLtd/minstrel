import { test, expect } from "bun:test";
import { parseSearchToolCall } from "./tools";

test("normalises a full vibe query", () => {
  const q = parseSearchToolCall({
    semantic_text: "  sludgy doom metal  ",
    bpm_min: 60,
    bpm_max: 90,
    energy_min: 0.7,
    count: 25,
  });
  expect(q.semanticText).toBe("sludgy doom metal");
  expect(q.filters).toEqual({ bpmMin: 60, bpmMax: 90, energyMin: 0.7 });
  expect(q.count).toBe(25);
  expect(q.seedArtists).toBeUndefined();
});

test("decomposes 'like X but higher energy'", () => {
  const q = parseSearchToolCall({
    seed_artists: ["Mastodon"],
    semantic_text: "early sludge metal",
    energy_min: 0.8,
  });
  expect(q.seedArtists).toEqual(["Mastodon"]);
  expect(q.filters.energyMin).toBe(0.8);
});

test("clamps unit ranges and count, ignores junk types", () => {
  const q = parseSearchToolCall({
    semantic_text: "anything",
    energy_min: 5, // -> 1
    acoustic_max: -2, // -> 0
    bpm_min: "fast", // -> dropped
    count: 999, // -> 100
  });
  expect(q.filters.energyMin).toBe(1);
  expect(q.filters.acousticMax).toBe(0);
  expect(q.filters.bpmMin).toBeUndefined();
  expect(q.count).toBe(100);
});

test("count floors to 1", () => {
  expect(parseSearchToolCall({ semantic_text: "x", count: 0 }).count).toBe(1);
});

test("rejects a query with neither semantic_text nor seed_artists", () => {
  expect(() => parseSearchToolCall({ energy_min: 0.5 })).toThrow();
});

test("drops empty/blank seed artist names", () => {
  const q = parseSearchToolCall({ seed_artists: ["", "  ", "Tool"] });
  expect(q.seedArtists).toEqual(["Tool"]);
});
