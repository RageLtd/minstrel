import { expect, test } from "bun:test";
import {
  describeSearchDiagnostics,
  describeSearchQuery,
} from "./query-display";

test("describes the normalized query in human-readable terms", () => {
  expect(
    describeSearchQuery({
      version: 2,
      semanticText: "technical progressive metal, heavier and more aggressive",
      seedArtists: ["Eidola"],
      excludeSeedArtists: true,
      constraints: [
        { feature: "tempoBpm", operator: "min", value: 120, evidence: "over 120 BPM" },
      ],
      preferences: [
        {
          feature: "aggression",
          direction: "higher",
          strength: "strong",
          evidence: "more aggressive",
        },
        {
          feature: "novelty",
          direction: "higher",
          strength: "normal",
          evidence: "technical",
        },
      ],
      selection: { count: 25, artistVariety: "wide" },
      warnings: [],
    }),
  ).toEqual([
    {
      label: "Semantic sound",
      value: "technical progressive metal, heavier and more aggressive",
    },
    { label: "Seed artists", value: "Eidola" },
    { label: "Seed tracks", value: "Excluded" },
    {
      label: "Artist variety",
      value: "Wide — favor 1 track per artist before backfill",
    },
    { label: "Hard constraint", value: "at least 120 BPM" },
    { label: "Aggression preference", value: "strong, higher" },
    { label: "Corpus novelty preference", value: "normal, higher" },
    { label: "Requested tracks", value: "25" },
  ]);
});

test("omits absent seeds and filters", () => {
  expect(
    describeSearchQuery({
      version: 2,
      semanticText: "quiet ambient drone",
      constraints: [],
      preferences: [],
      selection: { count: 30, artistVariety: "balanced" },
      warnings: [],
    }),
  ).toEqual([
    { label: "Semantic sound", value: "quiet ambient drone" },
    {
      label: "Artist variety",
      value: "Balanced — favor up to 2 tracks per artist before backfill",
    },
    { label: "Requested tracks", value: "30" },
  ]);
});

test("describes execution diagnostics and warnings", () => {
  expect(
    describeSearchDiagnostics({
      requested: 30,
      returned: 4,
      searchableTracks: 2386,
      retrievalPasses: 3,
      retrievedCandidates: 2386,
      eligibleCandidates: 4,
      preferenceReranked: false,
      diversityBackfilled: 0,
      exhaustedCorpus: true,
      classifierEvaluated: 12,
      classifierRejected: 8,
      warnings: ["Dropped an invented BPM constraint."],
      shortfallReason: "constraints",
    }),
  ).toContainEqual({ label: "Result count", value: "4 of 30" });
});

test("reports the membership decision outcome", () => {
  const base = {
    requested: 4,
    returned: 4,
    searchableTracks: 10,
    retrievalPasses: 1,
    retrievedCandidates: 10,
    eligibleCandidates: 10,
    preferenceReranked: false,
    diversityBackfilled: 0,
    exhaustedCorpus: false,
    warnings: [],
  };
  expect(
    describeSearchDiagnostics({ ...base, classifierEvaluated: 12, classifierRejected: 8 }),
  ).toContainEqual({ label: "Membership decision", value: "12 judged, 8 rejected" });
  expect(
    describeSearchDiagnostics({ ...base, classifierEvaluated: 0, classifierRejected: 0 }),
  ).toContainEqual({ label: "Membership decision", value: "Not applied" });
});
