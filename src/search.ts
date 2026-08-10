import type { Database } from "bun:sqlite";
import {
  artistEmbeddings,
  audioFeatureDataReady,
  featureDistributions,
  searchTracks,
  searchableTrackCount,
  type FeatureDistributions,
  type PreferenceFeature,
  type SearchHit,
} from "./repo";
import type { EmbedText } from "./embedder";
import type {
  FeaturePercentileConstraint,
  SearchConstraint,
  SearchPreference,
  SearchQuery,
  TempoConstraint,
} from "./tools";
import { meanNormalize } from "./vec";

export interface SearchResult {
  hits: SearchHit[];
  /** Seed artists the user named that aren't in the library (for the reply). */
  missingSeedArtists: string[];
  diagnostics: SearchDiagnostics;
}

export interface SearchDiagnostics {
  requested: number;
  returned: number;
  searchableTracks: number;
  retrievalPasses: number;
  retrievedCandidates: number;
  eligibleCandidates: number;
  preferenceReranked: boolean;
  diversityBackfilled: number;
  exhaustedCorpus: boolean;
  warnings: string[];
  shortfallReason?: "constraints" | "exclusions" | "corpus" | "strictArtistCap";
}

interface RankedCandidate {
  hit: SearchHit;
  bestSeedRank?: number;
  semanticRank?: number;
  insertionOrder: number;
}

export interface BaseCandidate {
  hit: SearchHit;
  baseScore: number;
}

const SEMANTIC_TIEBREAK_WEIGHT = 0.5;

const VARIETY = {
  focused: 6,
  balanced: 2,
  wide: 1,
} as const;

export interface DiversityResult {
  hits: SearchHit[];
  backfilled: number;
}

export function selectWithArtistVariety(
  candidates: SearchHit[],
  count: number,
  variety: SearchQuery["selection"]["artistVariety"] = "balanced",
  strictArtistCap?: number,
): DiversityResult {
  const artistCounts = new Map<string, number>();
  const selected: SearchHit[] = [];
  const selectedIds = new Set<number>();
  const skipped: SearchHit[] = [];
  const maximumArtistCap = strictArtistCap ?? Number.POSITIVE_INFINITY;
  const firstPassCap = Math.min(VARIETY[variety], maximumArtistCap);
  const artistKey = (hit: SearchHit) =>
    hit.artist?.trim().toLocaleLowerCase() || `unknown:${hit.id}`;

  for (const hit of candidates) {
    const key = artistKey(hit);
    const artistCount = artistCounts.get(key) ?? 0;
    if (artistCount >= firstPassCap) {
      skipped.push(hit);
      continue;
    }
    selected.push(hit);
    selectedIds.add(hit.id);
    artistCounts.set(key, artistCount + 1);
    if (selected.length === count) return { hits: selected, backfilled: 0 };
  }

  let backfilled = 0;
  for (const hit of skipped) {
    if (selectedIds.has(hit.id)) continue;
    const key = artistKey(hit);
    const artistCount = artistCounts.get(key) ?? 0;
    if (artistCount >= maximumArtistCap) continue;
    selected.push(hit);
    selectedIds.add(hit.id);
    artistCounts.set(key, artistCount + 1);
    backfilled += 1;
    if (selected.length === count) break;
  }
  return { hits: selected, backfilled };
}

export function fuseSearches(
  seedResults: SearchHit[][],
  semanticResults: SearchHit[] | undefined,
  candidateCount: number,
): BaseCandidate[] {
  const candidates = new Map<number, RankedCandidate>();
  let insertionOrder = 0;
  const candidate = (hit: SearchHit): RankedCandidate => {
    const found = candidates.get(hit.id);
    if (found) return found;
    const created = { hit, insertionOrder: insertionOrder++ };
    candidates.set(hit.id, created);
    return created;
  };

  for (const results of seedResults) {
    results.forEach((hit, index) => {
      const ranked = candidate(hit);
      const rank = index + 1;
      ranked.bestSeedRank = Math.min(ranked.bestSeedRank ?? rank, rank);
      if (hit.distance < ranked.hit.distance) ranked.hit = hit;
    });
  }
  semanticResults?.forEach((hit, index) => {
    const ranked = candidate(hit);
    ranked.semanticRank = index + 1;
    if (seedResults.length === 0 || hit.distance < ranked.hit.distance) {
      ranked.hit = hit;
    }
  });

  const missingRank = candidateCount * 2;
  return [...candidates.values()]
    .map((candidate) => {
      const seedRank = candidate.bestSeedRank ?? missingRank;
      const baseScore = seedResults.length
        ? seedRank +
          SEMANTIC_TIEBREAK_WEIGHT *
            ((candidate.semanticRank ?? candidateCount + 1) / (candidateCount + 1))
        : (candidate.semanticRank ?? missingRank);
      return { candidate, baseScore };
    })
    .sort((a, b) => {
      const scoreDifference = a.baseScore - b.baseScore;
      if (scoreDifference !== 0) return scoreDifference;
      return a.candidate.insertionOrder - b.candidate.insertionOrder;
    })
    .map(({ candidate, baseScore }) => ({ hit: candidate.hit, baseScore }));
}

export function percentileRank(sorted: number[], value: number): number | undefined {
  if (sorted.length === 0) return undefined;
  if (sorted.length === 1) return 0.5;
  let lower = 0;
  let upper = sorted.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (sorted[middle]! < value) lower = middle + 1;
    else upper = middle;
  }
  const first = lower;
  upper = sorted.length;
  while (lower < upper) {
    const middle = Math.floor((lower + upper) / 2);
    if (sorted[middle]! <= value) lower = middle + 1;
    else upper = middle;
  }
  const last = lower - 1;
  const averageRank = first <= last ? (first + last) / 2 : first;
  return Math.min(1, Math.max(0, averageRank / (sorted.length - 1)));
}

const STRENGTH_WEIGHT = {
  subtle: 0.35,
  normal: 0.65,
  strong: 1,
} as const;

const FEATURE_GROUP: Record<PreferenceFeature, string> = {
  tempo: "motion",
  energy: "motion",
  aggression: "motion",
  danceability: "motion",
  acousticness: "production",
  novelty: "novelty",
  rhythmicIrregularity: "rhythm",
  timbralComplexity: "texture",
  dynamicContrast: "texture",
  harmonicInstability: "texture",
};

function preferenceFit(
  hit: SearchHit,
  preferences: SearchPreference[],
  distributions: FeatureDistributions,
): number {
  const groups = new Map<
    string,
    { weightedFit: number; totalWeight: number; groupWeight: number }
  >();
  for (const preference of preferences) {
    const value = hit.featureValues[preference.feature];
    if (value === undefined) continue;
    const percentile = percentileRank(distributions[preference.feature], value);
    if (percentile === undefined) continue;
    const fit = preference.direction === "higher" ? percentile : 1 - percentile;
    const weight = STRENGTH_WEIGHT[preference.strength];
    const group = FEATURE_GROUP[preference.feature];
    const aggregate = groups.get(group) ?? {
      weightedFit: 0,
      totalWeight: 0,
      groupWeight: 0,
    };
    aggregate.weightedFit += fit * weight;
    aggregate.totalWeight += weight;
    aggregate.groupWeight = Math.max(aggregate.groupWeight, weight);
    groups.set(group, aggregate);
  }
  if (groups.size === 0) return 0.5;
  let weightedFit = 0;
  let totalWeight = 0;
  for (const aggregate of groups.values()) {
    weightedFit +=
      (aggregate.weightedFit / aggregate.totalWeight) * aggregate.groupWeight;
    totalWeight += aggregate.groupWeight;
  }
  return weightedFit / totalWeight;
}

export function rerankByPreferences(
  candidates: BaseCandidate[],
  preferences: SearchPreference[],
  distributions: FeatureDistributions,
  requestedCount: number,
): SearchHit[] {
  if (preferences.length === 0 || candidates.length < 2) {
    return candidates.map(({ hit }) => hit);
  }
  const lastRank = candidates.length - 1;
  const boundaryRank = Math.min(Math.max(requestedCount - 1, 0), lastRank);
  const outerRank = Math.min(Math.max(requestedCount * 2 - 1, 0), lastRank);
  const relevanceGap = Math.max(
    0,
    candidates[outerRank]!.baseScore - candidates[boundaryRank]!.baseScore,
  );
  const strength = Math.max(
    ...preferences.map((preference) => STRENGTH_WEIGHT[preference.strength]),
  );
  const preferenceBudget = strength * relevanceGap;
  return candidates
    .map((candidate, index) => {
      const score =
        candidate.baseScore +
        preferenceBudget *
          (1 - preferenceFit(candidate.hit, preferences, distributions));
      return { hit: candidate.hit, score, index };
    })
    .sort((a, b) => a.score - b.score || a.index - b.index)
    .map(({ hit }) => hit);
}

const DERIVED_FEATURES = new Set<PreferenceFeature>([
  "novelty",
  "rhythmicIrregularity",
  "timbralComplexity",
  "dynamicContrast",
  "harmonicInstability",
]);

function satisfiesTempoConstraints(
  hit: SearchHit,
  constraints: TempoConstraint[],
): boolean {
  const tempo = hit.featureValues.tempo;
  if (constraints.length > 0 && tempo === undefined) return false;
  return constraints.every((constraint) =>
    constraint.operator === "min"
      ? tempo! >= constraint.value
      : tempo! <= constraint.value,
  );
}

function satisfiesPercentileConstraints(
  hit: SearchHit,
  constraints: FeaturePercentileConstraint[],
  distributions: FeatureDistributions,
): boolean {
  return constraints.every((constraint) => {
    const value = hit.featureValues[constraint.percentileFeature];
    if (value === undefined) return false;
    const percentile = percentileRank(
      distributions[constraint.percentileFeature],
      value,
    );
    if (percentile === undefined) return false;
    return constraint.operator === "min"
      ? percentile >= constraint.value
      : percentile <= constraint.value;
  });
}

/**
 * Turn a validated SearchQuery into ranked tracks. Searches each seed-artist
 * centroid and the CLAP-embedded semantic text independently, fuses those
 * neighborhoods, then applies validated constraints, percentile preferences,
 * adaptive retrieval, and artist diversity with cardinality backfill.
 * Missing seed artists are reported and fall through to the semantic text.
 */
export async function executeSearch(
  db: Database,
  embedText: EmbedText,
  query: SearchQuery,
): Promise<SearchResult> {
  const seedQueries: Float32Array[] = [];
  const missingSeedArtists: string[] = [];

  if (query.seedArtists) {
    for (const artist of query.seedArtists) {
      const found = artistEmbeddings(db, artist);
      if (found.length === 0) missingSeedArtists.push(artist);
      else seedQueries.push(meanNormalize(found));
    }
  }

  const semanticQuery = query.semanticText
    ? await embedText(query.semanticText)
    : undefined;

  const requested = query.selection.count;
  const searchableTracks = searchableTrackCount(db);
  const warnings = [...query.warnings];
  if (seedQueries.length === 0 && !semanticQuery) {
    return {
      hits: [],
      missingSeedArtists,
      diagnostics: {
        requested,
        returned: 0,
        searchableTracks,
        retrievalPasses: 0,
        retrievedCandidates: 0,
        eligibleCandidates: 0,
        preferenceReranked: false,
        diversityBackfilled: 0,
        exhaustedCorpus: true,
        warnings,
        shortfallReason: "corpus",
      },
    };
  }

  const excludedArtists = new Set(
    (query.excludeSeedArtists ? (query.seedArtists ?? []) : []).map((artist) =>
      artist.trim().toLocaleLowerCase(),
    ),
  );
  const tempoConstraints = query.constraints.filter(
    (constraint): constraint is TempoConstraint => constraint.feature === "tempoBpm",
  );
  const percentileConstraints = query.constraints.filter(
    (constraint): constraint is FeaturePercentileConstraint =>
      constraint.feature === "featurePercentile",
  );
  const needsDerivedFeatures = [...query.preferences, ...percentileConstraints].some(
    (item) =>
      DERIVED_FEATURES.has(
        "percentileFeature" in item ? item.percentileFeature : item.feature,
      ),
  );
  const derivedFeaturesReady = audioFeatureDataReady(db);
  const activePreferences = query.preferences.filter(
    (preference) => !DERIVED_FEATURES.has(preference.feature) || derivedFeaturesReady,
  );
  if (needsDerivedFeatures && !derivedFeaturesReady) {
    warnings.push("Derived audio preferences are unavailable until analysis completes.");
  }
  const hasUnavailableConstraint = percentileConstraints.some(
    (constraint) =>
      DERIVED_FEATURES.has(constraint.percentileFeature) && !derivedFeaturesReady,
  );
  const includeFeatureValues =
    activePreferences.length > 0 || query.constraints.length > 0;
  const distributions = includeFeatureValues
    ? featureDistributions(db)
    : undefined;

  if (searchableTracks === 0) {
    return {
      hits: [],
      missingSeedArtists,
      diagnostics: {
        requested,
        returned: 0,
        searchableTracks: 0,
        retrievalPasses: 0,
        retrievedCandidates: 0,
        eligibleCandidates: 0,
        preferenceReranked: false,
        diversityBackfilled: 0,
        exhaustedCorpus: true,
        warnings,
        shortfallReason: "corpus",
      },
    };
  }

  let neighborCount = Math.min(
    searchableTracks,
    Math.max(requested * 4, 128),
  );
  let retrievalPasses = 0;
  let finalCandidates: BaseCandidate[] = [];
  let finalAfterExclusions: BaseCandidate[] = [];
  let finalEligible: BaseCandidate[] = [];
  let finalSelection: DiversityResult = { hits: [], backfilled: 0 };

  while (true) {
    retrievalPasses += 1;
    const search = (vector: Float32Array) =>
      searchTracks(db, vector, neighborCount, includeFeatureValues);
    const seedResults = seedQueries.map(search);
    const semanticResults = semanticQuery ? search(semanticQuery) : undefined;
    finalCandidates = fuseSearches(seedResults, semanticResults, neighborCount);
    finalAfterExclusions = finalCandidates.filter(({ hit }) => {
      const artist = hit.artist?.trim().toLocaleLowerCase();
      return !artist || !excludedArtists.has(artist);
    });
    finalEligible = finalAfterExclusions.filter(({ hit }) => {
      if (!satisfiesTempoConstraints(hit, tempoConstraints)) return false;
      if (hasUnavailableConstraint) return false;
      if (!distributions) return percentileConstraints.length === 0;
      return satisfiesPercentileConstraints(
        hit,
        percentileConstraints,
        distributions,
      );
    });
    const ranked = distributions && activePreferences.length > 0
      ? rerankByPreferences(
          finalEligible,
          activePreferences,
          distributions,
          requested,
        )
      : finalEligible.map(({ hit }) => hit);
    finalSelection = selectWithArtistVariety(
      ranked,
      requested,
      query.selection.artistVariety,
      query.selection.strictArtistCap,
    );
    if (
      finalSelection.hits.length >= requested ||
      neighborCount >= searchableTracks
    ) {
      break;
    }
    neighborCount = Math.min(searchableTracks, neighborCount * 2);
  }

  const exhaustedCorpus = neighborCount >= searchableTracks;
  const rerankedPreferences = activePreferences.filter(
    (preference) =>
      (distributions?.[preference.feature].length ?? 0) > 0 &&
      finalCandidates.some(
        ({ hit }) => hit.featureValues[preference.feature] !== undefined,
      ),
  );
  for (const preference of activePreferences) {
    if (!rerankedPreferences.includes(preference)) {
      warnings.push(`No usable ${preference.feature} values were available for reranking.`);
    } else if (
      finalCandidates.some(
        ({ hit }) => hit.featureValues[preference.feature] === undefined,
      )
    ) {
      warnings.push(`Some candidates lacked ${preference.feature} values.`);
    }
  }
  let shortfallReason: SearchDiagnostics["shortfallReason"];
  if (finalSelection.hits.length < requested) {
    if (
      query.constraints.length > 0 &&
      finalEligible.length < finalAfterExclusions.length
    ) {
      shortfallReason = "constraints";
    } else if (
      excludedArtists.size > 0 &&
      finalAfterExclusions.length < finalCandidates.length
    ) {
      shortfallReason = "exclusions";
    } else if (
      query.selection.strictArtistCap !== undefined &&
      finalSelection.hits.length < finalEligible.length
    ) {
      shortfallReason = "strictArtistCap";
    } else {
      shortfallReason = "corpus";
    }
  }
  return {
    hits: finalSelection.hits,
    missingSeedArtists,
    diagnostics: {
      requested,
      returned: finalSelection.hits.length,
      searchableTracks,
      retrievalPasses,
      retrievedCandidates: finalCandidates.length,
      eligibleCandidates: finalEligible.length,
      preferenceReranked: rerankedPreferences.length > 0,
      diversityBackfilled: finalSelection.backfilled,
      exhaustedCorpus,
      warnings,
      shortfallReason,
    },
  };
}
