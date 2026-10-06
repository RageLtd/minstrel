import type { Database } from "bun:sqlite";
import {
  artistEmbeddings,
  audioFeatureDataReady,
  featureDistributions,
  searchTracks,
  searchableTrackCount,
  type PreferenceFeature,
  type SearchHit,
} from "./repo";
import type { EmbedText } from "./embedder";
import {
  DECISION_OVERFETCH,
  makeMembershipGate,
  type MembershipGate,
} from "./membership";
import {
  rerankByPreferences,
  satisfiesPercentileConstraints,
  satisfiesTempoConstraints,
  type BaseCandidate,
} from "./search-rank";
import type {
  FeaturePercentileConstraint,
  SearchQuery,
  TempoConstraint,
} from "./tools";
import { meanNormalize } from "./vec";

export type { BaseCandidate } from "./search-rank";

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
  /** Candidates the decision model judged; 0 when no gate is configured. */
  classifierEvaluated: number;
  classifierRejected: number;
  warnings: string[];
  shortfallReason?:
    | "constraints"
    | "exclusions"
    | "classifier"
    | "corpus"
    | "strictArtistCap";
}

interface RankedCandidate {
  hit: SearchHit;
  bestSeedRank?: number;
  semanticRank?: number;
  insertionOrder: number;
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

const DERIVED_FEATURES = new Set<PreferenceFeature>([
  "novelty",
  "rhythmicIrregularity",
  "timbralComplexity",
  "dynamicContrast",
  "harmonicInstability",
]);

/**
 * Turn a validated SearchQuery into ranked tracks. Searches each seed-artist
 * centroid and the CLAP-embedded semantic text independently, fuses those
 * neighborhoods, then applies validated constraints, percentile preferences,
 * adaptive retrieval, and artist diversity with cardinality backfill.
 * Missing seed artists are reported and fall through to the semantic text.
 * With a membership gate, a decision model then judges the top candidates and
 * only tracks it accepts reach the final selection.
 */
export async function executeSearch(
  db: Database,
  embedText: EmbedText,
  query: SearchQuery,
  gate?: MembershipGate,
) {
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
    const nothingToSearch: SearchResult = {
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
        classifierEvaluated: 0,
        classifierRejected: 0,
        warnings,
        shortfallReason: "corpus",
      },
    };
    return nothingToSearch;
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
    const emptyLibrary: SearchResult = {
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
        classifierEvaluated: 0,
        classifierRejected: 0,
        warnings,
        shortfallReason: "corpus",
      },
    };
    return emptyLibrary;
  }

  let neighborCount = Math.min(
    searchableTracks,
    Math.max(requested * 4, 128),
  );
  let retrievalPasses = 0;
  let finalCandidates: BaseCandidate[] = [];
  let finalAfterExclusions: BaseCandidate[] = [];
  let finalEligible: BaseCandidate[] = [];
  let finalAdmitted: SearchHit[] = [];
  let finalSelection: DiversityResult = { hits: [], backfilled: 0 };
  const membership = gate ? makeMembershipGate(db, gate) : undefined;
  let classifierEvaluated = 0;
  let classifierRejected = 0;

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
    finalAdmitted = ranked;
    if (membership) {
      const gated = await membership(ranked, requested);
      finalAdmitted = gated.kept;
      classifierEvaluated = gated.evaluated;
      classifierRejected = gated.rejected;
    }
    finalSelection = selectWithArtistVariety(
      finalAdmitted,
      requested,
      query.selection.artistVariety,
      query.selection.strictArtistCap,
    );
    // With a gate, a full over-fetch window is the most the model will judge;
    // wider retrieval cannot admit more, so stop rather than re-judge.
    const gateWindowFull =
      membership !== undefined && ranked.length >= requested * DECISION_OVERFETCH;
    if (
      finalSelection.hits.length >= requested ||
      neighborCount >= searchableTracks ||
      gateWindowFull
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
    } else if (classifierRejected > 0 && finalAdmitted.length < requested) {
      shortfallReason = "classifier";
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
      classifierEvaluated,
      classifierRejected,
      warnings,
      shortfallReason,
    },
  };
}
