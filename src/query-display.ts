import type { SearchDiagnostics } from "./search";
import type { SearchQuery } from "./tools";

export interface QueryDetail {
  label: string;
  value: string;
}

const FEATURE_LABELS = {
  tempo: "Tempo",
  energy: "Energy",
  aggression: "Aggression",
  danceability: "Danceability",
  acousticness: "Acoustic character",
  novelty: "Corpus novelty",
  rhythmicIrregularity: "Rhythmic irregularity",
  timbralComplexity: "Timbral complexity",
  dynamicContrast: "Dynamic contrast",
  harmonicInstability: "Harmonic instability",
} as const;

const VARIETY_LABELS = {
  focused: "Focused — favor up to 6 tracks per artist before backfill",
  balanced: "Balanced — favor up to 2 tracks per artist before backfill",
  wide: "Wide — favor 1 track per artist before backfill",
} as const;

export function describeSearchQuery(query: SearchQuery): QueryDetail[] {
  const details: QueryDetail[] = [
    { label: "Semantic sound", value: query.semanticText ?? "None" },
  ];

  if (query.seedArtists?.length) {
    details.push({ label: "Seed artists", value: query.seedArtists.join(", ") });
    details.push({
      label: "Seed tracks",
      value: query.excludeSeedArtists ? "Excluded" : "Included",
    });
  }

  details.push({
    label: "Artist variety",
    value: VARIETY_LABELS[query.selection.artistVariety],
  });

  if (query.selection.strictArtistCap !== undefined) {
    details.push({
      label: "Strict artist cap",
      value: `${query.selection.strictArtistCap} tracks per artist`,
    });
  }

  for (const constraint of query.constraints) {
    const bound = constraint.operator === "min" ? "at least" : "at most";
    const value =
      constraint.feature === "tempoBpm"
        ? `${constraint.value} BPM`
        : `${Math.round(constraint.value * 100)}th percentile ${FEATURE_LABELS[constraint.percentileFeature].toLocaleLowerCase()}`;
    details.push({ label: "Hard constraint", value: `${bound} ${value}` });
  }

  for (const preference of query.preferences) {
    details.push({
      label: `${FEATURE_LABELS[preference.feature]} preference`,
      value: `${preference.strength}, ${preference.direction}`,
    });
  }

  details.push({ label: "Requested tracks", value: String(query.selection.count) });
  return details;
}

export function describeSearchDiagnostics(
  diagnostics: SearchDiagnostics,
): QueryDetail[] {
  const details: QueryDetail[] = [
    {
      label: "Result count",
      value: `${diagnostics.returned} of ${diagnostics.requested}`,
    },
    { label: "Searchable tracks", value: String(diagnostics.searchableTracks) },
    { label: "Retrieval passes", value: String(diagnostics.retrievalPasses) },
    {
      label: "Eligible candidates",
      value: `${diagnostics.eligibleCandidates} of ${diagnostics.retrievedCandidates} retrieved`,
    },
    {
      label: "Preference reranking",
      value: diagnostics.preferenceReranked ? "Applied" : "Not requested",
    },
    {
      label: "Diversity backfill",
      value: `${diagnostics.diversityBackfilled} tracks`,
    },
    {
      label: "Membership decision",
      value:
        diagnostics.classifierEvaluated > 0
          ? `${diagnostics.classifierEvaluated} judged, ${diagnostics.classifierRejected} rejected`
          : "Not applied",
    },
  ];
  if (diagnostics.shortfallReason) {
    details.push({ label: "Shortfall reason", value: diagnostics.shortfallReason });
  }
  for (const warning of diagnostics.warnings) {
    details.push({ label: "Warning", value: warning });
  }
  return details;
}
