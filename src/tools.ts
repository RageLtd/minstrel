import type { SearchFilters } from "./repo";
import type { OllamaTool } from "./ollama";

// The language→query boundary: the LLM only ever emits a search_tracks call,
// and parseSearchToolCall validates/normalises its arguments into a typed query.
// The model never sees the database or invents track titles.

export interface SearchQuery {
  semanticText?: string;
  seedArtists?: string[];
  filters: SearchFilters;
  count: number;
}

export const SEARCH_TRACKS_TOOL: OllamaTool = {
  type: "function",
  function: {
    name: "search_tracks",
    description:
      "Find tracks in the user's music library by sonic similarity and audio " +
      "features. Call this for every request to build or change a playlist by vibe.",
    parameters: {
      type: "object",
      properties: {
        semantic_text: {
          type: "string",
          description:
            "Free-text description of the desired sound, mood, instrumentation and " +
            "genre, e.g. 'sludgy doom metal, slow and crushing'. Matched against the " +
            "audio itself. Provide this for nearly every query.",
        },
        seed_artists: {
          type: "array",
          items: { type: "string" },
          description:
            "Real artist names the user wants the result to sound like. Anchors the " +
            "'sounds like X' half of a request.",
        },
        bpm_min: { type: "number", description: "Minimum tempo in BPM." },
        bpm_max: { type: "number", description: "Maximum tempo in BPM." },
        energy_min: {
          type: "number",
          description: "0..1 energy floor. Raise for 'higher energy' / 'more intense'.",
        },
        aggressive_min: { type: "number", description: "0..1 aggressiveness floor." },
        danceable_min: { type: "number", description: "0..1 danceability floor." },
        acoustic_max: {
          type: "number",
          description: "0..1 acoustic ceiling. Lower it for 'more electronic'.",
        },
        count: { type: "integer", description: "How many tracks to return (default 30)." },
      },
    },
  },
};

export const SYSTEM_PROMPT = `You are Minstrel's music librarian. The user describes a vibe; you translate it into a single search_tracks call. You do not have access to the library and must never invent song, album, or artist names — your only job is to express the request as search parameters.

Rules:
- Always provide semantic_text capturing the desired sound, mood, instrumentation and genre.
- If the user names artists to sound like, pass the real ones in seed_artists.
- Decompose "like X but more Y": the "like X" half goes to seed_artists/semantic_text; the "more Y" modifier sets a feature bound — "higher energy" -> energy_min, "more aggressive" -> aggressive_min, "more electronic" -> a low acoustic_max, "faster" -> bpm_min.
- Feature values are 0..1. Be moderate (around 0.6-0.8 for "more X"), not extreme.
- Always call search_tracks first. Never reply in prose before the search has run.
- After a search_tracks result comes back, the translation is done — reply in plain prose: briefly describe the found tracks for the user, mention any missing seed artists, and do not call the tool again.`;

function finiteNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function unitInterval(v: unknown): number | undefined {
  const n = finiteNumber(v);
  return n === undefined ? undefined : Math.min(1, Math.max(0, n));
}

/** Normalise raw tool-call arguments into a validated SearchQuery. */
export function parseSearchToolCall(args: Record<string, unknown>): SearchQuery {
  const semanticText =
    typeof args.semantic_text === "string" && args.semantic_text.trim()
      ? args.semantic_text.trim()
      : undefined;

  const seedArtists = Array.isArray(args.seed_artists)
    ? args.seed_artists.filter(
        (x): x is string => typeof x === "string" && x.trim().length > 0,
      )
    : undefined;

  const rawFilters: SearchFilters = {
    bpmMin: finiteNumber(args.bpm_min),
    bpmMax: finiteNumber(args.bpm_max),
    energyMin: unitInterval(args.energy_min),
    aggressiveMin: unitInterval(args.aggressive_min),
    danceableMin: unitInterval(args.danceable_min),
    acousticMax: unitInterval(args.acoustic_max),
  };
  const filters = Object.fromEntries(
    Object.entries(rawFilters).filter(([, v]) => v !== undefined),
  ) as SearchFilters;

  const count = Math.min(100, Math.max(1, Math.trunc(finiteNumber(args.count) ?? 30)));

  if (!semanticText && !(seedArtists && seedArtists.length > 0)) {
    throw new Error("search_tracks needs at least semantic_text or seed_artists");
  }

  return {
    semanticText,
    seedArtists: seedArtists && seedArtists.length > 0 ? seedArtists : undefined,
    filters,
    count,
  };
}
