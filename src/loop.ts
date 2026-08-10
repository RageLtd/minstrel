import type { Database } from "bun:sqlite";
import {
  chatWithTools,
  firstToolCall,
  type ChatOptions,
  type OllamaMessage,
} from "./ollama";
import {
  NARRATION_PROMPT,
  SEARCH_TRACKS_TOOL,
  SEARCH_RETRY_PROMPT,
  SYSTEM_PROMPT,
  parseSearchToolCall,
  type SearchQuery,
} from "./tools";
import {
  executeSearch,
  type SearchDiagnostics,
  type SearchResult,
} from "./search";
import type { EmbedText } from "./embedder";
import type { SearchHit } from "./repo";

/**
 * Sampling temperature for the translation turn. Vibe→query is a structured
 * task with a schema to hit — near-greedy decoding kills the coin-flip
 * failure class (no tool call at all) that default temperature produces.
 * The narration turn deliberately keeps the server default: variety in
 * prose (and in how a vibe is interpreted) is a feature, not a flake.
 */
export const TRANSLATION_TEMPERATURE = 0.1;

export interface HandleResult {
  reply: string;
  hits: SearchHit[];
  query?: SearchQuery;
  diagnostics?: SearchDiagnostics;
  request: string;
  missingSeedArtists: string[];
}

export interface HandleDeps {
  db: Database;
  embedText: EmbedText;
  /** Injectable for tests; defaults to the real Ollama client. */
  chat?: typeof chatWithTools;
  chatOptions?: ChatOptions;
}

/** Compact, vector-free result facts for the narration turn. */
function summarizeForModel(result: SearchResult, fallbackReply: string) {
  return {
    found: result.hits.length,
    requested: result.diagnostics.requested,
    tracks: result.hits.slice(0, 50).map((hit) => ({
      title: hit.title,
      artist: hit.artist,
    })),
    missing_seed_artists: result.missingSeedArtists,
    shortfall_reason: result.diagnostics.shortfallReason,
    deterministic_summary: fallbackReply,
  };
}

/** Grounded summary built only from deterministic search output. */
function resultReply(result: SearchResult): string {
  const missingSuffix =
    result.missingSeedArtists.length > 0
      ? ` Seed artists not found: ${result.missingSeedArtists.join(", ")}.`
      : "";
  if (result.hits.length === 0) {
    const reason =
      result.diagnostics.searchableTracks === 0
        ? "The library has no analyzed tracks."
        : result.diagnostics.shortfallReason === "constraints"
        ? "No tracks satisfied the explicit constraints."
        : result.diagnostics.shortfallReason === "exclusions"
          ? "No tracks remained after the requested exclusions."
        : "No tracks matched that search.";
    return `${reason}${missingSuffix}`;
  }

  const artistsByKey = new Map<string, string>();
  for (const hit of result.hits) {
    const artist = hit.artist?.trim();
    if (artist && !artistsByKey.has(artist.toLocaleLowerCase())) {
      artistsByKey.set(artist.toLocaleLowerCase(), artist);
    }
  }
  const artists = [...artistsByKey.values()];
  const namedArtists = artists.slice(0, 4).join(", ");
  const remainingArtists = artists.length > 4 ? ` and ${artists.length - 4} more` : "";
  const artistSuffix = namedArtists ? ` across ${namedArtists}${remainingArtists}` : "";
  const trackLabel = result.hits.length === 1 ? "track" : "tracks";
  const requestedTrackLabel =
    result.diagnostics.requested === 1 ? "track" : "tracks";
  const countPrefix =
    result.diagnostics.returned < result.diagnostics.requested
      ? `Found ${result.diagnostics.returned} of ${result.diagnostics.requested} requested ${requestedTrackLabel}`
      : `Found ${result.hits.length} ${trackLabel}`;
  const shortfallSuffix =
    result.diagnostics.shortfallReason === "constraints"
      ? " Explicit constraints limited the result."
      : result.diagnostics.shortfallReason === "exclusions"
        ? " Requested exclusions limited the result."
      : result.diagnostics.shortfallReason === "strictArtistCap"
        ? " The requested per-artist cap limited the result."
        : "";
  return `${countPrefix}${artistSuffix}.${shortfallSuffix}${missingSuffix}`;
}

/**
 * One conversational turn: translate the user's vibe into a search via the LLM,
 * run deterministic matching, then summarize only the grounded result.
 */
export async function handleMessage(
  deps: HandleDeps,
  userText: string,
): Promise<HandleResult> {
  const chat = deps.chat ?? chatWithTools;
  const options = deps.chatOptions ?? {};

  const messages: OllamaMessage[] = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userText },
  ];

  const translationOptions: ChatOptions = {
    temperature: TRANSLATION_TEMPERATURE,
    ...options,
  };
  let first = await chat(messages, [SEARCH_TRACKS_TOOL], translationOptions);
  let call = firstToolCall(first);
  if (!call || call.name !== "search_tracks") {
    messages[0] = { role: "system", content: SEARCH_RETRY_PROMPT };
    first = await chat(messages, [SEARCH_TRACKS_TOOL], translationOptions);
    call = firstToolCall(first);
  }
  if (!call || call.name !== "search_tracks") {
    throw new Error("music search planner did not call search_tracks");
  }

  const query = parseSearchToolCall(call.args, userText);
  const result = await executeSearch(deps.db, deps.embedText, query);
  const fallbackReply = resultReply(result);

  messages[0] = { role: "system", content: NARRATION_PROMPT };
  messages.push(first.message);
  messages.push({
    role: "tool",
    tool_name: "search_tracks",
    content: JSON.stringify(summarizeForModel(result, fallbackReply)),
  });

  let reply = fallbackReply;
  try {
    const second = await chat(messages, [], options);
    reply = second.message.content.trim() || fallbackReply;
  } catch {
    // Search succeeded; narration failure must not discard grounded results.
  }

  return {
    reply,
    hits: result.hits,
    query,
    diagnostics: result.diagnostics,
    request: userText.trim(),
    missingSeedArtists: result.missingSeedArtists,
  };
}
