import type { Database } from "bun:sqlite";
import {
  chatWithTools,
  firstToolCall,
  type ChatOptions,
  type OllamaMessage,
} from "./ollama";
import {
  SEARCH_TRACKS_TOOL,
  SYSTEM_PROMPT,
  parseSearchToolCall,
  type SearchQuery,
} from "./tools";
import { executeSearch, type SearchResult } from "./search";
import type { EmbedText } from "./embedder";
import type { SearchHit } from "./repo";

export interface HandleResult {
  reply: string;
  hits: SearchHit[];
  query?: SearchQuery;
  missingSeedArtists: string[];
}

export interface HandleDeps {
  db: Database;
  embedText: EmbedText;
  /** Injectable for tests; defaults to the real Ollama client. */
  chat?: typeof chatWithTools;
  chatOptions?: ChatOptions;
}

/** Compact, vector-free view of the results for the model to narrate. */
function summarizeForModel(result: SearchResult) {
  return {
    found: result.hits.length,
    tracks: result.hits.slice(0, 50).map((h) => ({ title: h.title, artist: h.artist })),
    missing_seed_artists: result.missingSeedArtists,
  };
}

/** Fallback reply if the model returns empty content on the second turn. */
function defaultReply(result: SearchResult): string {
  const missing =
    result.missingSeedArtists.length > 0
      ? ` (nothing by ${result.missingSeedArtists.join(", ")} in your library)`
      : "";
  if (result.hits.length === 0) {
    return `I couldn't find anything matching that${missing}.`;
  }
  return `Found ${result.hits.length} tracks${missing}.`;
}

/**
 * One conversational turn: translate the user's vibe into a search via the LLM,
 * run it, then let the LLM narrate the result. The second turn passes no tools so
 * the model produces prose rather than calling the tool again.
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

  const first = await chat(messages, [SEARCH_TRACKS_TOOL], options);
  const call = firstToolCall(first);
  if (!call || call.name !== "search_tracks") {
    return {
      reply: first.message.content || "I couldn't turn that into a search.",
      hits: [],
      missingSeedArtists: [],
    };
  }

  const query = parseSearchToolCall(call.args);
  const result = await executeSearch(deps.db, deps.embedText, query);

  messages.push(first.message);
  messages.push({
    role: "tool",
    tool_name: "search_tracks",
    content: JSON.stringify(summarizeForModel(result)),
  });
  const second = await chat(messages, [], options);

  return {
    reply: second.message.content || defaultReply(result),
    hits: result.hits,
    query,
    missingSeedArtists: result.missingSeedArtists,
  };
}
