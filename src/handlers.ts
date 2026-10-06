import type { Database } from "bun:sqlite";
import { handleMessage } from "./loop";
import type { DecideMembership } from "./decider";
import type { EmbedText } from "./embedder";
import type { NavidromeClient } from "./navidrome";
import type { chatWithTools } from "./ollama";

// Pure request handlers, dependency-injected so they're testable without a live
// Ollama, embed-server, or Navidrome. server.ts wires the real deps + transport.

export interface ChatDeps {
  db: Database;
  embedText: EmbedText;
  chat?: typeof chatWithTools;
  decide?: DecideMembership;
}

export async function chatHandler(deps: ChatDeps, body: unknown) {
  const message = (body as { message?: unknown } | null)?.message;
  if (typeof message !== "string" || !message.trim()) {
    return Response.json({ error: "message required" }, { status: 400 });
  }
  const result = await handleMessage(
    {
      db: deps.db,
      embedText: deps.embedText,
      chat: deps.chat,
      decide: deps.decide,
    },
    message,
  );
  return Response.json(result);
}

export async function playlistHandler(
  navidrome: NavidromeClient,
  body: unknown,
): Promise<Response> {
  const b = (body ?? {}) as { name?: unknown; songIds?: unknown };
  if (
    typeof b.name !== "string" ||
    !b.name.trim() ||
    !Array.isArray(b.songIds) ||
    b.songIds.length === 0 ||
    !b.songIds.every((x): x is string => typeof x === "string")
  ) {
    return Response.json(
      { error: "name and a non-empty array of string songIds are required" },
      { status: 400 },
    );
  }
  const id = await navidrome.createPlaylist(b.name, b.songIds);
  return Response.json({ id });
}
