import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import { upsertTrack, upsertFeatures, setEmbedding } from "./repo";
import { handleMessage, TRANSLATION_TEMPERATURE } from "./loop";
import {
  type ChatOptions,
  type OllamaChatResponse,
  type OllamaMessage,
  type OllamaTool,
} from "./ollama";
import type { chatWithTools } from "./ollama";

function basis(...pairs: [number, number][]): Float32Array {
  const v = new Array<number>(EMBED_DIM).fill(0);
  for (const [i, val] of pairs) v[i] = val;
  return toEmbedding(v);
}

function seed(db: ReturnType<typeof openDb>, artist: string, emb: Float32Array): number {
  const id = upsertTrack(db, {
    navidromeId: artist,
    title: `${artist} song`,
    artist,
  });
  upsertFeatures(db, id, { rmsEnergy: 0.5, bpm: 120 });
  setEmbedding(db, id, emb);
  return id;
}

function toolResponse(args: Record<string, unknown>): OllamaChatResponse {
  return {
    message: {
      role: "assistant",
      content: "",
      tool_calls: [{ function: { name: "search_tracks", arguments: args } }],
    },
    done: true,
  };
}

function textResponse(content: string): OllamaChatResponse {
  return { message: { role: "assistant", content }, done: true };
}

/** Fake chat that replays a queue and records what each call saw. */
function queuedChat(responses: OllamaChatResponse[]) {
  const calls: { messages: OllamaMessage[]; tools: OllamaTool[]; opts?: ChatOptions }[] = [];
  let i = 0;
  const fn = (async (messages: OllamaMessage[], tools: OllamaTool[], opts?: ChatOptions) => {
    calls.push({ messages: structuredClone(messages), tools, opts });
    return responses[i++]!;
  }) as unknown as typeof chatWithTools;
  return { fn, calls };
}

const noEmbed = async (): Promise<Float32Array> => {
  throw new Error("embedText should not be called");
};

test("happy path: translates, searches, and narrates", async () => {
  const db = openDb(":memory:");
  const id = seed(db, "Mastodon", basis([0, 1]));

  const { fn, calls } = queuedChat([
    toolResponse({ seed_artists: ["Mastodon"], count: 5 }),
    textResponse("Here's a set in that vein."),
  ]);

  const res = await handleMessage(
    { db, embedText: noEmbed, chat: fn },
    "something like Mastodon",
  );

  expect(res.query?.seedArtists).toEqual(["Mastodon"]);
  expect(res.hits[0]!.id).toBe(id);
  expect(res.reply).toBe("Here's a set in that vein.");
  // second turn carried a tool-result message and no tools
  expect(calls[1]!.messages.some((m) => m.role === "tool")).toBe(true);
  expect(calls[1]!.tools).toEqual([]);
  // translation turn is near-greedy; narration keeps the server default
  expect(calls[0]!.opts?.temperature).toBe(TRANSLATION_TEMPERATURE);
  expect(calls[1]!.opts?.temperature).toBeUndefined();
  db.close();
});

test("model declining to call the tool surfaces its prose", async () => {
  const db = openDb(":memory:");
  const { fn } = queuedChat([textResponse("What sort of mood are you after?")]);

  const res = await handleMessage({ db, embedText: noEmbed, chat: fn }, "hi");

  expect(res.hits).toEqual([]);
  expect(res.reply).toBe("What sort of mood are you after?");
  db.close();
});

test("missing seed artist is reported to the model and the caller", async () => {
  const db = openDb(":memory:");
  const tool = seed(db, "Tool", basis([1, 1]));

  const { fn, calls } = queuedChat([
    toolResponse({ seed_artists: ["Ghost"], semantic_text: "knotty prog" }),
    textResponse("No Ghost on hand, but here's the sound."),
  ]);
  const embedText = async (): Promise<Float32Array> => basis([1, 1]);

  const res = await handleMessage({ db, embedText, chat: fn }, "like Ghost");

  expect(res.missingSeedArtists).toEqual(["Ghost"]);
  expect(res.hits[0]!.id).toBe(tool);
  const toolMsg = calls[1]!.messages.find((m) => m.role === "tool");
  expect(toolMsg?.content).toContain("Ghost");
  db.close();
});

test("empty model reply falls back to a generated summary", async () => {
  const db = openDb(":memory:");
  seed(db, "Mastodon", basis([0, 1]));

  const { fn } = queuedChat([
    toolResponse({ seed_artists: ["Mastodon"] }),
    textResponse(""),
  ]);

  const res = await handleMessage({ db, embedText: noEmbed, chat: fn }, "x");

  expect(res.reply).toBe("Found 1 tracks.");
  db.close();
});
