import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import { upsertTrack, upsertFeatures, setEmbedding, replaceSegments } from "./repo";
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
  replaceSegments(db, id, [{ startS: 0, endS: 10, embedding: emb }]);
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
    return responses[i++] ?? textResponse("");
  }) as unknown as typeof chatWithTools;
  return { fn, calls };
}

const noEmbed = async (): Promise<Float32Array> => {
  throw new Error("embedText should not be called");
};

test("happy path translates, searches, and summarizes grounded hits", async () => {
  const db = openDb(":memory:");
  const id = seed(db, "Mastodon", basis([0, 1]));

  const { fn, calls } = queuedChat([
    toolResponse({
      semantic_text: "progressive sludge metal",
      seed_artists: ["Mastodon"],
      count: 1,
    }),
    textResponse("Here's a set in that vein."),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "1 track like Mastodon",
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
  expect(res.request).toBe("1 track like Mastodon");
  db.close();
});

test("model declining to call the tool is retried instead of trusted", async () => {
  const db = openDb(":memory:");
  const id = seed(db, "Mastodon", basis([0, 1]));
  const { fn, calls } = queuedChat([
    textResponse("I couldn't find anything like that"),
    toolResponse({
      semantic_text: "progressive sludge metal",
      seed_artists: ["Mastodon"],
      count: 1,
    }),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "1 track please",
  );

  expect(res.hits[0]!.id).toBe(id);
  expect(res.reply).toBe("Found 1 track across Mastodon.");
  expect(calls).toHaveLength(3);
  expect(calls[1]!.messages[0]!.content).toContain("violated the protocol");
  expect(calls[1]!.opts?.temperature).toBe(TRANSLATION_TEMPERATURE);
  expect(calls[2]!.tools).toEqual([]);
  db.close();
});

test("repeated planner refusal fails instead of inventing a search result", async () => {
  const db = openDb(":memory:");
  const { fn } = queuedChat([
    textResponse("I couldn't find anything like that"),
    textResponse("Still nothing"),
  ]);

  await expect(handleMessage({ db, embedText: noEmbed, chat: fn }, "hi")).rejects.toThrow(
    "music search planner did not call search_tracks",
  );
  db.close();
});

test("missing seed artist is reported in the grounded summary and result", async () => {
  const db = openDb(":memory:");
  const tool = seed(db, "Tool", basis([1, 1]));

  const { fn } = queuedChat([
    toolResponse({
      seed_artists: ["Ghost"],
      semantic_text: "knotty prog",
      count: 1,
    }),
  ]);
  const embedText = async (): Promise<Float32Array> => basis([1, 1]);

  const res = await handleMessage({ db, embedText, chat: fn }, "1 track like Ghost");

  expect(res.missingSeedArtists).toEqual(["Ghost"]);
  expect(res.hits[0]!.id).toBe(tool);
  expect(res.reply).toBe("Found 1 track across Tool. Seed artists not found: Ghost.");
  db.close();
});

test("result summary uses only artists present in the hits", async () => {
  const db = openDb(":memory:");
  seed(db, "Mastodon", basis([0, 1]));

  const { fn } = queuedChat([
    toolResponse({
      semantic_text: "progressive sludge metal",
      seed_artists: ["Mastodon"],
      count: 1,
    }),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "1 track x",
  );

  expect(res.reply).toBe("Found 1 track across Mastodon.");
  db.close();
});

test("result summary collapses artist casing variants", async () => {
  const db = openDb(":memory:");
  seed(db, "Mastodon", basis([0, 1]));
  seed(db, "MASTODON", basis([0, 1], [1, 0.01]));
  const { fn } = queuedChat([
    toolResponse({
      semantic_text: "progressive sludge metal",
      seed_artists: ["Mastodon"],
      count: 2,
    }),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "2 tracks x",
  );

  expect(res.reply).toBe("Found 2 tracks across Mastodon.");
  db.close();
});

test("constraint shortfalls are explained instead of reported as generic matching", async () => {
  const db = openDb(":memory:");
  const slow = seed(db, "Slow", basis([0, 1]));
  const fast = seed(db, "Fast", basis([0, 1], [1, 0.1]));
  upsertFeatures(db, slow, { rmsEnergy: 0.5, bpm: 100 });
  upsertFeatures(db, fast, { rmsEnergy: 0.5, bpm: 160 });
  const { fn } = queuedChat([
    toolResponse({
      semantic_text: "fast metal",
      constraints: [
        {
          feature: "tempo_bpm",
          operator: "min",
          value: 140,
          evidence: "over 140 BPM",
        },
      ],
      count: 2,
    }),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "2 tracks of metal over 140 BPM",
  );

  expect(res.hits.map((hit) => hit.artist)).toEqual(["Fast"]);
  expect(res.reply).toBe(
    "Found 1 of 2 requested tracks across Fast. Explicit constraints limited the result.",
  );
  expect(res.diagnostics?.shortfallReason).toBe("constraints");
  db.close();
});

test("empty analyzed corpus is distinguished from an ordinary miss", async () => {
  const db = openDb(":memory:");
  const { fn } = queuedChat([
    toolResponse({ semantic_text: "metal", count: 1 }),
  ]);

  const res = await handleMessage(
    { db, embedText: async () => basis([0, 1]), chat: fn },
    "1 metal track",
  );

  expect(res.reply).toBe("The library has no analyzed tracks.");
  db.close();
});
