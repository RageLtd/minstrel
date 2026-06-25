import { test, expect } from "bun:test";
import { openDb, toEmbedding, EMBED_DIM } from "./db";
import { upsertTrack, upsertFeatures, setEmbedding } from "./repo";
import { chatHandler, playlistHandler } from "./handlers";
import type { OllamaChatResponse, chatWithTools } from "./ollama";
import type { NavidromeClient } from "./navidrome";

function basis(...pairs: [number, number][]): Float32Array {
  const v = new Array<number>(EMBED_DIM).fill(0);
  for (const [i, val] of pairs) v[i] = val;
  return toEmbedding(v);
}

function seed(db: ReturnType<typeof openDb>, artist: string): void {
  const id = upsertTrack(db, { navidromeId: `nav-${artist}`, title: `${artist} song`, artist });
  upsertFeatures(db, id, { rmsEnergy: 0.5, bpm: 120 });
  setEmbedding(db, id, basis([0, 1]));
}

function queuedChat(responses: OllamaChatResponse[]): typeof chatWithTools {
  let i = 0;
  return (async () => responses[i++]!) as unknown as typeof chatWithTools;
}

const noEmbed = async (): Promise<Float32Array> => basis([0, 1]);

test("chatHandler returns reply + hits for a valid message", async () => {
  const db = openDb(":memory:");
  seed(db, "Mastodon");
  const chat = queuedChat([
    {
      message: {
        role: "assistant",
        content: "",
        tool_calls: [{ function: { name: "search_tracks", arguments: { seed_artists: ["Mastodon"] } } }],
      },
      done: true,
    },
    { message: { role: "assistant", content: "Here you go." }, done: true },
  ]);

  const res = await chatHandler({ db, embedText: noEmbed, chat }, { message: "like Mastodon" });
  expect(res.status).toBe(200);
  const data = (await res.json()) as { reply: string; hits: unknown[] };
  expect(data.reply).toBe("Here you go.");
  expect(data.hits).toHaveLength(1);
  db.close();
});

test("chatHandler rejects an empty message with 400", async () => {
  const db = openDb(":memory:");
  const res = await chatHandler({ db, embedText: noEmbed }, { message: "   " });
  expect(res.status).toBe(400);
  db.close();
});

const fakeNav: NavidromeClient = {
  async createPlaylist() {
    return "pl-7";
  },
};

test("playlistHandler creates a playlist and returns its id", async () => {
  const res = await playlistHandler(fakeNav, { name: "Doom", songIds: ["a", "b"] });
  expect(res.status).toBe(200);
  expect(await res.json()).toEqual({ id: "pl-7" });
});

test("playlistHandler rejects missing name or empty songIds", async () => {
  expect((await playlistHandler(fakeNav, { songIds: ["a"] })).status).toBe(400);
  expect((await playlistHandler(fakeNav, { name: "x", songIds: [] })).status).toBe(400);
});
