import index from "../app.html";
import { openDb } from "./db";
import { makeDecider } from "./decider";
import { makeEmbedder } from "./embedder";
import { makeNavidrome } from "./navidrome";
import { chatHandler, playlistHandler } from "./handlers";

const db = openDb(process.env.MINSTREL_DB ?? "minstrel.db");
const embedText = makeEmbedder();
const navidrome = makeNavidrome();
// An empty MINSTREL_DECISION_MODEL disables the membership gate entirely.
const decide =
  process.env.MINSTREL_DECISION_MODEL === "" ? undefined : makeDecider();

async function parseJson(req: Request): Promise<unknown> {
  try {
    return await req.json();
  } catch {
    return null;
  }
}

function serverError(err: unknown): Response {
  return Response.json(
    { error: err instanceof Error ? err.message : String(err) },
    { status: 500 },
  );
}

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  routes: {
    "/": index,
    "/api/chat": {
      POST: async (req) => {
        try {
          return await chatHandler({ db, embedText, decide }, await parseJson(req));
        } catch (err) {
          return serverError(err);
        }
      },
    },
    "/api/playlist": {
      POST: async (req) => {
        try {
          return await playlistHandler(navidrome, await parseJson(req));
        } catch (err) {
          return serverError(err);
        }
      },
    },
  },
  development:
    process.env.NODE_ENV === "production"
      ? false
      : { hmr: true, console: true },
});

console.log(`Minstrel listening on http://localhost:${server.port}`);
