import { test, expect } from "bun:test";
import { makeEmbedder } from "./embedder";
import { EMBED_DIM } from "./db";

test("posts text and parses the embedding into a Float32Array", async () => {
  const vec = Array.from({ length: EMBED_DIM }, (_, i) => (i === 0 ? 1 : 0));
  let captured: { url: string; body: any } | undefined;
  const fakeFetch = (async (url: string, init: RequestInit) => {
    captured = { url, body: JSON.parse(init.body as string) };
    return new Response(JSON.stringify({ embedding: vec }), { status: 200 });
  }) as unknown as typeof fetch;

  const embed = makeEmbedder({ baseUrl: "http://clap:8001", fetchImpl: fakeFetch });
  const out = await embed("heavy metal");

  expect(captured?.url).toBe("http://clap:8001/embed_text");
  expect(captured?.body.text).toBe("heavy metal");
  expect(out).toBeInstanceOf(Float32Array);
  expect(out.length).toBe(EMBED_DIM);
  expect(out[0]).toBe(1);
});

test("throws on a non-ok response", async () => {
  const fakeFetch = (async () =>
    new Response("boom", { status: 500 })) as unknown as typeof fetch;
  const embed = makeEmbedder({ fetchImpl: fakeFetch });
  await expect(embed("x")).rejects.toThrow();
});
