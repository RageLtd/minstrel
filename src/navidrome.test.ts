import { test, expect } from "bun:test";
import { makeNavidrome } from "./navidrome";

function okResponse(body: unknown): typeof fetch {
  return (async () =>
    new Response(JSON.stringify(body), { status: 200 })) as unknown as typeof fetch;
}

test("createPlaylist posts name + repeated songId and auth, returns the id", async () => {
  let capturedUrl = "";
  const fakeFetch = (async (url: string) => {
    capturedUrl = url;
    return new Response(
      JSON.stringify({ "subsonic-response": { status: "ok", playlist: { id: "pl-1" } } }),
      { status: 200 },
    );
  }) as unknown as typeof fetch;

  const nav = makeNavidrome({
    baseUrl: "http://nav:4533",
    username: "alice",
    password: "secret",
    fetchImpl: fakeFetch,
  });
  const id = await nav.createPlaylist("Doom Sunday", ["s1", "s2", "s3"]);

  expect(id).toBe("pl-1");
  const u = new URL(capturedUrl);
  expect(u.pathname).toBe("/rest/createPlaylist");
  expect(u.searchParams.get("name")).toBe("Doom Sunday");
  expect(u.searchParams.getAll("songId")).toEqual(["s1", "s2", "s3"]);
  expect(u.searchParams.get("u")).toBe("alice");
  expect(u.searchParams.get("t")?.length).toBe(32); // md5 hex
  expect(u.searchParams.get("s")).toBeTruthy();
  expect(u.searchParams.get("f")).toBe("json");
});

test("throws on a Subsonic failure status (HTTP 200)", async () => {
  const nav = makeNavidrome({
    username: "u",
    password: "bad",
    fetchImpl: okResponse({
      "subsonic-response": {
        status: "failed",
        error: { code: 40, message: "Wrong username or password" },
      },
    }),
  });
  await expect(nav.createPlaylist("x", ["s1"])).rejects.toThrow(/Wrong username/);
});
