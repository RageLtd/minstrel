// Subsonic client for the orchestrator's last mile: write a playlist of track
// IDs into Navidrome. The IDs are exactly what searchTracks returns, so there's
// no mapping step. Auth mirrors the Python analyzer's client (token+salt).

const API_VERSION = "1.16.1";

export interface NavidromeOptions {
  baseUrl?: string;
  username?: string;
  password?: string;
  client?: string;
  fetchImpl?: typeof fetch;
}

export interface NavidromeClient {
  /** Create a playlist from Navidrome song IDs; returns the new playlist id. */
  createPlaylist(name: string, songIds: string[]): Promise<string>;
}

function authParams(username: string, password: string, client: string): [string, string][] {
  const salt = crypto.randomUUID().replaceAll("-", "").slice(0, 16);
  const token = new Bun.CryptoHasher("md5").update(password + salt).digest("hex");
  return [
    ["u", username],
    ["t", token],
    ["s", salt],
    ["v", API_VERSION],
    ["c", client],
    ["f", "json"],
  ];
}

interface SubsonicResponse {
  status?: string;
  error?: { code?: number; message?: string };
  playlist?: { id?: string | number };
}

export function makeNavidrome(opts: NavidromeOptions = {}): NavidromeClient {
  const baseUrl = (
    opts.baseUrl ??
    process.env.NAVIDROME_URL ??
    "http://localhost:4533"
  ).replace(/\/$/, "");
  const username = opts.username ?? process.env.NAVIDROME_USER ?? "";
  const password = opts.password ?? process.env.NAVIDROME_PASS ?? "";
  const client = opts.client ?? "minstrel";
  const doFetch = opts.fetchImpl ?? fetch;

  async function call(
    endpoint: string,
    params: [string, string][],
  ): Promise<SubsonicResponse> {
    const search = new URLSearchParams([
      ...params,
      ...authParams(username, password, client),
    ]);
    const res = await doFetch(`${baseUrl}/rest/${endpoint}?${search.toString()}`);
    if (!res.ok) {
      throw new Error(`navidrome ${endpoint} failed: ${res.status}`);
    }
    const body = (await res.json()) as { "subsonic-response"?: SubsonicResponse };
    const sub = body["subsonic-response"] ?? {};
    if (sub.status === "failed") {
      throw new Error(
        `navidrome ${endpoint}: ${sub.error?.message ?? "request failed"}`,
      );
    }
    return sub;
  }

  return {
    async createPlaylist(name, songIds) {
      const params: [string, string][] = [["name", name]];
      for (const id of songIds) params.push(["songId", id]);
      const sub = await call("createPlaylist", params);
      return String(sub.playlist?.id ?? "");
    },
  };
}
