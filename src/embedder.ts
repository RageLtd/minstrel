// Client for the Python CLAP text-embedding service. Query text must be embedded
// by the same CLAP model that embedded the audio, so this crosses to Python
// rather than embedding text in TypeScript.

export type EmbedText = (text: string) => Promise<Float32Array>;

export interface EmbedderOptions {
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

export function makeEmbedder(opts: EmbedderOptions = {}): EmbedText {
  const baseUrl =
    opts.baseUrl ?? process.env.MINSTREL_EMBED_URL ?? "http://localhost:8001";
  const doFetch = opts.fetchImpl ?? fetch;

  return async (text: string): Promise<Float32Array> => {
    const res = await doFetch(`${baseUrl}/embed_text`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      throw new Error(`embed_text failed: ${res.status} ${await res.text()}`);
    }
    const data = (await res.json()) as { embedding: number[] };
    return Float32Array.from(data.embedding);
  };
}
