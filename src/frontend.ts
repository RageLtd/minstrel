import html from "solid-js/html";
import { render } from "solid-js/web";
import { createSignal } from "solid-js";
import {
  describeSearchDiagnostics,
  describeSearchQuery,
} from "./query-display";
import type { SearchDiagnostics } from "./search";
import type { SearchQuery } from "./tools";
import "./index.css";

interface Hit {
  id: number;
  navidromeId: string;
  title: string | null;
  artist: string | null;
  album: string | null;
}
interface ChatResult {
  reply: string;
  hits: Hit[];
  query?: SearchQuery;
  diagnostics?: SearchDiagnostics;
  request: string;
  missingSeedArtists: string[];
}

function App() {
  const [input, setInput] = createSignal("");
  const [loading, setLoading] = createSignal(false);
  const [result, setResult] = createSignal<ChatResult | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [saved, setSaved] = createSignal<string | null>(null);
  let requestGeneration = 0;

  async function ask() {
    const message = input().trim();
    if (!message) return;
    const generation = ++requestGeneration;
    setLoading(true);
    setError(null);
    setSaved(null);
    setResult(null);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ message }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "request failed");
      if (generation !== requestGeneration) return;
      setResult(data as ChatResult);
    } catch (e) {
      if (generation !== requestGeneration) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (generation === requestGeneration) setLoading(false);
    }
  }

  async function save() {
    const r = result();
    if (!r || r.hits.length === 0) return;
    const name = r.request.slice(0, 60) || "Minstrel mix";
    try {
      const res = await fetch("/api/playlist", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, songIds: r.hits.map((h) => h.navidromeId) }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "save failed");
      if (result() !== r) return;
      setSaved(String(data.id));
    } catch (e) {
      if (result() !== r) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return html`
    <main class="app">
      <h1>Minstrel</h1>
      <form
        class="composer"
        onSubmit=${(e: Event) => {
          e.preventDefault();
          ask();
        }}
      >
        <input
          placeholder="something like early Mastodon but higher energy…"
          value=${input}
          onInput=${(e: Event) => setInput((e.target as HTMLInputElement).value)}
        />
        <button type="submit" disabled=${loading}>
          ${() => (loading() ? "…" : "Ask")}
        </button>
      </form>
      ${() => error() && html`<p class="error">${error}</p>`}
      ${() =>
        result() &&
        html`
          <section class="result">
            <p class="reply">${() => result()!.reply}</p>
            ${() =>
              result()!.query &&
              html`
                <details class="query-debug" open>
                  <summary>Search query</summary>
                  <dl class="query-details">
                    ${() =>
                      describeSearchQuery(result()!.query!).map(
                        (detail) => html`
                          <div>
                            <dt>${detail.label}</dt>
                            <dd>${detail.value}</dd>
                          </div>
                        `,
                      )}
                  </dl>
                  ${() =>
                    result()!.diagnostics &&
                    html`
                      <h3>Execution</h3>
                      <dl class="query-details">
                        ${() =>
                          describeSearchDiagnostics(result()!.diagnostics!).map(
                            (detail) => html`
                              <div>
                                <dt>${detail.label}</dt>
                                <dd>${detail.value}</dd>
                              </div>
                            `,
                          )}
                      </dl>
                    `}
                  <pre class="query-json" aria-label="Normalized search query JSON">${() =>
                    JSON.stringify(result()!.query, null, 2)}</pre>
                </details>
              `}
            <ul class="tracks">
              ${() =>
                result()!.hits.map(
                  (h) => html`
                    <li>
                      <span class="title">${h.title ?? "Unknown"}</span>
                      <span class="artist">${h.artist ?? ""}</span>
                    </li>
                  `,
                )}
            </ul>
            ${() =>
              result()!.hits.length > 0 &&
              html`<button class="save" onClick=${save}>Save to Navidrome</button>`}
            ${() => saved() && html`<p class="saved">Saved playlist ${saved}.</p>`}
          </section>
        `}
    </main>
  `;
}

const root = document.getElementById("app");
if (root) render(App, root);
