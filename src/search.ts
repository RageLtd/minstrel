import type { Database } from "bun:sqlite";
import { artistEmbeddings, searchTracks, type SearchHit } from "./repo";
import type { EmbedText } from "./embedder";
import type { SearchQuery } from "./tools";
import { meanNormalize } from "./vec";

export interface SearchResult {
  hits: SearchHit[];
  /** Seed artists the user named that aren't in the library (for the reply). */
  missingSeedArtists: string[];
}

/**
 * Turn a validated SearchQuery into ranked tracks. Builds the query vector from
 * a seed-artist centroid and/or the CLAP-embedded semantic text, then narrows by
 * the scalar filters. A named seed artist that isn't in the library is reported
 * back and the search falls through to the semantic text.
 */
export async function executeSearch(
  db: Database,
  embedText: EmbedText,
  query: SearchQuery,
): Promise<SearchResult> {
  const parts: Float32Array[] = [];
  const missingSeedArtists: string[] = [];

  if (query.seedArtists) {
    const seedVecs: Float32Array[] = [];
    for (const artist of query.seedArtists) {
      const found = artistEmbeddings(db, artist);
      if (found.length === 0) missingSeedArtists.push(artist);
      else seedVecs.push(...found);
    }
    if (seedVecs.length > 0) parts.push(meanNormalize(seedVecs));
  }

  if (query.semanticText) {
    parts.push(await embedText(query.semanticText));
  }

  if (parts.length === 0) {
    return { hits: [], missingSeedArtists };
  }

  const queryVec = parts.length === 1 ? parts[0]! : meanNormalize(parts);
  const hits = searchTracks(db, queryVec, query.count, query.filters);
  return { hits, missingSeedArtists };
}
