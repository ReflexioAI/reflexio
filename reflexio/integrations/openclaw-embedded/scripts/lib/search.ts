import { memorySearch, type MemorySearchResult } from "./openclaw-cli.js";
import { preprocessQuery } from "./dedup.js";

/**
 * Search memory with a raw query string, optionally filtering by type.
 */
export function rawSearch(
  query: string,
  maxResults: number = 5,
  type?: "profile" | "playbook"
): MemorySearchResult[] {
  let results = memorySearch(query, maxResults);
  if (type) {
    const typeDir = type === "profile" ? "/profiles/" : "/playbooks/";
    results = results.filter((r) => r.path.includes(typeDir));
  }
  return results;
}

/**
 * Preprocess query (via LLM rewrite) then search memory.
 */
export function search(
  rawQuery: string,
  maxResults: number = 5,
  type?: "profile" | "playbook"
): MemorySearchResult[] {
  const query = preprocessQuery(rawQuery);
  return rawSearch(query, maxResults, type);
}
