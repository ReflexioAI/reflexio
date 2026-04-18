import { memorySearch, type MemorySearchResult, type CommandRunner } from "./openclaw-cli.js";
import { preprocessQuery } from "./dedup.js";

/**
 * Search memory with a query string, optionally filtering by type.
 */
export async function rawSearch(
  query: string,
  maxResults: number,
  type: "profile" | "playbook" | undefined,
  runner: CommandRunner
): Promise<MemorySearchResult[]> {
  let results = await memorySearch(query, maxResults, runner);
  if (type) {
    const typeDir = type === "profile" ? "/profiles/" : "/playbooks/";
    results = results.filter((r) => r.path.includes(typeDir));
  }
  return results;
}

/**
 * Preprocess query (via LLM rewrite) then search memory.
 */
export async function search(
  rawQuery: string,
  maxResults: number,
  type: "profile" | "playbook" | undefined,
  runner: CommandRunner
): Promise<MemorySearchResult[]> {
  const query = await preprocessQuery(rawQuery, runner);
  return rawSearch(query, maxResults, type, runner);
}
