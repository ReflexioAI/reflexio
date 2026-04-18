import { execSync } from "node:child_process";

export interface MemorySearchResult {
  path: string;
  startLine: number;
  endLine: number;
  score: number;
  snippet: string;
  source: string;
}

export interface MemorySearchResponse {
  results: MemorySearchResult[];
}

/**
 * Call `openclaw memory search` CLI and return parsed results.
 * Returns empty array on any failure (graceful degradation).
 */
export function memorySearch(
  query: string,
  maxResults: number = 5
): MemorySearchResult[] {
  try {
    const escaped = query.replace(/'/g, "'\\''");
    const cmd = `openclaw memory search '${escaped}' --json --max-results ${maxResults}`;
    const output = execSync(cmd, {
      encoding: "utf8",
      timeout: 30_000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    const parsed: MemorySearchResponse = JSON.parse(output.trim());
    return parsed.results || [];
  } catch (err) {
    console.error(`[reflexio] openclaw memory search failed: ${err}`);
    return [];
  }
}

/**
 * Call `openclaw infer` CLI with a prompt and return the raw text response.
 * Returns null on any failure.
 */
export function infer(prompt: string): string | null {
  try {
    const escaped = prompt.replace(/'/g, "'\\''");
    const cmd = `openclaw infer '${escaped}'`;
    const output = execSync(cmd, {
      encoding: "utf8",
      timeout: 30_000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    return output.trim();
  } catch (err) {
    console.error(`[reflexio] openclaw infer failed: ${err}`);
    return null;
  }
}
