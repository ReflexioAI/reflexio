/**
 * Abstraction over command execution.
 * Plugin runtime injects api.runtime.system.runCommandWithTimeout.
 * Tests inject a mock.
 */
export type CommandRunner = (
  argv: string[],
  opts: { timeoutMs: number; input?: string }
) => Promise<{ stdout: string; stderr: string; code: number | null }>;

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
 * Call `openclaw memory search` via the injected runner.
 * Returns empty array on any failure (graceful degradation).
 */
export async function memorySearch(
  query: string,
  maxResults: number,
  runner: CommandRunner
): Promise<MemorySearchResult[]> {
  try {
    const result = await runner(
      ["openclaw", "memory", "search", query, "--json", "--max-results", String(maxResults)],
      { timeoutMs: 30_000 }
    );
    const parsed: MemorySearchResponse = JSON.parse(result.stdout.trim());
    return parsed.results || [];
  } catch (err) {
    console.error(`[reflexio] openclaw memory search failed: ${err}`);
    return [];
  }
}

/**
 * Call `openclaw infer` via the injected runner.
 * Returns null on any failure.
 */
export async function infer(
  prompt: string,
  runner: CommandRunner
): Promise<string | null> {
  try {
    const result = await runner(
      ["openclaw", "infer", prompt],
      { timeoutMs: 30_000 }
    );
    return result.stdout.trim() || null;
  } catch (err) {
    console.error(`[reflexio] openclaw infer failed: ${err}`);
    return null;
  }
}
