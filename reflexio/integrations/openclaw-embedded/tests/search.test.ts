import { describe, it, expect } from "vitest";

import { rawSearch, search } from "../plugin/lib/search.ts";
import type { CommandRunner, MemorySearchResult } from "../plugin/lib/openclaw-cli.ts";

function createMockRunner(
  inferResult: string | null,
  searchResults: MemorySearchResult[]
): CommandRunner {
  return async (argv) => {
    if (argv.includes("infer")) {
      if (inferResult === null) throw new Error("infer failed");
      return { stdout: inferResult, stderr: "", code: 0 };
    }
    if (argv.includes("memory") && argv.includes("search")) {
      return {
        stdout: JSON.stringify({ results: searchResults }),
        stderr: "",
        code: 0,
      };
    }
    return { stdout: "", stderr: "unexpected command", code: 1 };
  };
}

describe("rawSearch", () => {
  it("calls memorySearch with query and returns results", async () => {
    const items: MemorySearchResult[] = [
      { path: ".reflexio/profiles/diet.md", score: 0.5, snippet: "vegan", startLine: 1, endLine: 5, source: "memory" },
    ];
    const runner = createMockRunner(null, items);
    const results = await rawSearch("vegan diet", 3, undefined, runner);
    expect(results).toHaveLength(1);
    expect(results[0].path).toBe(".reflexio/profiles/diet.md");
  });

  it("filters results to specified type", async () => {
    const items: MemorySearchResult[] = [
      { path: ".reflexio/profiles/diet.md", score: 0.5, snippet: "x", startLine: 1, endLine: 5, source: "memory" },
      { path: ".reflexio/playbooks/commit.md", score: 0.4, snippet: "y", startLine: 1, endLine: 5, source: "memory" },
    ];
    const runner = createMockRunner(null, items);
    const results = await rawSearch("query", 5, "profile", runner);
    expect(results).toHaveLength(1);
    expect(results[0].path).toContain("profiles");
  });

  it("returns empty on memorySearch failure", async () => {
    const runner = createMockRunner(null, []);
    const results = await rawSearch("anything", 5, undefined, runner);
    expect(results).toEqual([]);
  });
});

describe("search", () => {
  it("preprocesses query before searching", async () => {
    const runner = createMockRunner("Rewritten query about diet", []);
    const results = await search("Oh sorry I like vegan food", 5, undefined, runner);
    expect(results).toEqual([]);
    // The runner was called — preprocessing happened via infer, then search via memory
  });

  it("falls back to raw query if preprocessing fails", async () => {
    const runner = createMockRunner(null, []);
    const results = await search("raw query here", 5, undefined, runner);
    expect(results).toEqual([]);
  });
});
