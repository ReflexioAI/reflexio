import { describe, it, expect, vi, beforeEach } from "vitest";

vi.mock("../lib/openclaw-cli.js", () => ({
  memorySearch: vi.fn(),
  infer: vi.fn(),
}));

import { rawSearch, search } from "../lib/search.js";
import { memorySearch, infer } from "../lib/openclaw-cli.js";

const mockMemorySearch = vi.mocked(memorySearch);
const mockInfer = vi.mocked(infer);

beforeEach(() => vi.clearAllMocks());

describe("rawSearch", () => {
  it("calls memorySearch with query and returns results", () => {
    mockMemorySearch.mockReturnValue([
      { path: ".reflexio/profiles/diet.md", score: 0.5, snippet: "vegan", startLine: 1, endLine: 5, source: "memory" },
    ]);
    const results = rawSearch("vegan diet", 3);
    expect(mockMemorySearch).toHaveBeenCalledWith("vegan diet", 3);
    expect(results).toHaveLength(1);
    expect(results[0].path).toBe(".reflexio/profiles/diet.md");
  });

  it("filters results to specified type", () => {
    mockMemorySearch.mockReturnValue([
      { path: ".reflexio/profiles/diet.md", score: 0.5, snippet: "x", startLine: 1, endLine: 5, source: "memory" },
      { path: ".reflexio/playbooks/commit.md", score: 0.4, snippet: "y", startLine: 1, endLine: 5, source: "memory" },
    ]);
    const results = rawSearch("query", 5, "profile");
    expect(results).toHaveLength(1);
    expect(results[0].path).toContain("profiles");
  });

  it("returns empty on memorySearch failure", () => {
    mockMemorySearch.mockReturnValue([]);
    expect(rawSearch("anything")).toEqual([]);
  });
});

describe("search", () => {
  it("preprocesses query before searching", () => {
    mockInfer.mockReturnValue("Rewritten query about diet");
    mockMemorySearch.mockReturnValue([]);
    search("Oh sorry I like vegan food");
    expect(mockInfer).toHaveBeenCalledOnce();
    expect(mockMemorySearch).toHaveBeenCalledWith("Rewritten query about diet", 5);
  });

  it("falls back to raw query if preprocessing fails", () => {
    mockInfer.mockReturnValue(null);
    mockMemorySearch.mockReturnValue([]);
    search("raw query here");
    expect(mockMemorySearch).toHaveBeenCalledWith("raw query here", 5);
  });
});
