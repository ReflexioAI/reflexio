import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

vi.mock("../lib/openclaw-cli.js", () => ({
  memorySearch: vi.fn(),
  infer: vi.fn(),
}));

import { writeProfile } from "../lib/write-profile.js";
import { memorySearch, infer } from "../lib/openclaw-cli.js";

const mockMemorySearch = vi.mocked(memorySearch);
const mockInfer = vi.mocked(infer);

let workspace: string;

beforeEach(() => {
  vi.clearAllMocks();
  workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-wp-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "profiles"), { recursive: true });
});

afterEach(() => {
  fs.rmSync(workspace, { recursive: true, force: true });
});

describe("writeProfile", () => {
  it("writes normally when no neighbors found", () => {
    mockInfer.mockReturnValue("diet vegan query");
    mockMemorySearch.mockReturnValue([]);

    const result = writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("User is vegan.");
    expect(content).not.toContain("supersedes");
  });

  it("writes normally when neighbor is below threshold", () => {
    mockInfer.mockReturnValue("diet query");
    mockMemorySearch.mockReturnValue([
      { path: ".reflexio/profiles/old.md", score: 0.3, snippet: "id: prof_old\n---\nOld fact", startLine: 1, endLine: 5, source: "memory" },
    ]);

    const result = writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).not.toContain("supersedes");
  });

  it("supersedes when neighbor above threshold and LLM says supersede", () => {
    mockInfer
      .mockReturnValueOnce("diet vegan query")    // preprocessQuery
      .mockReturnValueOnce('{"decision": "supersede"}');  // judgeContradiction

    const oldPath = path.join(workspace, ".reflexio", "profiles", "old.md");
    fs.writeFileSync(oldPath, "---\nid: prof_old\n---\nOld fact");

    mockMemorySearch.mockReturnValue([
      { path: oldPath, score: 0.5, snippet: "---\nid: prof_old\n---\nOld fact", startLine: 1, endLine: 5, source: "memory" },
    ]);

    const result = writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    // New file exists with supersedes
    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).toContain("supersedes: [prof_old]");

    // Old file deleted
    expect(fs.existsSync(oldPath)).toBe(false);
  });

  it("keeps both when LLM says keep_both", () => {
    mockInfer
      .mockReturnValueOnce("query")
      .mockReturnValueOnce('{"decision": "keep_both"}');

    const oldPath = path.join(workspace, ".reflexio", "profiles", "old.md");
    fs.writeFileSync(oldPath, "---\nid: prof_old\n---\nDifferent fact");

    mockMemorySearch.mockReturnValue([
      { path: oldPath, score: 0.5, snippet: "---\nid: prof_old\n---\nDifferent fact", startLine: 1, endLine: 5, source: "memory" },
    ]);

    const result = writeProfile({
      slug: "new-fact", ttl: "infinity",
      body: "New fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.existsSync(oldPath)).toBe(true);  // Old file preserved
    expect(fs.readFileSync(result, "utf8")).not.toContain("supersedes");
  });

  it("still writes when openclaw infer fails at preprocessing", () => {
    mockInfer.mockReturnValue(null);
    mockMemorySearch.mockReturnValue([]);

    const result = writeProfile({
      slug: "test", ttl: "infinity",
      body: "Fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
  });

  it("still writes when openclaw memory search fails", () => {
    mockInfer.mockReturnValue("query");
    mockMemorySearch.mockReturnValue([]);

    const result = writeProfile({
      slug: "test", ttl: "infinity",
      body: "Fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
  });

  it("throws on invalid slug", () => {
    expect(() =>
      writeProfile({
        slug: "INVALID", ttl: "infinity",
        body: "x", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      })
    ).toThrow("Invalid slug");
  });

  it("throws on invalid TTL", () => {
    expect(() =>
      writeProfile({
        slug: "valid", ttl: "bad_ttl" as any,
        body: "x", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      })
    ).toThrow("Invalid TTL");
  });
});
