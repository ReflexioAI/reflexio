import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { writeProfile } from "../lib/write-profile.js";
import type { CommandRunner, MemorySearchResult } from "../lib/openclaw-cli.js";

let inferCallCount: number;

function createMockRunner(
  inferResults: (string | null)[],
  searchResults: MemorySearchResult[]
): CommandRunner {
  inferCallCount = 0;
  return async (argv) => {
    if (argv.includes("infer")) {
      const result = inferResults[inferCallCount++] ?? null;
      if (result === null) throw new Error("infer failed");
      return { stdout: result, stderr: "", code: 0 };
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

let workspace: string;

beforeEach(() => {
  workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-wp-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "profiles"), { recursive: true });
});

afterEach(() => {
  fs.rmSync(workspace, { recursive: true, force: true });
});

describe("writeProfile", () => {
  it("writes normally when no neighbors found", async () => {
    const runner = createMockRunner(["diet vegan query"], []);

    const result = await writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("User is vegan.");
    expect(content).not.toContain("supersedes");
  });

  it("writes normally when neighbor is below threshold", async () => {
    const runner = createMockRunner(["diet query"], [
      { path: ".reflexio/profiles/old.md", score: 0.3, snippet: "id: prof_old\n---\nOld fact", startLine: 1, endLine: 5, source: "memory" },
    ]);

    const result = await writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).not.toContain("supersedes");
  });

  it("supersedes when neighbor above threshold and LLM says supersede", async () => {
    const oldPath = path.join(workspace, ".reflexio", "profiles", "old.md");
    fs.writeFileSync(oldPath, "---\nid: prof_old\n---\nOld fact");

    const runner = createMockRunner(
      ["diet vegan query", '{"decision": "supersede"}'],
      [{ path: oldPath, score: 0.5, snippet: "---\nid: prof_old\n---\nOld fact", startLine: 1, endLine: 5, source: "memory" }]
    );

    const result = await writeProfile({
      slug: "diet-vegan", ttl: "infinity",
      body: "User is vegan.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).toContain("supersedes: [prof_old]");
    expect(fs.existsSync(oldPath)).toBe(false);
  });

  it("keeps both when LLM says keep_both", async () => {
    const oldPath = path.join(workspace, ".reflexio", "profiles", "old.md");
    fs.writeFileSync(oldPath, "---\nid: prof_old\n---\nDifferent fact");

    const runner = createMockRunner(
      ["query", '{"decision": "keep_both"}'],
      [{ path: oldPath, score: 0.5, snippet: "---\nid: prof_old\n---\nDifferent fact", startLine: 1, endLine: 5, source: "memory" }]
    );

    const result = await writeProfile({
      slug: "new-fact", ttl: "infinity",
      body: "New fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.existsSync(oldPath)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).not.toContain("supersedes");
  });

  it("still writes when openclaw infer fails at preprocessing", async () => {
    const runner = createMockRunner([null], []);

    const result = await writeProfile({
      slug: "test", ttl: "infinity",
      body: "Fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
  });

  it("still writes when openclaw memory search fails", async () => {
    const runner = createMockRunner(["query"], []);

    const result = await writeProfile({
      slug: "test", ttl: "infinity",
      body: "Fact.", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
  });

  it("throws on invalid slug", async () => {
    const runner = createMockRunner([], []);
    await expect(
      writeProfile({
        slug: "INVALID", ttl: "infinity",
        body: "x", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
        runner,
      })
    ).rejects.toThrow("Invalid slug");
  });

  it("throws on invalid TTL", async () => {
    const runner = createMockRunner([], []);
    await expect(
      writeProfile({
        slug: "valid", ttl: "bad_ttl" as any,
        body: "x", workspace, config: { shallow_threshold: 0.4, top_k: 5 },
        runner,
      })
    ).rejects.toThrow("Invalid TTL");
  });
});
