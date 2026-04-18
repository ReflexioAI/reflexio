import { describe, it, expect, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import { writePlaybook } from "../plugin/lib/write-playbook.ts";
import type { CommandRunner, MemorySearchResult } from "../plugin/lib/openclaw-cli.ts";

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
  workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-wpb-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "playbooks"), { recursive: true });
});

afterEach(() => {
  fs.rmSync(workspace, { recursive: true, force: true });
});

describe("writePlaybook", () => {
  it("writes normally when no neighbors found", async () => {
    const runner = createMockRunner(["commit message query"], []);

    const result = await writePlaybook({
      slug: "commit-no-trailers",
      body: "## When\nCommit.\n\n## What\nNo trailers.\n\n## Why\nUser said.",
      workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("type: playbook");
    expect(content).toContain("## When");
  });

  it("supersedes when neighbor above threshold and LLM says supersede", async () => {
    const oldPath = path.join(workspace, ".reflexio", "playbooks", "old.md");
    fs.writeFileSync(oldPath, "---\nid: pbk_old\n---\nOld playbook");

    const runner = createMockRunner(
      ["commit query", '{"decision": "supersede"}'],
      [{ path: oldPath, score: 0.5, snippet: "---\nid: pbk_old\n---\nOld playbook", startLine: 1, endLine: 5, source: "memory" }]
    );

    const result = await writePlaybook({
      slug: "commit-no-trailers",
      body: "## When\nCommit.\n\n## What\nUpdated rule.",
      workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      runner,
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).toContain("supersedes: [pbk_old]");
    expect(fs.existsSync(oldPath)).toBe(false);
  });

  it("throws on invalid slug", async () => {
    const runner = createMockRunner([], []);
    await expect(
      writePlaybook({
        slug: "INVALID", body: "x",
        workspace, config: { shallow_threshold: 0.4, top_k: 5 },
        runner,
      })
    ).rejects.toThrow("Invalid slug");
  });
});
