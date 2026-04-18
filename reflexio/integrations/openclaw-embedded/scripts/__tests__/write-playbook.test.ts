import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

vi.mock("../lib/openclaw-cli.js", () => ({
  memorySearch: vi.fn(),
  infer: vi.fn(),
}));

import { writePlaybook } from "../lib/write-playbook.js";
import { memorySearch, infer } from "../lib/openclaw-cli.js";

const mockMemorySearch = vi.mocked(memorySearch);
const mockInfer = vi.mocked(infer);

let workspace: string;

beforeEach(() => {
  vi.clearAllMocks();
  workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-wpb-"));
  fs.mkdirSync(path.join(workspace, ".reflexio", "playbooks"), { recursive: true });
});

afterEach(() => {
  fs.rmSync(workspace, { recursive: true, force: true });
});

describe("writePlaybook", () => {
  it("writes normally when no neighbors found", () => {
    mockInfer.mockReturnValue("commit message query");
    mockMemorySearch.mockReturnValue([]);

    const result = writePlaybook({
      slug: "commit-no-trailers",
      body: "## When\nCommit.\n\n## What\nNo trailers.\n\n## Why\nUser said.",
      workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("type: playbook");
    expect(content).toContain("## When");
  });

  it("supersedes when neighbor above threshold and LLM says supersede", () => {
    mockInfer
      .mockReturnValueOnce("commit query")
      .mockReturnValueOnce('{"decision": "supersede"}');

    const oldPath = path.join(workspace, ".reflexio", "playbooks", "old.md");
    fs.writeFileSync(oldPath, "---\nid: pbk_old\n---\nOld playbook");

    mockMemorySearch.mockReturnValue([
      { path: oldPath, score: 0.5, snippet: "---\nid: pbk_old\n---\nOld playbook", startLine: 1, endLine: 5, source: "memory" },
    ]);

    const result = writePlaybook({
      slug: "commit-no-trailers",
      body: "## When\nCommit.\n\n## What\nUpdated rule.",
      workspace, config: { shallow_threshold: 0.4, top_k: 5 },
    });

    expect(fs.existsSync(result)).toBe(true);
    expect(fs.readFileSync(result, "utf8")).toContain("supersedes: [pbk_old]");
    expect(fs.existsSync(oldPath)).toBe(false);
  });

  it("throws on invalid slug", () => {
    expect(() =>
      writePlaybook({
        slug: "INVALID", body: "x",
        workspace, config: { shallow_threshold: 0.4, top_k: 5 },
      })
    ).toThrow("Invalid slug");
  });
});
