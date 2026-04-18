import { describe, it, expect } from "vitest";

import { preprocessQuery, judgeContradiction, extractId } from "../lib/dedup.js";
import type { CommandRunner } from "../lib/openclaw-cli.js";

function createMockRunner(inferResult: string | null): CommandRunner {
  return async (argv) => {
    if (argv.includes("infer")) {
      if (inferResult === null) throw new Error("infer failed");
      return { stdout: inferResult, stderr: "", code: 0 };
    }
    return { stdout: "", stderr: "unexpected command", code: 1 };
  };
}

describe("preprocessQuery", () => {
  it("returns LLM-rewritten query on success", async () => {
    const runner = createMockRunner(
      "User dietary preference vegan. Related: plant-based, no animal products"
    );
    const result = await preprocessQuery("Oh sorry I typed it wrong, I do like vegan food", runner);
    expect(result).toBe(
      "User dietary preference vegan. Related: plant-based, no animal products"
    );
  });

  it("falls back to raw text when infer fails", async () => {
    const runner = createMockRunner(null);
    const raw = "I like apple juice";
    const result = await preprocessQuery(raw, runner);
    expect(result).toBe(raw);
  });

  it("falls back to raw text when infer returns empty string", async () => {
    const runner = createMockRunner("");
    const raw = "timezone is PST";
    const result = await preprocessQuery(raw, runner);
    expect(result).toBe(raw);
  });
});

describe("judgeContradiction", () => {
  it("returns 'supersede' when LLM says supersede", async () => {
    const runner = createMockRunner('{"decision": "supersede"}');
    const result = await judgeContradiction("User is vegan", "User is pescatarian", runner);
    expect(result).toBe("supersede");
  });

  it("returns 'keep_both' when LLM says keep_both", async () => {
    const runner = createMockRunner('{"decision": "keep_both"}');
    const result = await judgeContradiction("User likes dark mode", "User is a developer", runner);
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' when infer fails", async () => {
    const runner = createMockRunner(null);
    const result = await judgeContradiction("A", "B", runner);
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' on malformed JSON", async () => {
    const runner = createMockRunner("I think they are related");
    const result = await judgeContradiction("A", "B", runner);
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' on unexpected decision value", async () => {
    const runner = createMockRunner('{"decision": "merge"}');
    const result = await judgeContradiction("A", "B", runner);
    expect(result).toBe("keep_both");
  });
});

describe("extractId", () => {
  it("extracts prof_ id from snippet", () => {
    const snippet = "---\ntype: profile\nid: prof_sdtk\ncreated: 2026\n---\nContent";
    expect(extractId(snippet)).toBe("prof_sdtk");
  });

  it("extracts pbk_ id from snippet", () => {
    const snippet = "---\ntype: playbook\nid: pbk_az4k\n---\nContent";
    expect(extractId(snippet)).toBe("pbk_az4k");
  });

  it("returns null when no id found", () => {
    expect(extractId("no frontmatter here")).toBeNull();
  });
});
