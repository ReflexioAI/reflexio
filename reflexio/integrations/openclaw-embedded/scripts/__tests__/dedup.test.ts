import { describe, it, expect, vi, beforeEach } from "vitest";

// Mock openclaw-cli before importing dedup
vi.mock("../lib/openclaw-cli.js", () => ({
  infer: vi.fn(),
}));

import { preprocessQuery, judgeContradiction, extractId } from "../lib/dedup.js";
import { infer } from "../lib/openclaw-cli.js";

const mockInfer = vi.mocked(infer);

beforeEach(() => {
  vi.clearAllMocks();
});

describe("preprocessQuery", () => {
  it("returns LLM-rewritten query on success", () => {
    mockInfer.mockReturnValue(
      "User dietary preference vegan. Related: plant-based, no animal products"
    );
    const result = preprocessQuery("Oh sorry I typed it wrong, I do like vegan food");
    expect(result).toBe(
      "User dietary preference vegan. Related: plant-based, no animal products"
    );
    expect(mockInfer).toHaveBeenCalledOnce();
    expect(mockInfer.mock.calls[0][0]).toContain("Rewrite the following text");
  });

  it("falls back to raw text when infer fails", () => {
    mockInfer.mockReturnValue(null);
    const raw = "I like apple juice";
    const result = preprocessQuery(raw);
    expect(result).toBe(raw);
  });

  it("falls back to raw text when infer returns empty string", () => {
    mockInfer.mockReturnValue("");
    const raw = "timezone is PST";
    const result = preprocessQuery(raw);
    expect(result).toBe(raw);
  });
});

describe("judgeContradiction", () => {
  it("returns 'supersede' when LLM says supersede", () => {
    mockInfer.mockReturnValue('{"decision": "supersede"}');
    const result = judgeContradiction("User is vegan", "User is pescatarian");
    expect(result).toBe("supersede");
  });

  it("returns 'keep_both' when LLM says keep_both", () => {
    mockInfer.mockReturnValue('{"decision": "keep_both"}');
    const result = judgeContradiction("User likes dark mode", "User is a developer");
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' when infer fails", () => {
    mockInfer.mockReturnValue(null);
    const result = judgeContradiction("A", "B");
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' on malformed JSON", () => {
    mockInfer.mockReturnValue("I think they are related");
    const result = judgeContradiction("A", "B");
    expect(result).toBe("keep_both");
  });

  it("defaults to 'keep_both' on unexpected decision value", () => {
    mockInfer.mockReturnValue('{"decision": "merge"}');
    const result = judgeContradiction("A", "B");
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
