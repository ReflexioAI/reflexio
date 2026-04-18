import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { generateNanoid, validateSlug, validateTtl, computeExpires, writeProfileFile, writePlaybookFile, deleteFile } from "../plugin/lib/io.ts";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

describe("generateNanoid", () => {
  it("returns a 4-character string of [a-z0-9]", () => {
    const id = generateNanoid();
    expect(id).toMatch(/^[a-z0-9]{4}$/);
  });

  it("produces different values across calls", () => {
    const ids = new Set(Array.from({ length: 10 }, () => generateNanoid()));
    expect(ids.size).toBeGreaterThan(1);
  });
});

describe("validateSlug", () => {
  it("accepts valid kebab-case slugs", () => {
    expect(() => validateSlug("diet-vegetarian")).not.toThrow();
    expect(() => validateSlug("abc")).not.toThrow();
    expect(() => validateSlug("a1b2")).not.toThrow();
  });

  it("rejects empty string", () => {
    expect(() => validateSlug("")).toThrow();
  });

  it("rejects uppercase", () => {
    expect(() => validateSlug("Diet-Vegetarian")).toThrow();
  });

  it("rejects leading hyphen", () => {
    expect(() => validateSlug("-diet")).toThrow();
  });

  it("rejects slashes", () => {
    expect(() => validateSlug("foo/bar")).toThrow();
  });

  it("rejects strings longer than 48 chars", () => {
    expect(() => validateSlug("a".repeat(49))).toThrow();
  });
});

describe("validateTtl", () => {
  it("accepts all valid TTL values", () => {
    for (const ttl of ["one_day", "one_week", "one_month", "one_quarter", "one_year", "infinity"]) {
      expect(() => validateTtl(ttl as any)).not.toThrow();
    }
  });

  it("rejects invalid TTL", () => {
    expect(() => validateTtl("one_millennium" as any)).toThrow();
  });
});

describe("computeExpires", () => {
  it("returns 'never' for infinity", () => {
    expect(computeExpires("infinity", "2026-04-17T00:00:00Z")).toBe("never");
  });

  it("returns a date string for one_year", () => {
    const result = computeExpires("one_year", "2026-04-17T00:00:00Z");
    expect(result).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(result).toBe("2027-04-17");
  });

  it("returns correct date for one_day", () => {
    expect(computeExpires("one_day", "2026-04-17T00:00:00Z")).toBe("2026-04-18");
  });
});

describe("writeProfileFile", () => {
  let workspace: string;

  beforeEach(() => {
    workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-test-"));
    fs.mkdirSync(path.join(workspace, ".reflexio", "profiles"), { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(workspace, { recursive: true, force: true });
  });

  it("creates a profile file with correct frontmatter", () => {
    const result = writeProfileFile({
      slug: "diet-vegan",
      ttl: "infinity",
      body: "User is vegan.",
      workspace,
    });
    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("type: profile");
    expect(content).toContain("id: prof_");
    expect(content).toContain("ttl: infinity");
    expect(content).toContain("expires: never");
    expect(content).toContain("User is vegan.");
  });

  it("includes supersedes when provided", () => {
    const result = writeProfileFile({
      slug: "diet-vegan",
      ttl: "infinity",
      body: "User is vegan.",
      supersedes: ["prof_abc1"],
      workspace,
    });
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("supersedes: [prof_abc1]");
  });

  it("omits supersedes when not provided", () => {
    const result = writeProfileFile({
      slug: "diet-vegan",
      ttl: "infinity",
      body: "User is vegan.",
      workspace,
    });
    const content = fs.readFileSync(result, "utf8");
    expect(content).not.toContain("supersedes");
  });

  it("leaves no .tmp files on success", () => {
    writeProfileFile({ slug: "test", ttl: "infinity", body: "x", workspace });
    const tmps = fs.readdirSync(path.join(workspace, ".reflexio", "profiles"))
      .filter((f) => f.includes(".tmp"));
    expect(tmps).toHaveLength(0);
  });
});

describe("writePlaybookFile", () => {
  let workspace: string;

  beforeEach(() => {
    workspace = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-test-"));
    fs.mkdirSync(path.join(workspace, ".reflexio", "playbooks"), { recursive: true });
  });

  afterEach(() => {
    fs.rmSync(workspace, { recursive: true, force: true });
  });

  it("creates a playbook file with correct frontmatter (no ttl/expires)", () => {
    const result = writePlaybookFile({
      slug: "commit-no-trailers",
      body: "## When\nCommit message.\n\n## What\nNo trailers.\n\n## Why\nUser said so.",
      workspace,
    });
    expect(fs.existsSync(result)).toBe(true);
    const content = fs.readFileSync(result, "utf8");
    expect(content).toContain("type: playbook");
    expect(content).toContain("id: pbk_");
    expect(content).not.toContain("ttl:");
    expect(content).not.toContain("expires:");
    expect(content).toContain("## When");
  });
});

describe("deleteFile", () => {
  it("deletes an existing file", () => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "rfx-del-"));
    const f = path.join(tmp, "test.md");
    fs.writeFileSync(f, "x");
    deleteFile(f);
    expect(fs.existsSync(f)).toBe(false);
    fs.rmSync(tmp, { recursive: true, force: true });
  });

  it("does not throw when file is missing", () => {
    expect(() => deleteFile("/tmp/nonexistent-reflexio-test.md")).not.toThrow();
  });
});
