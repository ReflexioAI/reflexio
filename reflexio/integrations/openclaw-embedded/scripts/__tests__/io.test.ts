import { describe, it, expect } from "vitest";
import { generateNanoid, validateSlug } from "../lib/io.js";

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
