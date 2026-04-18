import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as path from "node:path";

const SLUG_REGEX = /^[a-z0-9][a-z0-9-]{0,47}$/;
const VALID_TTLS = [
  "one_day",
  "one_week",
  "one_month",
  "one_quarter",
  "one_year",
  "infinity",
] as const;
export type Ttl = (typeof VALID_TTLS)[number];

export function generateNanoid(): string {
  const bytes = crypto.randomBytes(3);
  return Array.from(bytes)
    .map((b) => (b % 36).toString(36))
    .join("")
    .slice(0, 4)
    .padEnd(4, "0");
}

export function validateSlug(slug: string): void {
  if (!slug || !SLUG_REGEX.test(slug)) {
    throw new Error(`Invalid slug: "${slug}". Must match ${SLUG_REGEX}`);
  }
}
