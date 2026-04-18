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

export function validateTtl(ttl: string): asserts ttl is Ttl {
  if (!VALID_TTLS.includes(ttl as Ttl)) {
    throw new Error(
      `Invalid TTL: "${ttl}". Must be one of: ${VALID_TTLS.join(", ")}`
    );
  }
}

export function computeExpires(ttl: Ttl, created: string): string {
  if (ttl === "infinity") return "never";
  const date = new Date(created);
  const offsets: Record<string, () => void> = {
    one_day: () => date.setUTCDate(date.getUTCDate() + 1),
    one_week: () => date.setUTCDate(date.getUTCDate() + 7),
    one_month: () => date.setUTCMonth(date.getUTCMonth() + 1),
    one_quarter: () => date.setUTCMonth(date.getUTCMonth() + 3),
    one_year: () => date.setUTCFullYear(date.getUTCFullYear() + 1),
  };
  offsets[ttl]();
  return date.toISOString().slice(0, 10);
}

export function resolveWorkspace(): string {
  return process.env.WORKSPACE || process.cwd();
}

interface WriteProfileOpts {
  slug: string;
  ttl: Ttl;
  body: string;
  supersedes?: string[];
  workspace?: string;
}

export function writeProfileFile(opts: WriteProfileOpts): string {
  validateSlug(opts.slug);
  validateTtl(opts.ttl);

  const suffix = generateNanoid();
  const id = `prof_${suffix}`;
  const created = new Date().toISOString().replace(/\.\d+Z$/, "Z");
  const expires = computeExpires(opts.ttl, created);

  const ws = opts.workspace || resolveWorkspace();
  const dir = path.join(ws, ".reflexio", "profiles");
  fs.mkdirSync(dir, { recursive: true });

  const filePath = path.join(dir, `${opts.slug}-${suffix}.md`);
  const tmpPath = `${filePath}.tmp.${process.pid}`;

  const lines = [
    "---",
    "type: profile",
    `id: ${id}`,
    `created: ${created}`,
    `ttl: ${opts.ttl}`,
    `expires: ${expires}`,
  ];
  if (opts.supersedes && opts.supersedes.length > 0) {
    lines.push(`supersedes: [${opts.supersedes.join(", ")}]`);
  }
  lines.push("---", "", opts.body, "");

  try {
    fs.writeFileSync(tmpPath, lines.join("\n"));
    fs.renameSync(tmpPath, filePath);
  } catch (err) {
    try { fs.unlinkSync(tmpPath); } catch {}
    throw err;
  }

  return filePath;
}

interface WritePlaybookOpts {
  slug: string;
  body: string;
  supersedes?: string[];
  workspace?: string;
}

export function writePlaybookFile(opts: WritePlaybookOpts): string {
  validateSlug(opts.slug);

  const suffix = generateNanoid();
  const id = `pbk_${suffix}`;
  const created = new Date().toISOString().replace(/\.\d+Z$/, "Z");

  const ws = opts.workspace || resolveWorkspace();
  const dir = path.join(ws, ".reflexio", "playbooks");
  fs.mkdirSync(dir, { recursive: true });

  const filePath = path.join(dir, `${opts.slug}-${suffix}.md`);
  const tmpPath = `${filePath}.tmp.${process.pid}`;

  const lines = [
    "---",
    "type: playbook",
    `id: ${id}`,
    `created: ${created}`,
  ];
  if (opts.supersedes && opts.supersedes.length > 0) {
    lines.push(`supersedes: [${opts.supersedes.join(", ")}]`);
  }
  lines.push("---", "", opts.body, "");

  try {
    fs.writeFileSync(tmpPath, lines.join("\n"));
    fs.renameSync(tmpPath, filePath);
  } catch (err) {
    try { fs.unlinkSync(tmpPath); } catch {}
    throw err;
  }

  return filePath;
}

export function deleteFile(filePath: string): void {
  try {
    fs.unlinkSync(filePath);
  } catch (err: any) {
    if (err.code !== "ENOENT") throw err;
    console.error(`[reflexio] warning: file already gone: ${filePath}`);
  }
}
