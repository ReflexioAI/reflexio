// Render search hits as the markdown block injected before a turn, plus a
// registry mapping each `[oc:…]` tag back to the stored learning.
//
// Tags are `{p|s}-{hash}`, the hash taken over the learning's kind and full
// id. A rank-plus-id-prefix tag collided: ids 10001 and 10002 both rendered as
// `s1-1000`, and an agent and a user playbook sharing a numeric id rendered
// identically, so a citation resolved to whichever was injected last. Hashing
// the identity also keeps one learning's tag stable across turns.

import { createHash } from "node:crypto";
import type {
  Citation,
  Playbook,
  Profile,
  RetrievedLearning,
  SearchHits,
} from "./client.js";

export interface RegistryEntry {
  tag: string;
  kind: RetrievedLearning["kind"];
  realId?: string;
  title: string;
  exposureReceipt?: string;
}

const RECEIPT_MAX = 8192;

export const CITATION_INSTRUCTION =
  "_First, fully answer the user — citation does not change what or how you " +
  "reply. Then, if — and only if — an injected `[oc:…]` item materially " +
  "changed your reply, append exactly one final line: " +
  "`✨ 1 Reflexio learning applied [oc:s-1a2b3c]` (or " +
  "`✨ 2 Reflexio learnings applied [oc:s-1a2b3c,p-4d5e6f]`). Copy ids verbatim " +
  "from the tags. If an item is merely on-topic, do not cite. When unsure, skip._";

const CITATION_LINE =
  /^\s*✨\s+\d+\s+(?:Reflexio|openclaw-smart) learnings? applied\s+\[oc:([^\]]+)\]\s*$/gim;
const TAG = /^(?:oc:)?([ps]-[0-9a-f]{6})$/i;

function tagFor(prefix: "p" | "s", kind: RetrievedLearning["kind"], identity: string): string {
  return `${prefix}-${createHash("sha256").update(`${kind}\u0000${identity}`).digest("hex").slice(0, 6)}`;
}

function titleOf(content: string, limit = 80): string {
  const text = content.trim();
  for (const terminator of [". ", "\n"]) {
    const idx = text.indexOf(terminator);
    if (idx > 0 && idx <= limit) return text.slice(0, idx).trimEnd();
  }
  return text.length <= limit ? text : `${text.slice(0, limit - 1).trimEnd()}…`;
}

function idOf(value: unknown): string | undefined {
  return value === undefined || value === null || value === "" ? undefined : String(value);
}

function receiptOf(pb: Playbook): string | undefined {
  const r = pb.exposure_receipt;
  return typeof r === "string" && r.length <= RECEIPT_MAX ? r : undefined;
}

export function renderInjection(hits: SearchHits): {
  markdown: string;
  registry: RegistryEntry[];
} {
  const registry: RegistryEntry[] = [];
  const playbookLines: string[] = [];
  const profileLines: string[] = [];

  const addPlaybook = (pb: Playbook, kind: "agent_playbook" | "user_playbook") => {
    const content = (pb.content ?? "").trim();
    if (!content) return;
    const realId = idOf(kind === "agent_playbook" ? pb.agent_playbook_id : pb.user_playbook_id);
    const tag = tagFor("s", kind, realId ?? content);
    let line = `- [oc:${tag}] ${content}`;
    if (pb.trigger?.trim()) line += ` _(when: ${pb.trigger.trim()})_`;
    if (pb.rationale?.trim()) line += ` — *why:* ${pb.rationale.trim()}`;
    playbookLines.push(line);
    registry.push({ tag, kind, realId, title: titleOf(content), exposureReceipt: receiptOf(pb) });
  };
  // Agent playbooks (shared, distilled) rank ahead of this project's own.
  for (const pb of hits.agent_playbooks) addPlaybook(pb, "agent_playbook");
  for (const pb of hits.user_playbooks) addPlaybook(pb, "user_playbook");

  for (const p of hits.profiles as Profile[]) {
    const content = (p.content ?? "").trim();
    if (!content) continue;
    const realId = idOf(p.profile_id);
    const tag = tagFor("p", "profile", realId ?? content);
    profileLines.push(`- [oc:${tag}] ${content}`);
    registry.push({ tag, kind: "profile", realId, title: titleOf(content) });
  }

  if (!playbookLines.length && !profileLines.length) return { markdown: "", registry: [] };
  const sections: string[] = [];
  if (playbookLines.length) sections.push("### Relevant skills (Reflexio)", ...playbookLines);
  if (profileLines.length) sections.push("### Relevant preferences (Reflexio)", ...profileLines);
  sections.push(CITATION_INSTRUCTION);
  return { markdown: `${sections.join("\n")}\n`, registry };
}

export function retrievedLearnings(registry: RegistryEntry[]): RetrievedLearning[] {
  const out: RetrievedLearning[] = [];
  const seen = new Set<string>();
  for (const e of registry) {
    if (!e.realId) continue;
    const key = `${e.kind}\u0000${e.realId}\u0000${e.exposureReceipt ?? ""}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({
      kind: e.kind,
      learning_id: e.realId,
      ...(e.exposureReceipt ? { exposure_receipt: e.exposureReceipt } : {}),
    });
  }
  return out;
}

// Only the final marker line counts, so mentioning a tag inside an answer is
// not a citation. Unknown tags (hallucinated, or from an older injection that
// was evicted) are dropped.
export function resolveCitations(text: string, registry: RegistryEntry[]): Citation[] {
  const matches = [...text.matchAll(CITATION_LINE)];
  const last = matches.at(-1);
  if (!last?.[1]) return [];
  const byTag = new Map(registry.map((e) => [e.tag.toLowerCase(), e]));
  const out: Citation[] = [];
  const seen = new Set<string>();
  for (const token of last[1].split(/[,\s]+/)) {
    const tag = TAG.exec(token.trim())?.[1]?.toLowerCase();
    const entry = tag ? byTag.get(tag) : undefined;
    if (!entry?.realId || seen.has(entry.tag)) continue;
    seen.add(entry.tag);
    out.push({ kind: entry.kind, real_id: entry.realId, tag: entry.tag, title: entry.title });
  }
  return out;
}
