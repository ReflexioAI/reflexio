// Turn the transcript OpenClaw hands to agent_end into the interactions for
// the run that just finished.
//
// agent_end receives the whole session history. before_prompt_build sees the
// history *before* the run's prompt, so its length is where the run starts;
// handlers record it per runId. Without it (or when compaction shrank the
// history mid-run) the run is taken to start at the last message the user
// wrote. Steering messages the user sends during a run stay part of it.
//
// Injected context never appears here: OpenClaw keeps it on
// `__openclaw.modelPromptProjection`, separate from the user's own text.

import type { ToolUsed } from "./client.js";
import { DEFAULT_MAX_CONTENT_LENGTH } from "./config.js";
import { characterPrefix } from "./text.js";

const TOOL_STRING_MAX = 256;
const TOOL_DATA_MAX = 2048;

type Block = { type?: unknown; text?: unknown; name?: unknown; id?: unknown; arguments?: unknown };
type Message = {
  role?: unknown;
  content?: unknown;
  toolCallId?: unknown;
  isError?: unknown;
  runtimeContext?: unknown;
  runtimeContextCarrier?: unknown;
  operatorMessage?: unknown;
};

export interface TurnContent {
  user: string;
  assistant: string;
  tools: ToolUsed[];
  // The whole reply, never published: citations are read from its final
  // line, which clipping `assistant` would cut off on a long answer.
  fullAssistant: string;
}

// Credentials that show up in tool arguments and output. The value group is
// what gets masked; the prefix (key, flag, scheme) is kept for readability.
const SECRET_PATTERNS: RegExp[] = [
  // Well-known token shapes, wherever they appear.
  /()(\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}))/g,
  // Authorization: Bearer <token>
  /(\b(?:Bearer|Basic|Token)\s+)([A-Za-z0-9._~+/=-]{12,})/gi,
  // KEY=value / key: value / "key": "value" / --key=value, for credential-ish names.
  // The name must END in the credential word, so `authorName` or `keyboard`
  // are left alone.
  /((?:--?|\b)[A-Za-z0-9_.-]*(?:key|token|secret|passw(?:or)?d|pwd|credentials?|authorization)["']?\s*[:=]\s*["']?)([^\s"',;&|]{8,})/gi,
];

// Fallback for credentials no name or shape above recognises
// (GITLAB_PAT=glpat-…, a vendor's own prefix): an upper-case environment
// assignment whose value mixes upper and lower case letters and digits. Hex
// digests and ordinary words lack one of the three and are left alone.
const ENV_ASSIGNMENT = /(\b[A-Z][A-Z0-9_]{2,}=["']?)([A-Za-z0-9+/=_-]{20,})/g;

function looksLikeSecret(value: string): boolean {
  return /[a-z]/.test(value) && /[A-Z]/.test(value) && /[0-9]/.test(value);
}

export function redact(text: string): string {
  let out = text;
  for (const pattern of SECRET_PATTERNS) {
    out = out.replace(pattern, (_whole, prefix: string, value: string) => `${prefix}<redacted:${value.length}>`);
  }
  return out.replace(ENV_ASSIGNMENT, (whole, prefix: string, value: string) =>
    looksLikeSecret(value) ? `${prefix}<redacted:${value.length}>` : whole,
  );
}

function clip(text: string, max: number): string {
  const { end } = characterPrefix(text, max);
  return end < text.length ? `${text.slice(0, end)}…(truncated)` : text;
}

// A key that names a credential (DATABASE_PASSWORD, apiKey, x-auth-token,
// Authorization, Cookie). Its value is masked whatever it looks like: a
// structured value carries no "KEY=" prefix for the text patterns to see.
const CREDENTIAL_KEY = /(?:key|token|secret|passw(?:or)?d|pwd|credentials?|authorization|cookie)$/i;

function mask(value: unknown): unknown {
  return typeof value === "string" ? `<redacted:${value.length}>` : "<redacted>";
}

// Redact and clip every string at any depth; tool payloads nest (env maps,
// header objects, edit lists), and a top-level-only pass let nested secrets
// through.
function scrub(value: unknown, depth = 0): unknown {
  if (typeof value === "string") return clip(redact(value), TOOL_STRING_MAX);
  if (value === null || typeof value !== "object") return value;
  // Past the depth limit nothing is inspected, so nothing may pass through.
  if (depth > 6) return "<truncated>";
  if (Array.isArray(value)) return value.slice(0, 50).map((v) => scrub(v, depth + 1));
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .slice(0, 50)
      .map(([k, v]) => [k, CREDENTIAL_KEY.test(k.replace(/[-_\s]/g, "")) ? mask(v) : scrub(v, depth + 1)]),
  );
}

function bounded(data: Record<string, unknown>): Record<string, unknown> {
  const json = JSON.stringify(data);
  return json.length <= TOOL_DATA_MAX ? data : { summary: clip(json, TOOL_DATA_MAX) };
}

function blocks(content: unknown): Block[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  return Array.isArray(content) ? (content as Block[]) : [];
}

function textOf(content: unknown): string {
  const parts: string[] = [];
  for (const b of blocks(content)) {
    if (b.type === "text" && typeof b.text === "string" && b.text.trim()) {
      // OpenClaw occasionally repeats the same text block; keep one copy.
      if (parts.at(-1) !== b.text) parts.push(b.text);
    }
  }
  return parts.join("\n\n");
}

// Host-authored `user` messages: runtime context (and its pre-2026.9.7
// carrier form) and operator notes are not something the user said.
function isUserWritten(m: Message): boolean {
  return (
    m.role === "user" &&
    m.runtimeContext === undefined &&
    m.runtimeContextCarrier === undefined &&
    m.operatorMessage === undefined
  );
}

// Where the run begins, given the history before_prompt_build saw. That
// history usually ends just before the run's prompt, but on some host paths
// (the Agents API) it already includes the admitted user message; then the
// run starts at that message, not after it.
export function runStartIn(history: unknown[], prompt: string): number {
  const last = history.at(-1);
  if (typeof last === "object" && last !== null && isUserWritten(last as Message)) {
    if (textOf((last as Message).content).trim() === prompt.trim()) return history.length - 1;
  }
  return history.length;
}

function runStart(list: Message[], recordedStart: number | undefined): number {
  if (recordedStart !== undefined && recordedStart < list.length && isUserWritten(list[recordedStart] as Message)) {
    return recordedStart;
  }
  for (let i = list.length - 1; i >= 0; i--) {
    if (isUserWritten(list[i] as Message)) return i;
  }
  return -1;
}

export function currentTurn(messages: unknown[], recordedStart?: number, maxContentLength = DEFAULT_MAX_CONTENT_LENGTH): TurnContent | undefined {
  const list = messages.filter((m): m is Message => typeof m === "object" && m !== null);
  const start = runStart(list, recordedStart);
  if (start < 0) return undefined;

  const userParts: string[] = [];
  const assistantParts: string[] = [];
  const tools = new Map<string, ToolUsed>();
  for (const m of list.slice(start)) {
    if (isUserWritten(m)) {
      const text = textOf(m.content);
      if (text) userParts.push(text);
    } else if (m.role === "assistant") {
      const text = textOf(m.content);
      if (text) assistantParts.push(text);
      for (const b of blocks(m.content)) {
        if (b.type !== "toolCall" || typeof b.name !== "string") continue;
        tools.set(String(b.id ?? tools.size), {
          tool_name: b.name,
          status: "success",
          tool_data: { input: scrub(b.arguments ?? {}) },
        });
      }
    } else if (m.role === "toolResult") {
      const tool = tools.get(String(m.toolCallId));
      if (!tool) continue;
      if (m.isError === true) tool.status = "error";
      const output = textOf(m.content);
      if (output) tool.tool_data = { ...tool.tool_data, output: scrub(output) };
    }
  }
  const user = userParts.join("\n\n");
  const assistant = assistantParts.join("\n\n");
  if (!user && !assistant) return undefined;
  return {
    user: clip(redact(user), maxContentLength),
    assistant: clip(redact(assistant), maxContentLength),
    fullAssistant: assistant,
    tools: [...tools.values()].map((t) => ({ ...t, tool_data: bounded(t.tool_data ?? {}) })),
  };
}
