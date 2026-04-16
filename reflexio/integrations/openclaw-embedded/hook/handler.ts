import * as fs from "fs";
import * as path from "path";

/**
 * Openclaw hook event shape (best-effort typing — refine if Plugin SDK types are available).
 */
type HookEvent = {
  type: string;
  action?: string;
  sessionKey?: string;
  timestamp?: string;
  messages?: unknown[];
  context?: {
    bootstrapFiles?: Array<{ path: string; content: string }>;
    messages?: Array<{ role: string; content: string; timestamp?: string }>;
    [key: string]: unknown;
  };
};

type HookApi = {
  runtime?: {
    subagent?: {
      run: (args: {
        task: string;
        agentId?: string;
        runTimeoutSeconds?: number;
        mode?: "run" | "session";
      }) => Promise<{ runId: string; childSessionKey?: string }>;
    };
    config?: {
      load: () => Promise<Record<string, unknown>>;
    };
  };
};

/**
 * Find the workspace root. Openclaw typically runs with CWD = workspace,
 * but we look upward for a .reflexio/ marker as well.
 */
function resolveWorkspace(): string {
  // Prefer explicit env override (useful in tests)
  if (process.env.WORKSPACE) return process.env.WORKSPACE;
  // Otherwise pwd
  return process.cwd();
}

/**
 * TTL sweep: scan .reflexio/profiles/*.md and unlink expired files.
 * Cheap: filesystem + YAML frontmatter parse only. Target <50ms for dozens of files.
 */
async function ttlSweepProfiles(workspace: string): Promise<void> {
  const dir = path.join(workspace, ".reflexio", "profiles");
  if (!fs.existsSync(dir)) return;

  const today = new Date().toISOString().slice(0, 10); // YYYY-MM-DD
  const entries = await fs.promises.readdir(dir);

  for (const entry of entries) {
    if (!entry.endsWith(".md")) continue;
    const full = path.join(dir, entry);
    let contents: string;
    try {
      contents = await fs.promises.readFile(full, "utf8");
    } catch {
      continue;
    }
    const expiresMatch = /^expires:\s*(\S+)/m.exec(contents);
    if (!expiresMatch) continue;
    const expires = expiresMatch[1];
    if (expires === "never") continue;
    if (expires < today) {
      try {
        await fs.promises.unlink(full);
      } catch (err) {
        console.error(`[reflexio-embedded] ttl sweep: failed to unlink ${full}: ${err}`);
      }
    }
  }
}

/**
 * Handle agent:bootstrap — runs TTL sweep and injects reminder.
 */
async function handleBootstrap(event: HookEvent, api: HookApi, workspace: string): Promise<void> {
  await ttlSweepProfiles(workspace);

  // Inject a bootstrap reminder so the SKILL.md is prominent
  if (event.context?.bootstrapFiles && Array.isArray(event.context.bootstrapFiles)) {
    const reminder = [
      "# Reflexio Embedded",
      "",
      "This agent has the openclaw-embedded plugin installed. Its SKILL.md",
      "describes how to capture user facts and corrections into .reflexio/.",
      "",
      "Load the skill when: user states a preference/fact/config, user corrects",
      "you and later confirms the fix, or you need to retrieve past context.",
    ].join("\n");
    event.context.bootstrapFiles.push({
      path: "REFLEXIO_EMBEDDED_REMINDER.md",
      content: reminder,
    });
  }
}

/**
 * Main handler — Openclaw invokes this for each subscribed event.
 */
export const handler = async (event: HookEvent, api: HookApi): Promise<void> => {
  const workspace = resolveWorkspace();
  try {
    if (event.type === "agent" && event.action === "bootstrap") {
      await handleBootstrap(event, api, workspace);
      return;
    }
    if (event.type === "session" && event.action === "compact:before") {
      await handleBatchExtraction(event, api, workspace);
      return;
    }
    if (event.type === "command" && (event.action === "stop" || event.action === "reset")) {
      await handleBatchExtraction(event, api, workspace);
      return;
    }
  } catch (err) {
    console.error(`[reflexio-embedded] hook error on ${event.type}:${event.action}: ${err}`);
  }
};

/**
 * Decide whether the current transcript is worth extracting from.
 * Skip if there are no user messages or fewer than 2 turns total.
 */
function transcriptWorthExtracting(event: HookEvent): boolean {
  const messages = event.context?.messages;
  if (!Array.isArray(messages) || messages.length < 2) return false;
  const hasUser = messages.some((m) => (m as any).role === "user");
  return hasUser;
}

/**
 * Serialize transcript into a plain-text form suitable for the sub-agent's task prompt.
 */
function serializeTranscript(event: HookEvent): string {
  const messages = event.context?.messages || [];
  return messages
    .map((m: any) => {
      const role = m.role || "unknown";
      const content = typeof m.content === "string" ? m.content : JSON.stringify(m.content);
      const ts = m.timestamp ? ` [${m.timestamp}]` : "";
      return `### ${role}${ts}\n${content}`;
    })
    .join("\n\n");
}

/**
 * Build the task prompt handed to the reflexio-extractor sub-agent.
 * The sub-agent's system prompt already contains its workflow (from agents/reflexio-extractor.md).
 * This prompt just provides the transcript and reminds it of its job.
 */
function buildExtractionTaskPrompt(event: HookEvent): string {
  const transcript = serializeTranscript(event);
  return [
    "Run your extraction workflow on the following transcript.",
    "",
    "Follow your system prompt: extract profiles and playbooks, then run shallow pairwise dedup against existing .reflexio/ entries.",
    "",
    "## Transcript",
    "",
    transcript,
  ].join("\n");
}

async function handleBatchExtraction(event: HookEvent, api: HookApi, workspace: string): Promise<void> {
  // Always run TTL sweep (cheap, sync)
  await ttlSweepProfiles(workspace);

  if (!transcriptWorthExtracting(event)) {
    return;
  }

  if (!api.runtime?.subagent?.run) {
    console.error("[reflexio-embedded] subagent.run not available; skipping extraction");
    return;
  }

  // Fire-and-forget: Openclaw manages lifecycle via its Background Tasks ledger
  void api.runtime.subagent.run({
    task: buildExtractionTaskPrompt(event),
    agentId: "reflexio-extractor",
    runTimeoutSeconds: 120,
    mode: "run",
  }).catch((err) => {
    console.error(`[reflexio-embedded] failed to spawn extractor: ${err}`);
  });

  // Return immediately — do not await the subagent run
}

export default handler;
