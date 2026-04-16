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

// Implemented in next task
async function handleBatchExtraction(event: HookEvent, api: HookApi, workspace: string): Promise<void> {
  // Stub — implemented in Task 19
  console.log("[reflexio-embedded] batch extraction not yet implemented");
}

export default handler;
