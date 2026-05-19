// openclaw-smart — TS shim that forwards every openClaw hook to the
// Python openclaw_smart package via bash + uv.
//
// All logic lives in src/openclaw_smart/. This file only does SDK wiring.
//
// Spawn-runner choice: openClaw's runtime currently injects
// `runtime.system.runCommandWithTimeout` only for *trusted* in-process
// plugins. User-installed plugins (us) get the runtime without `system`,
// so capturing it at register-time threw
// `TypeError: Cannot read properties of undefined (reading 'runCommandWithTimeout')`
// and prevented our plugin from loading. We spawn via Node's built-in
// `child_process.spawn` instead, which is not gated by the trust boundary.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { spawn } from "node:child_process";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

type AnyHandler = (event: unknown, ctx: unknown) => unknown;

interface SpawnResult {
  stdout: string;
  stderr: string;
  code: number | null;
}

// Compiled output lives at <plugin>/dist/index.js; scripts/ + skills/ stay
// at the plugin root, so we resolve one level up from this module's dir.
const _MODULE_DIR = path.dirname(fileURLToPath(import.meta.url));
const PLUGIN_ROOT = path.resolve(_MODULE_DIR, "..");
const HOOK_ENTRY = path.join(PLUGIN_ROOT, "scripts", "hook_entry.sh");

// Map openClaw hook name → bash event token + per-event timeout (ms).
const HOOKS: { name: string; token: string; timeoutMs: number }[] = [
  { name: "session_start", token: "session-start", timeoutMs: 30000 },
  { name: "before_prompt_build", token: "before-prompt-build", timeoutMs: 15000 },
  { name: "before_tool_call", token: "before-tool-call", timeoutMs: 10000 },
  { name: "after_tool_call", token: "after-tool-call", timeoutMs: 15000 },
  { name: "agent_end", token: "agent-end", timeoutMs: 30000 },
  { name: "session_end", token: "session-end", timeoutMs: 60000 },
];

function runScript(
  argv: string[],
  opts: { timeoutMs: number; input?: string },
): Promise<SpawnResult> {
  return new Promise((resolve) => {
    const [cmd, ...rest] = argv;
    const child = spawn(cmd, rest, { stdio: ["pipe", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    let settled = false;

    const finish = (code: number | null) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      resolve({ stdout, stderr, code });
    };

    const timer = setTimeout(() => {
      try {
        child.kill("SIGKILL");
      } catch {
        // ignore — the close listener still fires.
      }
      finish(null);
    }, opts.timeoutMs);

    child.stdout.on("data", (d) => {
      stdout += d.toString();
    });
    child.stderr.on("data", (d) => {
      stderr += d.toString();
    });
    child.on("error", () => finish(null));
    child.on("close", (code) => finish(code));

    if (opts.input !== undefined) {
      child.stdin.write(opts.input);
    }
    child.stdin.end();
  });
}

export default definePluginEntry({
  id: "reflexio-openclaw-smart",
  name: "Reflexio openClaw Smart",
  description:
    "Cross-session memory via reflexio. Publishes conversations for extraction, " +
    "injects relevant profiles and playbooks before each response.",
  register(api) {
    const log = api.logger;
    const pluginConfig = api.pluginConfig ?? {};

    // Hook-only plugin — `openclaw plugins doctor` reports we are on the
    // supported compatibility path. Force-publish from inside a session is
    // exposed via the `learn` skill (plugin/skills/learn/SKILL.md), which
    // shells out to the same scripts/cli.sh entry point as a tool would.
    // Registering an agent tool here requires a manifest contracts.tools
    // declaration; we don't need one — keeping the surface skill-only.
    for (const { name, token, timeoutMs } of HOOKS) {
      const handler: AnyHandler = async (event, ctx) => {
        const ctxObj = (ctx ?? {}) as { sessionKey?: string };
        // Python handlers read a flat dict (e.g. payload.get("sessionKey"),
        // payload.get("prompt")). Merge ctx first, event on top so any
        // event-specific overrides win on key clash.
        const payload = {
          ...(ctxObj as Record<string, unknown>),
          ...((event ?? {}) as Record<string, unknown>),
          plugin_config: pluginConfig,
        };
        try {
          const r = await runScript(
            ["bash", HOOK_ENTRY, "openclaw", token],
            { timeoutMs, input: JSON.stringify(payload) },
          );
          if (r.code !== 0) {
            log.debug?.(`hook ${name} exited ${r.code}: ${r.stderr.slice(0, 200)}`);
            return undefined;
          }
          const out = r.stdout.trim();
          if (!out) return undefined;
          return JSON.parse(out);
        } catch (e) {
          log.debug?.(`hook ${name} failed: ${(e as Error).message}`);
          return undefined;
        }
      };
      // Cast: AnyHandler is the most general shape but the SDK's `on` overloads
      // narrow specific events. We delegate all to the same shell entry point,
      // so `as never` opts out of the specialized signatures.
      api.on(name as never, handler as never);
    }
  },
});
