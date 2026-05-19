// openclaw-smart — TS shim that forwards every openClaw hook to the
// Python openclaw_smart package via bash + uv.
//
// All logic lives in src/openclaw_smart/. This file only does SDK wiring.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import * as path from "node:path";

type AnyHandler = (event: unknown, ctx: unknown) => unknown;

const PLUGIN_ROOT = __dirname;
const HOOK_ENTRY = path.join(PLUGIN_ROOT, "scripts", "hook_entry.sh");
const CLI_SCRIPT = path.join(PLUGIN_ROOT, "scripts", "cli.sh");

// Map openClaw hook name → bash event token + per-event timeout (ms).
const HOOKS: { name: string; token: string; timeoutMs: number }[] = [
  { name: "session_start", token: "session-start", timeoutMs: 30000 },
  { name: "before_prompt_build", token: "before-prompt-build", timeoutMs: 15000 },
  { name: "before_tool_call", token: "before-tool-call", timeoutMs: 10000 },
  { name: "after_tool_call", token: "after-tool-call", timeoutMs: 15000 },
  { name: "agent_end", token: "agent-end", timeoutMs: 30000 },
  { name: "session_end", token: "session-end", timeoutMs: 60000 },
];

export default definePluginEntry({
  id: "reflexio-openclaw-smart",
  name: "Reflexio openClaw Smart",
  description:
    "Cross-session memory via reflexio. Publishes conversations for extraction, " +
    "injects relevant profiles and playbooks before each response.",
  register(api) {
    const log = api.logger;
    const runner = api.runtime.system.runCommandWithTimeout;
    const pluginConfig = api.pluginConfig ?? {};

    // Track the most recent session key for the reflexio_publish tool callback.
    let activeSessionKey = "";

    for (const { name, token, timeoutMs } of HOOKS) {
      const handler: AnyHandler = async (event, ctx) => {
        const ctxObj = (ctx ?? {}) as { sessionKey?: string };
        if (ctxObj.sessionKey) activeSessionKey = ctxObj.sessionKey;
        // Python handlers read a flat dict (e.g. payload.get("sessionKey"),
        // payload.get("prompt")). Merge ctx first, event on top so any
        // event-specific overrides win on key clash.
        const payload = {
          ...(ctxObj as Record<string, unknown>),
          ...((event ?? {}) as Record<string, unknown>),
          plugin_config: pluginConfig,
        };
        try {
          const r = await runner(
            ["bash", HOOK_ENTRY, "openclaw", token],
            { timeoutMs, input: JSON.stringify(payload) },
          );
          if (r.code !== 0) {
            log.debug?.(`hook ${name} exited ${r.code}: ${r.stderr?.slice(0, 200)}`);
            return undefined;
          }
          const out = (r.stdout ?? "").trim();
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

    // Agent-invoked immediate publish.
    api.registerTool({
      name: "reflexio_publish",
      description:
        "Immediately publish all buffered conversation turns to the Reflexio server. " +
        "Use after user corrections or high-signal moments when you don't want to wait " +
        "for the automatic session-end publish.",
      parameters: { type: "object", properties: {} },
      optional: true,
      async execute(_id, _params) {
        try {
          const r = await runner(
            ["bash", CLI_SCRIPT, "learn", "--session", activeSessionKey],
            { timeoutMs: 30000 },
          );
          const text =
            r.code === 0
              ? r.stdout ?? "publish queued"
              : `publish failed: ${r.stderr?.slice(0, 200)}`;
          return { content: [{ type: "text" as const, text }] };
        } catch (e) {
          return {
            content: [
              { type: "text" as const, text: `publish error: ${(e as Error).message}` },
            ],
          };
        }
      },
    });
  },
});
