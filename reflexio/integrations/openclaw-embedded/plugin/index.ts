// Reflexio Embedded — Openclaw plugin entry.
//
// Registers lifecycle hooks against the Openclaw Plugin API:
//   - before_prompt_build: inject SKILL.md reminder into system prompt
//   - before_agent_start:  TTL sweep, workspace setup
//   - before_compaction:   run extractor subagent over the session transcript
//   - before_reset:        run extractor subagent before the transcript is wiped
//   - session_end:         run extractor subagent on session termination
//
// The TTL sweep + extractor spawning logic lives in ./hook/handler.ts and is
// re-used verbatim — this file is only the SDK wiring.
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

import {
  injectBootstrapReminder,
  spawnExtractor,
  ttlSweepProfiles,
} from "./hook/handler.ts";
import { setupWorkspaceResources } from "./hook/setup.ts";
import { writeProfile } from "./lib/write-profile.ts";
import { writePlaybook } from "./lib/write-playbook.ts";
import { search } from "./lib/search.ts";

export default definePluginEntry({
  id: "reflexio-embedded",
  name: "Reflexio Embedded",
  description:
    "Reflexio-style user profile and playbook extraction using Openclaw's native memory engine, hooks, and sub-agents.",
  register(api) {
    const log = api.logger;
    const pluginDir = import.meta.dirname || __dirname;

    // Load agent system prompts from the plugin's own directory
    let extractorSystemPrompt: string | undefined;
    let consolidatorSystemPrompt: string | undefined;
    try {
      extractorSystemPrompt = fs.readFileSync(
        path.join(pluginDir, "agents", "reflexio-extractor.md"),
        "utf8",
      );
    } catch {
      log.warn?.("[reflexio-embedded] could not load reflexio-extractor.md agent definition");
    }
    try {
      consolidatorSystemPrompt = fs.readFileSync(
        path.join(pluginDir, "agents", "reflexio-consolidator.md"),
        "utf8",
      );
    } catch {
      log.warn?.("[reflexio-embedded] could not load reflexio-consolidator.md agent definition");
    }

    // before_prompt_build: inject a short system-prompt reminder so the LLM
    // knows the SKILL.md is available (replaces deprecated before_agent_start
    // for prompt mutation).
    api.on("before_prompt_build", async () => {
      return {
        prependSystemContext: injectBootstrapReminder(),
      };
    });

    // before_agent_start: workspace setup + TTL sweep (non-prompt tasks).
    api.on("before_agent_start", async (_event, ctx) => {
      try {
        setupWorkspaceResources(pluginDir);
      } catch (err) {
        log.error?.(`[reflexio-embedded] workspace setup failed: ${err}`);
      }
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
      } catch (err) {
        log.error?.(`[reflexio-embedded] ttl sweep failed: ${err}`);
      }
    });

    // before_compaction: spawn extractor BEFORE the LLM compacts history so we
    // still have the raw transcript to extract from.
    api.on("before_compaction", async (event, ctx) => {
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
        await spawnExtractor({
          runtime: api.runtime,
          workspaceDir: ctx.workspaceDir,
          sessionKey: ctx.sessionKey,
          messages: event.messages,
          sessionFile: event.sessionFile,
          extraSystemPrompt: extractorSystemPrompt,
          log,
          reason: "before_compaction",
        });
      } catch (err) {
        log.error?.(`[reflexio-embedded] before_compaction failed: ${err}`);
      }
    });

    // before_reset: user ran /reset — flush current transcript to the extractor.
    api.on("before_reset", async (event, ctx) => {
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
        await spawnExtractor({
          runtime: api.runtime,
          workspaceDir: ctx.workspaceDir,
          sessionKey: ctx.sessionKey,
          messages: event.messages,
          sessionFile: event.sessionFile,
          extraSystemPrompt: extractorSystemPrompt,
          log,
          reason: `before_reset:${event.reason ?? "unknown"}`,
        });
      } catch (err) {
        log.error?.(`[reflexio-embedded] before_reset failed: ${err}`);
      }
    });

    // session_end: fires when a session terminates for any reason (stop, idle,
    // daily rollover, etc.).
    api.on("session_end", async (event, ctx) => {
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
        await spawnExtractor({
          runtime: api.runtime,
          workspaceDir: ctx.workspaceDir,
          sessionKey: ctx.sessionKey ?? event.sessionKey,
          messages: undefined, // transcript lives on disk at this point
          sessionFile: event.sessionFile,
          extraSystemPrompt: extractorSystemPrompt,
          log,
          reason: `session_end:${event.reason ?? "unknown"}`,
        });
      } catch (err) {
        log.error?.(`[reflexio-embedded] session_end failed: ${err}`);
      }
    });

    // ──────────────────────────────────────────────────────────
    // Agent tools — deterministic control flow for writes + search
    // ──────────────────────────────────────────────────────────
    const runner = api.runtime.system.runCommandWithTimeout;
    const config = api.pluginConfig ?? {
      dedup: { shallow_threshold: 0.7, top_k: 5 },
      consolidation: { threshold_hours: 24 },
    };

    /**
     * Resolve the agent's workspace directory.
     * Mirrors Openclaw's resolveDefaultAgentWorkspaceDir logic:
     *   ~/.openclaw/workspace (default)
     *   ~/.openclaw/workspace-{profile} (if OPENCLAW_PROFILE is set)
     */
    function resolveWorkspaceDir(): string {
      const profile = process.env.OPENCLAW_PROFILE?.trim();
      if (profile && profile.toLowerCase() !== "default") {
        return path.join(os.homedir(), ".openclaw", `workspace-${profile}`);
      }
      return path.join(os.homedir(), ".openclaw", "workspace");
    }

    api.registerTool({
      name: "reflexio_write_profile",
      description:
        "Write a user profile to .reflexio/profiles/ with automatic query preprocessing, memory search, contradiction detection, dedup, and old-file cleanup. Returns the new file path.",
      parameters: {
        type: "object",
        properties: {
          slug: { type: "string", description: "kebab-case topic, e.g. diet-vegan" },
          ttl: {
            type: "string",
            description: "one_day | one_week | one_month | one_quarter | one_year | infinity",
          },
          body: { type: "string", description: "1-3 sentences, one fact per profile" },
        },
        required: ["slug", "ttl", "body"],
      },
      optional: true,
      async execute(_id: string, params: { slug: string; ttl: string; body: string }) {
        const workspaceDir = resolveWorkspaceDir();
        const filePath = await writeProfile({
          slug: params.slug,
          ttl: params.ttl,
          body: params.body,
          workspace: workspaceDir,
          config: config.dedup,
          runner,
        });
        return { content: [{ type: "text" as const, text: filePath }] };
      },
    });

    api.registerTool({
      name: "reflexio_write_playbook",
      description:
        "Write a playbook to .reflexio/playbooks/ with automatic dedup and contradiction detection. Returns the new file path.",
      parameters: {
        type: "object",
        properties: {
          slug: { type: "string", description: "kebab-case trigger summary, e.g. commit-no-trailers" },
          body: {
            type: "string",
            description: "Playbook body with ## When, ## What, ## Why sections",
          },
        },
        required: ["slug", "body"],
      },
      optional: true,
      async execute(_id: string, params: { slug: string; body: string }) {
        const workspaceDir = resolveWorkspaceDir();
        const filePath = await writePlaybook({
          slug: params.slug,
          body: params.body,
          workspace: workspaceDir,
          config: config.dedup,
          runner,
        });
        return { content: [{ type: "text" as const, text: filePath }] };
      },
    });

    api.registerTool({
      name: "reflexio_search",
      description:
        "Search .reflexio/ memory with automatic query preprocessing for better results. Returns JSON with results array.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "raw query — preprocessing is automatic" },
        },
        required: ["query"],
      },
      async execute(_id: string, params: { query: string }) {
        const results = await search(params.query, 5, undefined, runner);
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ results }, null, 2) }],
        };
      },
    });

    // ──────────────────────────────────────────────────────────
    // Consolidation — spawn consolidator sub-agent
    // ──────────────────────────────────────────────────────────
    api.registerTool({
      name: "reflexio_run_consolidation",
      description:
        "Spawn the reflexio-consolidator sub-agent to run a full consolidation sweep. Returns the runId. Call reflexio_consolidation_mark_done after it completes.",
      parameters: { type: "object", properties: {} },
      optional: true,
      async execute() {
        const runFn = api.runtime?.subagent?.run;
        if (!runFn) {
          return { content: [{ type: "text" as const, text: "ERROR: subagent.run unavailable" }] };
        }
        if (!consolidatorSystemPrompt) {
          return { content: [{ type: "text" as const, text: "ERROR: consolidator agent definition not loaded" }] };
        }
        try {
          const result = await runFn({
            sessionKey: `reflexio-consolidator:${Date.now()}`,
            message: "Run your full-sweep consolidation workflow now. Follow your system prompt in full.",
            extraSystemPrompt: consolidatorSystemPrompt,
            lane: "reflexio-consolidator",
          });
          return { content: [{ type: "text" as const, text: `Consolidation started. runId: ${result.runId}` }] };
        } catch (err) {
          return { content: [{ type: "text" as const, text: `ERROR: failed to spawn consolidator: ${err}` }] };
        }
      },
    });

    // ──────────────────────────────────────────────────────────
    // Heartbeat — consolidation check (replaces cron job)
    // ──────────────────────────────────────────────────────────
    const consolidationStateFile = path.join(os.homedir(), ".openclaw", "reflexio-consolidation-state.json");

    api.registerTool({
      name: "reflexio_consolidation_check",
      description:
        "Check if reflexio consolidation is due. Returns OK or ALERT. Called by the agent on heartbeat.",
      parameters: { type: "object", properties: {} },
      async execute() {
        const thresholdHours = config.consolidation?.threshold_hours ?? 24;
        try {
          const state = JSON.parse(fs.readFileSync(consolidationStateFile, "utf8"));
          const elapsedMs = Date.now() - new Date(state.last_consolidation).getTime();
          const elapsedHours = elapsedMs / 3_600_000;
          if (elapsedHours < thresholdHours) {
            const remaining = Math.round(thresholdHours - elapsedHours);
            return { content: [{ type: "text" as const, text: `OK: Last consolidation ${Math.round(elapsedHours)}h ago. Next due in ${remaining}h.` }] };
          }
        } catch {
          // no state file = never consolidated
        }
        return { content: [{ type: "text" as const, text: "ALERT: Consolidation due." }] };
      },
    });

    api.registerTool({
      name: "reflexio_consolidation_mark_done",
      description:
        "Mark consolidation as complete. Call this after a successful consolidation run.",
      parameters: { type: "object", properties: {} },
      async execute() {
        const state = { last_consolidation: new Date().toISOString() };
        fs.mkdirSync(path.dirname(consolidationStateFile), { recursive: true });
        fs.writeFileSync(consolidationStateFile, JSON.stringify(state, null, 2), "utf8");
        return { content: [{ type: "text" as const, text: `Consolidation marked complete at ${state.last_consolidation}.` }] };
      },
    });
  },
});
