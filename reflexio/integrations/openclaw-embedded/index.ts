// Reflexio Embedded — Openclaw plugin entry.
//
// Registers lifecycle hooks against the modern Openclaw Plugin API:
//   - before_agent_start: TTL sweep of .reflexio/profiles, inject SKILL.md reminder
//   - before_compaction:  run extractor subagent over the session transcript
//   - before_reset:       run extractor subagent before the transcript is wiped
//   - session_end:        run extractor subagent on session termination (covers /stop)
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
} from "./hook/handler.js";
import { writeProfile } from "./scripts/lib/write-profile.js";
import { writePlaybook } from "./scripts/lib/write-playbook.js";
import { search } from "./scripts/lib/search.js";

export default definePluginEntry({
  id: "reflexio-embedded",
  name: "Reflexio Embedded",
  description:
    "Reflexio-style user profile and playbook extraction using Openclaw's native memory engine, hooks, and sub-agents.",
  register(api) {
    const log = api.logger;

    // before_agent_start: cheap per-run entry point. Run TTL sweep and inject a
    // short system-prompt reminder so the LLM knows the SKILL.md is available.
    api.on("before_agent_start", async (_event, ctx) => {
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
      } catch (err) {
        log.error?.(`[reflexio-embedded] ttl sweep failed: ${err}`);
      }
      return {
        prependSystemContext: injectBootstrapReminder(),
      };
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
          log,
          reason: `before_reset:${event.reason ?? "unknown"}`,
        });
      } catch (err) {
        log.error?.(`[reflexio-embedded] before_reset failed: ${err}`);
      }
    });

    // session_end: fires when a session terminates for any reason (stop, idle,
    // daily rollover, etc.). Covers the legacy `command:stop` case.
    api.on("session_end", async (event, ctx) => {
      try {
        await ttlSweepProfiles(ctx.workspaceDir);
        await spawnExtractor({
          runtime: api.runtime,
          workspaceDir: ctx.workspaceDir,
          sessionKey: ctx.sessionKey ?? event.sessionKey,
          messages: undefined, // transcript lives on disk at this point
          sessionFile: event.sessionFile,
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

    function loadPluginConfig() {
      try {
        const cfgPath = path.resolve(import.meta.dirname || __dirname, "config.json");
        return JSON.parse(fs.readFileSync(cfgPath, "utf8"));
      } catch {
        return { dedup: { shallow_threshold: 0.4, top_k: 5 } };
      }
    }

    /**
     * Resolve the agent's workspace directory.
     * Mirrors Openclaw's resolveDefaultAgentWorkspaceDir logic:
     *   ~/.openclaw/workspace (default)
     *   ~/.openclaw/workspace-{profile} (if OPENCLAW_PROFILE is set)
     *
     * We can't use api.runtime.agent.resolveAgentWorkspaceDir(cfg, agentId)
     * because tool execute handlers don't receive agent context — we don't
     * know which agentId invoked the tool. This matches the default agent's
     * workspace which is correct for the common single-agent setup.
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
      async execute(_id: string, params: { slug: string; ttl: string; body: string }) {
        const workspaceDir = resolveWorkspaceDir();
        const config = loadPluginConfig();
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
      async execute(_id: string, params: { slug: string; body: string }) {
        const workspaceDir = resolveWorkspaceDir();
        const config = loadPluginConfig();
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
  },
});
