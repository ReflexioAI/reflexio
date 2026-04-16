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

import {
  injectBootstrapReminder,
  spawnExtractor,
  ttlSweepProfiles,
} from "./hook/handler.js";

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
  },
});
