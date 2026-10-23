// OpenClaw plugin entry: wires Reflexio into the agent turn in-process.
//
//   before_prompt_build -> search Reflexio, inject matches as prependContext
//   agent_end           -> publish the finished run (user + assistant + tools)
//   session_end         -> flush queued publishes, then finalize the session
//   /reflexio           -> status | show | learn
//
// before_prompt_build and agent_end are conversation hooks: the operator must
// set plugins.entries.reflexio-openclaw-smart.hooks.allowConversationAccess.

import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";
import { isIncognitoSessionKey } from "openclaw/plugin-sdk/session-key-runtime";
import { ReflexioClient } from "./client.js";
import { resolveSettings } from "./config.js";
import { ReflexioHooks } from "./handlers.js";

// Set by Reflexio's server-side `openclaw` LLM provider on the CLI it spawns
// for extraction, so that subprocess never feeds its own prompts back in.
const INTERNAL_ENV = "OPENCLAW_SMART_INTERNAL";

export default definePluginEntry({
  id: "reflexio-openclaw-smart",
  name: "Reflexio",
  description:
    "Cross-session memory via Reflexio: injects relevant preferences and skills " +
    "before each turn and publishes completed turns for extraction.",
  register(api) {
    if (process.env[INTERNAL_ENV] === "1") {
      api.logger.debug?.("[reflexio] internal extraction run; hooks disabled");
      return;
    }
    const settings = resolveSettings(api.pluginConfig as Record<string, unknown> | undefined);
    const hooks = new ReflexioHooks(
      settings,
      new ReflexioClient(settings),
      api.logger,
      isIncognitoSessionKey,
    );

    api.on("before_prompt_build", (event, ctx) => hooks.beforePromptBuild(event, ctx), {
      timeoutMs: settings.searchTimeoutMs + 1000,
    });
    api.on("agent_end", (event, ctx) => hooks.agentEnd(event, ctx));
    api.on("session_end", (event, ctx) => hooks.sessionEnd(event, ctx));
    api.registerCommand({
      name: "reflexio",
      description: "Reflexio memory: status | show | learn",
      acceptsArgs: true,
      handler: async (ctx) => ({ text: await hooks.command(ctx.args ?? "", ctx) }),
    });
    api.logger.info?.(`[reflexio] ready; backend ${settings.url}`);
  },
});
