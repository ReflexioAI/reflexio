// Narrow ambient shim for the Openclaw Plugin SDK surface that this plugin
// actually touches. The real types ship with the `openclaw` npm package, but
// we avoid depending on the full host at build time — plugins load at runtime
// via the host's own jiti instance.
//
// If you need richer type coverage, install `openclaw` as a devDependency and
// delete this file; the real .d.ts files will take over.
declare module "openclaw/plugin-sdk/plugin-entry" {
  export type PluginHookName =
    | "before_model_resolve"
    | "before_prompt_build"
    | "before_agent_start"
    | "before_agent_reply"
    | "llm_input"
    | "llm_output"
    | "agent_end"
    | "before_compaction"
    | "after_compaction"
    | "before_reset"
    | "inbound_claim"
    | "message_received"
    | "message_sending"
    | "message_sent"
    | "before_tool_call"
    | "after_tool_call"
    | "tool_result_persist"
    | "before_message_write"
    | "session_start"
    | "session_end"
    | "subagent_spawning"
    | "subagent_delivery_target"
    | "subagent_spawned"
    | "subagent_ended"
    | "gateway_start"
    | "gateway_stop"
    | "before_dispatch"
    | "reply_dispatch"
    | "before_install";

  export type PluginLogger = {
    info?: (msg: string) => void;
    warn?: (msg: string) => void;
    error?: (msg: string) => void;
    debug?: (msg: string) => void;
  };

  export type PluginRuntime = {
    subagent?: {
      run: (params: {
        sessionKey: string;
        message: string;
        provider?: string;
        model?: string;
        extraSystemPrompt?: string;
        lane?: string;
        deliver?: boolean;
        idempotencyKey?: string;
      }) => Promise<{ runId: string }>;
    };
  };

  export type PluginHookAgentContext = {
    runId?: string;
    agentId?: string;
    sessionKey?: string;
    sessionId?: string;
    workspaceDir?: string;
  };

  export type PluginHookSessionContext = {
    agentId?: string;
    sessionId: string;
    sessionKey?: string;
    workspaceDir?: string;
  };

  export type PluginHookBeforeAgentStartEvent = {
    prompt: string;
    messages?: unknown[];
  };

  export type PluginHookBeforeAgentStartResult = {
    prependSystemContext?: string;
    systemPrompt?: string;
    prependContext?: string;
    appendSystemContext?: string;
    modelOverride?: string;
    providerOverride?: string;
  };

  export type PluginHookBeforeCompactionEvent = {
    messageCount: number;
    compactingCount?: number;
    tokenCount?: number;
    messages?: unknown[];
    sessionFile?: string;
  };

  export type PluginHookBeforeResetEvent = {
    sessionFile?: string;
    messages?: unknown[];
    reason?: string;
  };

  export type PluginHookSessionEndEvent = {
    sessionId: string;
    sessionKey?: string;
    messageCount: number;
    durationMs?: number;
    reason?: string;
    sessionFile?: string;
  };

  export type OpenClawPluginApi = {
    id: string;
    name: string;
    runtime: PluginRuntime;
    logger: PluginLogger;
    on: {
      (
        hookName: "before_agent_start",
        handler: (
          event: PluginHookBeforeAgentStartEvent,
          ctx: PluginHookAgentContext,
        ) =>
          | Promise<PluginHookBeforeAgentStartResult | void>
          | PluginHookBeforeAgentStartResult
          | void,
        opts?: { priority?: number },
      ): void;
      (
        hookName: "before_compaction",
        handler: (
          event: PluginHookBeforeCompactionEvent,
          ctx: PluginHookAgentContext,
        ) => Promise<void> | void,
        opts?: { priority?: number },
      ): void;
      (
        hookName: "before_reset",
        handler: (
          event: PluginHookBeforeResetEvent,
          ctx: PluginHookAgentContext,
        ) => Promise<void> | void,
        opts?: { priority?: number },
      ): void;
      (
        hookName: "session_end",
        handler: (
          event: PluginHookSessionEndEvent,
          ctx: PluginHookSessionContext,
        ) => Promise<void> | void,
        opts?: { priority?: number },
      ): void;
      (
        hookName: PluginHookName,
        handler: (event: unknown, ctx: unknown) => unknown,
        opts?: { priority?: number },
      ): void;
    };
  };

  export type OpenClawPluginDefinition = {
    id: string;
    name: string;
    description?: string;
    version?: string;
    register: (api: OpenClawPluginApi) => void | Promise<void>;
  };

  export function definePluginEntry(
    def: OpenClawPluginDefinition,
  ): OpenClawPluginDefinition;
}
