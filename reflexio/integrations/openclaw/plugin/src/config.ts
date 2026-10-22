// Plugin settings, resolved once per register() from plugin config with
// environment fallbacks. Plugin config wins so a gateway can point at
// managed/self-hosted Reflexio without touching the process environment.
//
// Only the backend address and credential are configurable. Budgets below are
// constants: each knob would be another configuration input to document and
// support, and none has a concrete need for per-install tuning yet.

// Reflexio's own default port (`reflexio services start`).
export const DEFAULT_URL = "http://127.0.0.1:8061/";

export interface Settings {
  url: string;
  apiKey: string;
  topK: number;
  minPromptLength: number;
  searchTimeoutMs: number;
  publishTimeoutMs: number;
}

function str(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

export function resolveSettings(
  pluginConfig: Record<string, unknown> | undefined,
  env: NodeJS.ProcessEnv = process.env,
): Settings {
  const cfg = pluginConfig ?? {};
  const url = str(cfg.url) || str(env.REFLEXIO_URL) || DEFAULT_URL;
  return {
    url: url.endsWith("/") ? url : `${url}/`,
    apiKey: str(cfg.apiKey) || str(env.REFLEXIO_API_KEY),
    topK: 3,
    minPromptLength: 5,
    // before_prompt_build delays the user's turn, so search gets a tight budget.
    searchTimeoutMs: 3000,
    // Two attempts (one queued retry, then the current run) must fit inside
    // agent_end's 30 s host budget. A publish that outlives this may still
    // commit server-side; the retry then reports it as already admitted.
    publishTimeoutMs: 12000,
  };
}
