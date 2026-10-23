// Resolve installation policy once per register(). Connection configuration
// wins over environment fallbacks; search and publication remain separate
// policies. Hook-owned deadlines are supplied to the client by the caller.

export const DEFAULT_URL = "http://127.0.0.1:8061/";
export const DEFAULT_MAX_CONTENT_LENGTH = 10_000;

export interface Settings {
  url: string;
  apiKey: string;
  topK: number;
  minPromptLength: number;
  searchTimeoutMs: number;
  maxContentLength: number;
}

function str(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function section(config: Record<string, unknown>, name: string): Record<string, unknown> {
  const value = config[name];
  if (value === undefined) return {};
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`Invalid Reflexio ${name} policy`);
  }
  return value as Record<string, unknown>;
}

function integer(
  config: Record<string, unknown>,
  key: string,
  fallback: number,
  minimum: number,
  path: string,
  maximum = Number.MAX_SAFE_INTEGER,
): number {
  const value = config[key] === undefined ? fallback : config[key];
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < minimum ||
    value > maximum
  ) {
    throw new Error(`Invalid Reflexio ${path}.${key} policy`);
  }
  return value;
}

export function resolveSettings(
  pluginConfig: Record<string, unknown> | undefined,
  env: NodeJS.ProcessEnv = process.env,
): Settings {
  const cfg = pluginConfig ?? {};
  const search = section(cfg, "search");
  const publish = section(cfg, "publish");
  const url = str(cfg.url) || str(env.REFLEXIO_URL) || DEFAULT_URL;
  return {
    url: url.endsWith("/") ? url : `${url}/`,
    apiKey: str(cfg.apiKey) || str(env.REFLEXIO_API_KEY),
    topK: integer(search, "top_k", 3, 1, "search"),
    minPromptLength: integer(search, "min_prompt_length", 5, 0, "search"),
    // Registration adds 1s of host overhead; keep its timer in the signed range.
    searchTimeoutMs: integer(search, "timeout_ms", 3000, 1, "search", 2 ** 31 - 1001),
    maxContentLength: integer(
      publish,
      "max_content_length",
      DEFAULT_MAX_CONTENT_LENGTH,
      1,
      "publish",
    ),
  };
}
