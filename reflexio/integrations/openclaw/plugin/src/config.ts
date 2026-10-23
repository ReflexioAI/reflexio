// Resolve installation policy once per register(). Connection configuration
// wins over environment fallbacks; search and publication remain separate
// policies. Hook-owned deadlines are supplied to the client by the caller.

import { createRequire } from "node:module";

type IntegerPolicy = { default: number; minimum: number; maximum: number };
type Manifest = {
  configSchema: {
    properties: {
      search: { properties: Record<"top_k" | "min_prompt_length" | "timeout_ms", IntegerPolicy> };
      publish: { properties: { max_content_length: IntegerPolicy } };
    };
  };
};

// Both src/ and packaged dist/ live beside the authoritative plugin manifest.
const manifest = createRequire(import.meta.url)("../openclaw.plugin.json") as Manifest;
const searchPolicy = manifest.configSchema.properties.search.properties;
const publishPolicy = manifest.configSchema.properties.publish.properties;

export const DEFAULT_URL = "http://127.0.0.1:8061/";
export const DEFAULT_MAX_CONTENT_LENGTH = publishPolicy.max_content_length.default;

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
  policy: IntegerPolicy,
  path: string,
): number {
  const value = config[key] === undefined ? policy.default : config[key];
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < policy.minimum ||
    value > policy.maximum
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
    topK: integer(search, "top_k", searchPolicy.top_k, "search"),
    minPromptLength: integer(search, "min_prompt_length", searchPolicy.min_prompt_length, "search"),
    searchTimeoutMs: integer(search, "timeout_ms", searchPolicy.timeout_ms, "search"),
    maxContentLength: integer(
      publish,
      "max_content_length",
      publishPolicy.max_content_length,
      "publish",
    ),
  };
}
