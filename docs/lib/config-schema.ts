// TypeScript mirror of the subset of `reflexio/models/config_schema.py`
// that the interactive /configure page edits. Only SQLite storage is
// represented — this repo does not support Supabase or Disk backends.

export type ExtractionPreset =
  | "quick_chat"
  | "standard"
  | "long_form"
  | "high_volume";

export interface StorageConfigSQLite {
  db_path: string | null;
}

export interface OpenAIConfig {
  api_key: string | null;
}

export interface SimpleKeyConfig {
  api_key: string;
}

export interface DashScopeConfig {
  api_key: string;
  api_base: string | null;
}

export interface CustomEndpointConfig {
  model: string;
  api_key: string;
  api_base: string;
}

export interface APIKeyConfig {
  custom_endpoint: CustomEndpointConfig | null;
  openai: OpenAIConfig | null;
  anthropic: SimpleKeyConfig | null;
  openrouter: SimpleKeyConfig | null;
  gemini: SimpleKeyConfig | null;
  minimax: SimpleKeyConfig | null;
  deepseek: SimpleKeyConfig | null;
  dashscope: DashScopeConfig | null;
  zai: SimpleKeyConfig | null;
  moonshot: SimpleKeyConfig | null;
  xai: SimpleKeyConfig | null;
}

export interface LLMConfig {
  should_run_model_name: string | null;
  generation_model_name: string | null;
  embedding_model_name: string | null;
  pre_retrieval_model_name: string | null;
}

export interface ProfileExtractorConfig {
  extractor_name: string;
  extraction_definition_prompt: string;
  context_prompt: string | null;
  metadata_definition_prompt: string | null;
  manual_trigger: boolean;
  batch_size_override: number | null;
  batch_interval_override: number | null;
}

export interface PlaybookAggregatorConfig {
  min_cluster_size: number;
  reaggregation_trigger_count: number;
  clustering_similarity: number;
  direction_overlap_threshold: number;
}

export interface DeduplicationConfig {
  search_threshold: number;
  search_top_k: number;
}

export interface UserPlaybookExtractorConfig {
  extractor_name: string;
  extraction_definition_prompt: string;
  context_prompt: string | null;
  metadata_definition_prompt: string | null;
  aggregation_config: PlaybookAggregatorConfig | null;
  deduplication_config: DeduplicationConfig | null;
  batch_size_override: number | null;
  batch_interval_override: number | null;
}

export interface AgentSuccessConfig {
  evaluation_name: string;
  success_definition_prompt: string;
  metadata_definition_prompt: string | null;
  sampling_rate: number;
  batch_size_override: number | null;
  batch_interval_override: number | null;
}

export interface ToolUseConfig {
  tool_name: string;
  tool_description: string;
}

export interface ReflexioConfig {
  storage_config: StorageConfigSQLite | null;
  agent_context_prompt: string | null;
  tool_can_use: ToolUseConfig[] | null;
  profile_extractor_configs: ProfileExtractorConfig[] | null;
  user_playbook_extractor_configs: UserPlaybookExtractorConfig[] | null;
  agent_success_configs: AgentSuccessConfig[] | null;
  extraction_preset: ExtractionPreset | null;
  batch_size: number;
  batch_interval: number;
  api_key_config: APIKeyConfig | null;
  llm_config: LLMConfig | null;
  enable_document_expansion: boolean;
}

// Preset → (batch_size, batch_interval). Mirrors _PRESET_VALUES in
// config_schema.py so the UI can preview what a preset will set.
export const PRESET_VALUES: Record<ExtractionPreset, [number, number]> = {
  quick_chat: [5, 3],
  standard: [10, 5],
  long_form: [25, 10],
  high_volume: [15, 8],
};

export const EXTRACTION_PRESET_LABELS: Record<ExtractionPreset, string> = {
  quick_chat: "Quick chat (support bots, quick Q&A)",
  standard: "Standard (general conversational agents)",
  long_form: "Long form (coding assistants, research)",
  high_volume: "High volume (1000+ daily interactions)",
};

export function defaultConfig(): ReflexioConfig {
  return {
    storage_config: { db_path: null },
    agent_context_prompt: null,
    tool_can_use: null,
    profile_extractor_configs: [],
    user_playbook_extractor_configs: [],
    agent_success_configs: null,
    extraction_preset: null,
    batch_size: 10,
    batch_interval: 5,
    api_key_config: null,
    llm_config: null,
    enable_document_expansion: false,
  };
}

export function defaultProfileExtractor(): ProfileExtractorConfig {
  return {
    extractor_name: "new_profile_extractor",
    extraction_definition_prompt: "",
    context_prompt: null,
    metadata_definition_prompt: null,
    manual_trigger: false,
    batch_size_override: null,
    batch_interval_override: null,
  };
}

export function defaultPlaybookExtractor(): UserPlaybookExtractorConfig {
  return {
    extractor_name: "new_playbook_extractor",
    extraction_definition_prompt: "",
    context_prompt: null,
    metadata_definition_prompt: null,
    aggregation_config: null,
    deduplication_config: null,
    batch_size_override: null,
    batch_interval_override: null,
  };
}

export function defaultAgentSuccess(): AgentSuccessConfig {
  return {
    evaluation_name: "new_evaluation",
    success_definition_prompt: "",
    metadata_definition_prompt: null,
    sampling_rate: 1.0,
    batch_size_override: null,
    batch_interval_override: null,
  };
}

export function defaultTool(): ToolUseConfig {
  return { tool_name: "", tool_description: "" };
}

// Strip empty strings → null and empty arrays → null so the server receives
// a clean payload. The Pydantic model rejects empty strings for `NonEmptyStr`
// fields, so leaving them in would produce confusing 422s.
export function serializeConfig(config: ReflexioConfig): unknown {
  const clean = <T,>(v: T): T | null => {
    if (v === null || v === undefined) return null;
    if (typeof v === "string" && v.trim() === "") return null;
    return v;
  };

  const cleanApiKeys = (keys: APIKeyConfig | null): APIKeyConfig | null => {
    if (!keys) return null;
    const out: APIKeyConfig = { ...keys };
    for (const provider of [
      "anthropic",
      "openrouter",
      "gemini",
      "minimax",
      "deepseek",
      "zai",
      "moonshot",
      "xai",
    ] as const) {
      const entry = out[provider];
      if (!entry || clean(entry.api_key) === null) out[provider] = null;
    }
    if (out.openai && clean(out.openai.api_key) === null) out.openai = null;
    if (out.dashscope && clean(out.dashscope.api_key) === null) {
      out.dashscope = null;
    } else if (out.dashscope) {
      out.dashscope = {
        api_key: out.dashscope.api_key,
        api_base: clean(out.dashscope.api_base),
      };
    }
    if (out.custom_endpoint) {
      const ce = out.custom_endpoint;
      if (
        clean(ce.model) === null ||
        clean(ce.api_key) === null ||
        clean(ce.api_base) === null
      ) {
        out.custom_endpoint = null;
      }
    }
    const hasAny = Object.values(out).some((v) => v !== null);
    return hasAny ? out : null;
  };

  const cleanLlm = (llm: LLMConfig | null): LLMConfig | null => {
    if (!llm) return null;
    const out: LLMConfig = {
      should_run_model_name: clean(llm.should_run_model_name),
      generation_model_name: clean(llm.generation_model_name),
      embedding_model_name: clean(llm.embedding_model_name),
      pre_retrieval_model_name: clean(llm.pre_retrieval_model_name),
    };
    return Object.values(out).some((v) => v !== null) ? out : null;
  };

  return {
    storage_config: config.storage_config
      ? { db_path: clean(config.storage_config.db_path) }
      : null,
    agent_context_prompt: clean(config.agent_context_prompt),
    tool_can_use: config.tool_can_use?.length ? config.tool_can_use : null,
    profile_extractor_configs: config.profile_extractor_configs?.length
      ? config.profile_extractor_configs
      : null,
    user_playbook_extractor_configs: config.user_playbook_extractor_configs
      ?.length
      ? config.user_playbook_extractor_configs
      : null,
    agent_success_configs: config.agent_success_configs?.length
      ? config.agent_success_configs
      : null,
    extraction_preset: config.extraction_preset,
    batch_size: config.batch_size,
    batch_interval: config.batch_interval,
    api_key_config: cleanApiKeys(config.api_key_config),
    llm_config: cleanLlm(config.llm_config),
    enable_document_expansion: config.enable_document_expansion,
  };
}
