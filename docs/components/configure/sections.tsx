"use client";

import { Fragment } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Section,
  FieldRow,
  TextField,
  NumberField,
  SwitchField,
  TextAreaField,
} from "./primitives";
import {
  APIKeyConfig,
  AgentSuccessConfig,
  EXTRACTION_PRESET_LABELS,
  ExtractionPreset,
  LLMConfig,
  PRESET_VALUES,
  ProfileExtractorConfig,
  ReflexioConfig,
  StorageConfigSQLite,
  ToolUseConfig,
  UserPlaybookExtractorConfig,
  defaultAgentSuccess,
  defaultPlaybookExtractor,
  defaultProfileExtractor,
  defaultTool,
} from "@/lib/config-schema";

type SetConfig = (updater: (prev: ReflexioConfig) => ReflexioConfig) => void;

export function StorageSection({
  value,
  setConfig,
}: {
  value: StorageConfigSQLite | null;
  setConfig: SetConfig;
}) {
  const current = value ?? { db_path: null };
  return (
    <Section
      title="Storage (SQLite)"
      description="This build only supports SQLite storage. Leave blank to use the default file from SQLITE_FILE_DIRECTORY."
    >
      <FieldRow
        label="Database file path"
        htmlFor="storage-db-path"
        hint="Example: ~/.reflexio/reflexio.db"
      >
        <TextField
          id="storage-db-path"
          value={current.db_path}
          onChange={(v) =>
            setConfig((prev) => ({
              ...prev,
              storage_config: { db_path: v },
            }))
          }
          placeholder="(default)"
        />
      </FieldRow>
    </Section>
  );
}

export function AgentContextSection({
  value,
  setConfig,
}: {
  value: string | null;
  setConfig: SetConfig;
}) {
  return (
    <Section
      title="Agent context"
      description="Describes the agent's working environment. Used as shared context when extracting profiles, playbooks, and success signals."
    >
      <FieldRow label="Agent context prompt" htmlFor="agent-context">
        <TextAreaField
          id="agent-context"
          value={value}
          onChange={(v) =>
            setConfig((prev) => ({ ...prev, agent_context_prompt: v }))
          }
          placeholder="e.g. You are a customer-support assistant for Acme Corp's SaaS product…"
          rows={4}
        />
      </FieldRow>
    </Section>
  );
}

export function BatchingSection({
  config,
  setConfig,
}: {
  config: ReflexioConfig;
  setConfig: SetConfig;
}) {
  const preset = config.extraction_preset;
  const usingPreset = preset !== null;

  return (
    <Section
      title="Batching & presets"
      description="Controls how many interactions are processed per extraction run."
    >
      <FieldRow
        label="Extraction preset"
        htmlFor="extraction-preset"
        hint="Presets bundle batch size and interval. Pick 'Custom' to set values manually."
      >
        <Select
          value={preset ?? "custom"}
          onValueChange={(v) => {
            const next = v === "custom" ? null : (v as ExtractionPreset);
            setConfig((prev) => {
              if (next === null) return { ...prev, extraction_preset: null };
              const [bs, bi] = PRESET_VALUES[next];
              return {
                ...prev,
                extraction_preset: next,
                batch_size: bs,
                batch_interval: bi,
              };
            });
          }}
        >
          <SelectTrigger className="h-8 text-xs w-full">
            <SelectValue placeholder="Select preset…" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="custom" className="text-xs">
              Custom
            </SelectItem>
            {(Object.keys(EXTRACTION_PRESET_LABELS) as ExtractionPreset[]).map(
              (key) => (
                <SelectItem key={key} value={key} className="text-xs">
                  {EXTRACTION_PRESET_LABELS[key]}
                </SelectItem>
              )
            )}
          </SelectContent>
        </Select>
      </FieldRow>

      <div className="grid grid-cols-2 gap-3">
        <FieldRow
          label="Batch size"
          htmlFor="batch-size"
          hint="Messages per extraction run."
        >
          <NumberField
            id="batch-size"
            value={config.batch_size}
            onChange={(v) =>
              setConfig((prev) => ({ ...prev, batch_size: v ?? 1 }))
            }
            min={1}
            disabled={usingPreset}
          />
        </FieldRow>
        <FieldRow
          label="Batch interval"
          htmlFor="batch-interval"
          hint="Must be ≤ batch size."
        >
          <NumberField
            id="batch-interval"
            value={config.batch_interval}
            onChange={(v) =>
              setConfig((prev) => ({ ...prev, batch_interval: v ?? 1 }))
            }
            min={1}
            disabled={usingPreset}
          />
        </FieldRow>
      </div>

      <SwitchField
        id="document-expansion"
        label="Enable document expansion"
        hint="Rewrites stored text during indexing to improve full-text-search recall."
        checked={config.enable_document_expansion}
        onCheckedChange={(checked) =>
          setConfig((prev) => ({ ...prev, enable_document_expansion: checked }))
        }
      />
    </Section>
  );
}

type SimpleProvider =
  | "anthropic"
  | "openrouter"
  | "gemini"
  | "minimax"
  | "deepseek"
  | "zai"
  | "moonshot"
  | "xai";

const SIMPLE_PROVIDERS: { key: SimpleProvider; label: string }[] = [
  { key: "anthropic", label: "Anthropic" },
  { key: "openrouter", label: "OpenRouter" },
  { key: "gemini", label: "Google Gemini" },
  { key: "minimax", label: "MiniMax" },
  { key: "deepseek", label: "DeepSeek" },
  { key: "zai", label: "Zhipu AI (GLM)" },
  { key: "moonshot", label: "Moonshot (Kimi)" },
  { key: "xai", label: "xAI (Grok)" },
];

function emptyApiKeys(): APIKeyConfig {
  return {
    custom_endpoint: null,
    openai: null,
    anthropic: null,
    openrouter: null,
    gemini: null,
    minimax: null,
    deepseek: null,
    dashscope: null,
    zai: null,
    moonshot: null,
    xai: null,
  };
}

export function APIKeysSection({
  value,
  setConfig,
}: {
  value: APIKeyConfig | null;
  setConfig: SetConfig;
}) {
  const keys = value ?? emptyApiKeys();

  const update = (patch: Partial<APIKeyConfig>) => {
    setConfig((prev) => ({
      ...prev,
      api_key_config: { ...(prev.api_key_config ?? emptyApiKeys()), ...patch },
    }));
  };

  return (
    <Section
      title="LLM API keys"
      description="Credentials for LLM providers. Leave a field blank to unset it. A custom OpenAI-compatible endpoint, when fully filled in, takes priority over other providers for completion calls."
      defaultOpen={false}
    >
      <FieldRow label="OpenAI API key" htmlFor="key-openai">
        <TextField
          id="key-openai"
          type="password"
          value={keys.openai?.api_key ?? null}
          onChange={(v) => update({ openai: v ? { api_key: v } : null })}
          placeholder="sk-…"
        />
      </FieldRow>

      {SIMPLE_PROVIDERS.map(({ key, label }) => (
        <FieldRow key={key} label={`${label} API key`} htmlFor={`key-${key}`}>
          <TextField
            id={`key-${key}`}
            type="password"
            value={keys[key]?.api_key ?? null}
            onChange={(v) => update({ [key]: v ? { api_key: v } : null } as Partial<APIKeyConfig>)}
          />
        </FieldRow>
      ))}

      <FieldRow
        label="DashScope (Qwen) API key"
        htmlFor="key-dashscope"
        hint="Optional api_base for intl vs China endpoint."
      >
        <TextField
          id="key-dashscope"
          type="password"
          value={keys.dashscope?.api_key ?? null}
          onChange={(v) =>
            update({
              dashscope: v
                ? { api_key: v, api_base: keys.dashscope?.api_base ?? null }
                : null,
            })
          }
        />
        <TextField
          id="key-dashscope-base"
          value={keys.dashscope?.api_base ?? null}
          onChange={(v) =>
            update({
              dashscope: keys.dashscope?.api_key
                ? { api_key: keys.dashscope.api_key, api_base: v }
                : null,
            })
          }
          placeholder="api_base (optional)"
        />
      </FieldRow>

      <div className="space-y-2 pt-2 border-t border-border">
        <p className="text-xs font-semibold">Custom OpenAI-compatible endpoint</p>
        <FieldRow label="Model" htmlFor="custom-model">
          <TextField
            id="custom-model"
            value={keys.custom_endpoint?.model ?? null}
            onChange={(v) =>
              update({
                custom_endpoint: {
                  model: v ?? "",
                  api_key: keys.custom_endpoint?.api_key ?? "",
                  api_base: keys.custom_endpoint?.api_base ?? "",
                },
              })
            }
            placeholder="e.g. openai/mistral"
          />
        </FieldRow>
        <FieldRow label="API key" htmlFor="custom-key">
          <TextField
            id="custom-key"
            type="password"
            value={keys.custom_endpoint?.api_key ?? null}
            onChange={(v) =>
              update({
                custom_endpoint: {
                  model: keys.custom_endpoint?.model ?? "",
                  api_key: v ?? "",
                  api_base: keys.custom_endpoint?.api_base ?? "",
                },
              })
            }
          />
        </FieldRow>
        <FieldRow label="API base URL" htmlFor="custom-base">
          <TextField
            id="custom-base"
            value={keys.custom_endpoint?.api_base ?? null}
            onChange={(v) =>
              update({
                custom_endpoint: {
                  model: keys.custom_endpoint?.model ?? "",
                  api_key: keys.custom_endpoint?.api_key ?? "",
                  api_base: v ?? "",
                },
              })
            }
            placeholder="http://localhost:8000/v1"
          />
        </FieldRow>
      </div>
    </Section>
  );
}

const LLM_FIELDS: {
  key: keyof LLMConfig;
  label: string;
  hint: string;
}[] = [
  {
    key: "should_run_model_name",
    label: "Should-run model",
    hint: "Model that decides whether to run extraction on a batch.",
  },
  {
    key: "generation_model_name",
    label: "Generation model",
    hint: "Used for extraction, evaluation, and generation tasks.",
  },
  {
    key: "embedding_model_name",
    label: "Embedding model",
    hint: "Used for vector embeddings. Must match the configured dimension (512).",
  },
  {
    key: "pre_retrieval_model_name",
    label: "Pre-retrieval model",
    hint: "Used for query reformulation before retrieval.",
  },
];

function emptyLlm(): LLMConfig {
  return {
    should_run_model_name: null,
    generation_model_name: null,
    embedding_model_name: null,
    pre_retrieval_model_name: null,
  };
}

export function LLMModelsSection({
  value,
  setConfig,
}: {
  value: LLMConfig | null;
  setConfig: SetConfig;
}) {
  const llm = value ?? emptyLlm();
  return (
    <Section
      title="LLM model overrides"
      description="Override the default model names. Leave blank to fall back to the site-variable defaults."
      defaultOpen={false}
    >
      {LLM_FIELDS.map((f) => (
        <FieldRow key={f.key} label={f.label} htmlFor={`llm-${f.key}`} hint={f.hint}>
          <TextField
            id={`llm-${f.key}`}
            value={llm[f.key]}
            onChange={(v) =>
              setConfig((prev) => ({
                ...prev,
                llm_config: { ...(prev.llm_config ?? emptyLlm()), [f.key]: v },
              }))
            }
            placeholder="(default)"
          />
        </FieldRow>
      ))}
    </Section>
  );
}

function ListItemCard({
  title,
  onRemove,
  children,
}: {
  title: string;
  onRemove: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border p-3 space-y-3 bg-background">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold truncate">{title}</p>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onRemove}
          className="h-7 px-2 text-muted-foreground hover:text-destructive"
        >
          <Trash2 className="h-3.5 w-3.5" />
        </Button>
      </div>
      {children}
    </div>
  );
}

function AddButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      onClick={onClick}
      className="h-8 gap-1.5"
    >
      <Plus className="h-3.5 w-3.5" />
      {label}
    </Button>
  );
}

export function ProfileExtractorsSection({
  value,
  setConfig,
}: {
  value: ProfileExtractorConfig[] | null;
  setConfig: SetConfig;
}) {
  const items = value ?? [];
  const updateAt = (idx: number, patch: Partial<ProfileExtractorConfig>) => {
    setConfig((prev) => {
      const next = [...(prev.profile_extractor_configs ?? [])];
      next[idx] = { ...next[idx], ...patch };
      return { ...prev, profile_extractor_configs: next };
    });
  };
  const removeAt = (idx: number) => {
    setConfig((prev) => {
      const next = [...(prev.profile_extractor_configs ?? [])];
      next.splice(idx, 1);
      return { ...prev, profile_extractor_configs: next };
    });
  };
  const add = () => {
    setConfig((prev) => ({
      ...prev,
      profile_extractor_configs: [
        ...(prev.profile_extractor_configs ?? []),
        defaultProfileExtractor(),
      ],
    }));
  };

  return (
    <Section
      title="Profile extractors"
      description="Extract user-level memory (role, preferences, stable facts). At least one is required; a default is seeded if you remove them all."
    >
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground italic">
          No profile extractors configured — a default will be seeded on save.
        </p>
      )}
      {items.map((item, idx) => (
        <ListItemCard
          key={idx}
          title={item.extractor_name || "(unnamed)"}
          onRemove={() => removeAt(idx)}
        >
          <FieldRow label="Name">
            <TextField
              value={item.extractor_name}
              onChange={(v) => updateAt(idx, { extractor_name: v ?? "" })}
            />
          </FieldRow>
          <FieldRow label="Extraction definition prompt">
            <TextAreaField
              value={item.extraction_definition_prompt}
              onChange={(v) =>
                updateAt(idx, { extraction_definition_prompt: v ?? "" })
              }
              rows={4}
            />
          </FieldRow>
          <FieldRow label="Context prompt (optional)">
            <TextAreaField
              value={item.context_prompt}
              onChange={(v) => updateAt(idx, { context_prompt: v })}
              rows={2}
            />
          </FieldRow>
          <SwitchField
            label="Manual trigger only"
            hint="Skip automatic extraction — only run when triggered manually."
            checked={item.manual_trigger}
            onCheckedChange={(c) => updateAt(idx, { manual_trigger: c })}
          />
          <div className="grid grid-cols-2 gap-3">
            <FieldRow label="Batch size override">
              <NumberField
                value={item.batch_size_override}
                onChange={(v) => updateAt(idx, { batch_size_override: v })}
                min={1}
                allowNull
                placeholder="(inherit)"
              />
            </FieldRow>
            <FieldRow label="Batch interval override">
              <NumberField
                value={item.batch_interval_override}
                onChange={(v) => updateAt(idx, { batch_interval_override: v })}
                min={1}
                allowNull
                placeholder="(inherit)"
              />
            </FieldRow>
          </div>
        </ListItemCard>
      ))}
      <AddButton label="Add profile extractor" onClick={add} />
    </Section>
  );
}

export function PlaybookExtractorsSection({
  value,
  setConfig,
}: {
  value: UserPlaybookExtractorConfig[] | null;
  setConfig: SetConfig;
}) {
  const items = value ?? [];
  const updateAt = (
    idx: number,
    patch: Partial<UserPlaybookExtractorConfig>
  ) => {
    setConfig((prev) => {
      const next = [...(prev.user_playbook_extractor_configs ?? [])];
      next[idx] = { ...next[idx], ...patch };
      return { ...prev, user_playbook_extractor_configs: next };
    });
  };
  const removeAt = (idx: number) => {
    setConfig((prev) => {
      const next = [...(prev.user_playbook_extractor_configs ?? [])];
      next.splice(idx, 1);
      return { ...prev, user_playbook_extractor_configs: next };
    });
  };
  const add = () => {
    setConfig((prev) => ({
      ...prev,
      user_playbook_extractor_configs: [
        ...(prev.user_playbook_extractor_configs ?? []),
        defaultPlaybookExtractor(),
      ],
    }));
  };

  return (
    <Section
      title="Playbook extractors"
      description="Extract behavioral rules for the agent (what works, what doesn't). Defaults are seeded if empty."
      defaultOpen={false}
    >
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground italic">
          No playbook extractors configured — a default will be seeded on save.
        </p>
      )}
      {items.map((item, idx) => (
        <ListItemCard
          key={idx}
          title={item.extractor_name || "(unnamed)"}
          onRemove={() => removeAt(idx)}
        >
          <FieldRow label="Name">
            <TextField
              value={item.extractor_name}
              onChange={(v) => updateAt(idx, { extractor_name: v ?? "" })}
            />
          </FieldRow>
          <FieldRow label="Extraction definition prompt">
            <TextAreaField
              value={item.extraction_definition_prompt}
              onChange={(v) =>
                updateAt(idx, { extraction_definition_prompt: v ?? "" })
              }
              rows={4}
            />
          </FieldRow>
          <FieldRow label="Context prompt (optional)">
            <TextAreaField
              value={item.context_prompt}
              onChange={(v) => updateAt(idx, { context_prompt: v })}
              rows={2}
            />
          </FieldRow>
        </ListItemCard>
      ))}
      <AddButton label="Add playbook extractor" onClick={add} />
    </Section>
  );
}

export function AgentSuccessSection({
  value,
  setConfig,
}: {
  value: AgentSuccessConfig[] | null;
  setConfig: SetConfig;
}) {
  const items = value ?? [];
  const updateAt = (idx: number, patch: Partial<AgentSuccessConfig>) => {
    setConfig((prev) => {
      const next = [...(prev.agent_success_configs ?? [])];
      next[idx] = { ...next[idx], ...patch };
      return { ...prev, agent_success_configs: next };
    });
  };
  const removeAt = (idx: number) => {
    setConfig((prev) => {
      const next = [...(prev.agent_success_configs ?? [])];
      next.splice(idx, 1);
      return { ...prev, agent_success_configs: next.length ? next : null };
    });
  };
  const add = () => {
    setConfig((prev) => ({
      ...prev,
      agent_success_configs: [
        ...(prev.agent_success_configs ?? []),
        defaultAgentSuccess(),
      ],
    }));
  };

  return (
    <Section
      title="Agent success evaluations"
      description="Define what 'success' means for the agent so Reflexio can score interactions."
      defaultOpen={false}
    >
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground italic">
          No evaluations configured.
        </p>
      )}
      {items.map((item, idx) => (
        <ListItemCard
          key={idx}
          title={item.evaluation_name || "(unnamed)"}
          onRemove={() => removeAt(idx)}
        >
          <FieldRow label="Evaluation name">
            <TextField
              value={item.evaluation_name}
              onChange={(v) => updateAt(idx, { evaluation_name: v ?? "" })}
            />
          </FieldRow>
          <FieldRow label="Success definition prompt">
            <TextAreaField
              value={item.success_definition_prompt}
              onChange={(v) =>
                updateAt(idx, { success_definition_prompt: v ?? "" })
              }
              rows={4}
            />
          </FieldRow>
          <FieldRow
            label={`Sampling rate (${(item.sampling_rate * 100).toFixed(0)}%)`}
            hint="Fraction of interactions sampled for evaluation."
          >
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={item.sampling_rate}
              onChange={(e) =>
                updateAt(idx, { sampling_rate: Number(e.target.value) })
              }
              className="w-full"
            />
          </FieldRow>
        </ListItemCard>
      ))}
      <AddButton label="Add evaluation" onClick={add} />
    </Section>
  );
}

export function ToolsSection({
  value,
  setConfig,
}: {
  value: ToolUseConfig[] | null;
  setConfig: SetConfig;
}) {
  const items = value ?? [];
  const updateAt = (idx: number, patch: Partial<ToolUseConfig>) => {
    setConfig((prev) => {
      const next = [...(prev.tool_can_use ?? [])];
      next[idx] = { ...next[idx], ...patch };
      return { ...prev, tool_can_use: next };
    });
  };
  const removeAt = (idx: number) => {
    setConfig((prev) => {
      const next = [...(prev.tool_can_use ?? [])];
      next.splice(idx, 1);
      return { ...prev, tool_can_use: next.length ? next : null };
    });
  };
  const add = () => {
    setConfig((prev) => ({
      ...prev,
      tool_can_use: [...(prev.tool_can_use ?? []), defaultTool()],
    }));
  };

  return (
    <Section
      title="Agent tools"
      description="Tools the agent can call. Used by success evaluation and playbook extraction."
      defaultOpen={false}
    >
      {items.length === 0 && (
        <p className="text-xs text-muted-foreground italic">
          No tools configured.
        </p>
      )}
      {items.map((item, idx) => (
        <ListItemCard
          key={idx}
          title={item.tool_name || "(unnamed)"}
          onRemove={() => removeAt(idx)}
        >
          <FieldRow label="Tool name">
            <TextField
              value={item.tool_name}
              onChange={(v) => updateAt(idx, { tool_name: v ?? "" })}
            />
          </FieldRow>
          <FieldRow label="Tool description">
            <TextAreaField
              value={item.tool_description}
              onChange={(v) => updateAt(idx, { tool_description: v ?? "" })}
              rows={2}
            />
          </FieldRow>
        </ListItemCard>
      ))}
      <AddButton label="Add tool" onClick={add} />
      <Fragment />
    </Section>
  );
}

export function RawJsonSection({ payload }: { payload: unknown }) {
  return (
    <Section title="Raw JSON (read-only)" defaultOpen={false}>
      <pre className="text-[11px] font-mono bg-muted rounded-md p-3 overflow-auto max-h-96">
        {JSON.stringify(payload, null, 2)}
      </pre>
    </Section>
  );
}
