"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Settings, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSettings } from "@/hooks/use-settings";
import {
  ReflexioConfig,
  defaultConfig,
  serializeConfig,
} from "@/lib/config-schema";
import {
  StorageSection,
  AgentContextSection,
  BatchingSection,
  APIKeysSection,
  LLMModelsSection,
  ProfileExtractorsSection,
  PlaybookExtractorsSection,
  AgentSuccessSection,
  ToolsSection,
  RawJsonSection,
} from "./sections";

type Status =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "saving" }
  | { kind: "error"; message: string }
  | { kind: "success"; message: string };

// Merge server response into our typed defaults. Server may omit optional
// fields; we rely on defaults so subsequent form renders stay stable.
function hydrate(raw: unknown): ReflexioConfig {
  const base = defaultConfig();
  if (!raw || typeof raw !== "object") return base;
  return { ...base, ...(raw as Partial<ReflexioConfig>) };
}

function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  try {
    return JSON.stringify(err);
  } catch {
    return "Unknown error";
  }
}

export function ConfigEditor() {
  const { apiEndpoint } = useSettings();
  const [config, setConfig] = useState<ReflexioConfig>(defaultConfig);
  const [status, setStatus] = useState<Status>({ kind: "loading" });

  const baseUrl = useMemo(() => apiEndpoint.replace(/\/$/, ""), [apiEndpoint]);

  const fetchConfig = useCallback(async (): Promise<ReflexioConfig> => {
    const res = await fetch(`${baseUrl}/api/get_config`);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`GET /api/get_config failed (${res.status}): ${body}`);
    }
    return hydrate(await res.json());
  }, [baseUrl]);

  const load = useCallback(async () => {
    setStatus({ kind: "loading" });
    try {
      setConfig(await fetchConfig());
      setStatus({ kind: "idle" });
    } catch (err) {
      setStatus({ kind: "error", message: errorMessage(err) });
    }
  }, [fetchConfig]);

  useEffect(() => {
    void load();
  }, [load]);

  const save = useCallback(async () => {
    setStatus({ kind: "saving" });
    try {
      const payload = serializeConfig(config);
      const res = await fetch(`${baseUrl}/api/set_config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body}`);
      }
      setConfig(await fetchConfig());
      setStatus({ kind: "success", message: "Config saved." });
    } catch (err) {
      setStatus({ kind: "error", message: errorMessage(err) });
    }
  }, [baseUrl, config, fetchConfig]);

  const busy = status.kind === "loading" || status.kind === "saving";
  const payload = serializeConfig(config);

  return (
    <div className="flex flex-col h-full">
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center gap-3 mb-1">
          <Settings className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-lg font-semibold">Edit Configuration</h1>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Interactively edit the Reflexio config for this organization. Changes
          are saved via <code className="text-xs">POST /api/set_config</code>.
        </p>
      </div>

      <div className="flex-1 overflow-auto">
        <div className="max-w-3xl mx-auto p-6 space-y-4">
          <StatusBanner status={status} />

          <StorageSection value={config.storage_config} setConfig={setConfig} />
          <AgentContextSection
            value={config.agent_context_prompt}
            setConfig={setConfig}
          />
          <BatchingSection config={config} setConfig={setConfig} />
          <APIKeysSection value={config.api_key_config} setConfig={setConfig} />
          <LLMModelsSection value={config.llm_config} setConfig={setConfig} />
          <ProfileExtractorsSection
            value={config.profile_extractor_configs}
            setConfig={setConfig}
          />
          <PlaybookExtractorsSection
            value={config.user_playbook_extractor_configs}
            setConfig={setConfig}
          />
          <AgentSuccessSection
            value={config.agent_success_configs}
            setConfig={setConfig}
          />
          <ToolsSection value={config.tool_can_use} setConfig={setConfig} />
          <RawJsonSection payload={payload} />

          <div className="flex items-center gap-2 pt-2 sticky bottom-0 bg-background/80 backdrop-blur py-3 -mx-6 px-6 border-t border-border">
            <Button onClick={save} disabled={busy} className="gap-2">
              {status.kind === "saving" && (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              )}
              Save changes
            </Button>
            <Button
              variant="outline"
              onClick={load}
              disabled={busy}
              className="gap-2"
            >
              {status.kind === "loading" && (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              )}
              Reset from server
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusBanner({ status }: { status: Status }) {
  if (status.kind === "idle" || status.kind === "saving") return null;

  if (status.kind === "loading") {
    return (
      <div className="flex items-center gap-2 rounded-md border border-border bg-muted px-3 py-2 text-xs text-muted-foreground">
        <Loader2 className="h-3.5 w-3.5 animate-spin" />
        Loading current config…
      </div>
    );
  }

  if (status.kind === "error") {
    return (
      <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
        <AlertCircle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
        <div className="flex-1 break-words whitespace-pre-wrap">
          {status.message}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 rounded-md border border-green-500/40 bg-green-500/10 px-3 py-2 text-xs text-green-700 dark:text-green-400">
      <CheckCircle2 className="h-3.5 w-3.5" />
      {status.message}
    </div>
  );
}
