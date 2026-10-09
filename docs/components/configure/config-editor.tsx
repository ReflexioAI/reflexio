"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Settings, AlertCircle, CheckCircle2, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSettings } from "@/hooks/use-settings";
import {
  type ReflexioConfig,
  defaultConfig,
  serializeConfig,
} from "@/lib/config-schema";
import {
  StorageSection,
  AgentContextSection,
  WindowingSection,
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
  const incoming = raw as Partial<ReflexioConfig>;
  return {
    ...base,
    ...incoming,
    agent_success_config: incoming.agent_success_config ?? null,
  };
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
  const { apiEndpoint, apiKey } = useSettings();
  const [config, setConfig] = useState<ReflexioConfig>(defaultConfig);
  const [status, setStatus] = useState<Status>({ kind: "loading" });
  const operation = useRef(0);

  const baseUrl = useMemo(() => apiEndpoint.replace(/\/$/, ""), [apiEndpoint]);

  const fetchConfig = useCallback(async (signal?: AbortSignal): Promise<ReflexioConfig> => {
    const res = await fetch(`${baseUrl}/api/get_config`, {
      headers: apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined,
      signal,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`GET /api/get_config failed (${res.status}): ${body}`);
    }
    return hydrate(await res.json());
  }, [baseUrl, apiKey]);

  const load = useCallback(async (signal?: AbortSignal) => {
    const current = ++operation.current;
    setStatus({ kind: "loading" });
    try {
      const next = await fetchConfig(signal);
      if (current !== operation.current || signal?.aborted) return;
      setConfig(next);
      setStatus({ kind: "idle" });
    } catch (err) {
      if (current !== operation.current || signal?.aborted) return;
      setStatus({ kind: "error", message: errorMessage(err) });
    }
  }, [fetchConfig]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => {
      controller.abort();
    };
  }, [load]);

  const save = useCallback(async () => {
    const current = ++operation.current;
    setStatus({ kind: "saving" });
    try {
      const payload = serializeConfig(config);
      const res = await fetch(`${baseUrl}/api/set_config`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
        },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status}: ${body}`);
      }
      if (current !== operation.current) return;
      const next = await fetchConfig();
      if (current !== operation.current) return;
      setConfig(next);
      setStatus({ kind: "success", message: "Config saved." });
    } catch (err) {
      if (current !== operation.current) return;
      setStatus({ kind: "error", message: errorMessage(err) });
    }
  }, [baseUrl, apiKey, config, fetchConfig]);

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
          <WindowingSection config={config} setConfig={setConfig} />
          <APIKeysSection value={config.api_key_config} setConfig={setConfig} />
          <LLMModelsSection value={config.llm_config} setConfig={setConfig} />
          <ProfileExtractorsSection
            value={config.profile_extractor_config}
            setConfig={setConfig}
          />
          <PlaybookExtractorsSection
            value={config.user_playbook_extractor_config}
            setConfig={setConfig}
          />
          <AgentSuccessSection
            value={config.agent_success_config}
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
              onClick={() => void load()}
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
