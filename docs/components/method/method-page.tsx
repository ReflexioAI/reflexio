"use client";

import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import type { MethodDef } from "@/lib/types";
import { MethodBadge } from "./method-badge";
import { SplitPanel } from "@/components/layout/split-panel";
import { CodePanel } from "./code-panel";
import { ResultsPanel } from "./results-panel";
import { useSettings } from "@/hooks/use-settings";
import { useExecution } from "@/hooks/use-execution";
import { initialParams } from "@/lib/execution/default-params";

interface MethodPageProps {
  method: MethodDef;
}

export function MethodPage({ method }: MethodPageProps) {
  const { apiEndpoint, apiKey, sessionId, setSessionId } = useSettings();
  const { result, loading, error, execute } = useExecution();

  const [params, setParams] = useState<Record<string, unknown>>(() =>
    initialParams(method, sessionId)
  );
  // The visit identity arrives after hydration; explicit caller edits win.
  const effectiveParams = useMemo(
    () => ({ ...initialParams(method, sessionId), ...params }),
    [method, sessionId, params]
  );
  const handleParamsChange = (next: Record<string, unknown>) => {
    setParams(next);
    if (typeof next.session_id === "string") setSessionId(next.session_id);
  };

  const handleRun = useCallback(
    (runParams: Record<string, unknown>) => {
      execute(method, runParams, apiEndpoint, apiKey);
    },
    [method, apiEndpoint, apiKey, execute]
  );

  // Auto-run on mount if no required parameters
  const hasAutoRun = useRef(false);
  useEffect(() => {
    if (hasAutoRun.current) return;
    const hasRequired = method.params.some((p) => p.required);
    if (!hasRequired && apiEndpoint) {
      hasAutoRun.current = true;
      handleRun(effectiveParams);
    }
  }, [method, apiEndpoint, handleRun, effectiveParams]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-border shrink-0">
        <div className="flex items-center gap-3 mb-1">
          <MethodBadge method={method.httpMethod} />
          <h1 className="text-lg font-semibold">{method.displayName}</h1>
        </div>
        <div className="flex items-center gap-2">
          <code className="text-xs font-mono text-muted-foreground bg-muted px-2 py-0.5 rounded">
            {method.endpoint}
          </code>
        </div>
        <p className="text-sm text-muted-foreground mt-2">
          {method.description}
        </p>
      </div>

      {/* Split Panel */}
      <div className="flex-1 p-4 min-h-0">
        <SplitPanel
          left={<ResultsPanel result={result} loading={loading} error={error} />}
          right={
            <CodePanel
              method={method}
              params={effectiveParams}
              onParamsChange={handleParamsChange}
              onRun={handleRun}
              loading={loading}
            />
          }
        />
      </div>
    </div>
  );
}
