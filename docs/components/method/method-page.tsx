"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { MethodDef } from "@/lib/types";
import { MethodBadge } from "./method-badge";
import { SplitPanel } from "@/components/layout/split-panel";
import { CodePanel } from "./code-panel";
import { ResultsPanel } from "./results-panel";
import { useSettings } from "@/hooks/use-settings";
import { useExecution } from "@/hooks/use-execution";

interface MethodPageProps {
  method: MethodDef;
}

export function MethodPage({ method }: MethodPageProps) {
  const { apiEndpoint } = useSettings();
  const { result, loading, error, execute } = useExecution();

  const [params, setParams] = useState<Record<string, unknown>>(() => {
    const defaults: Record<string, unknown> = {};
    for (const p of method.params) {
      if (p.default !== undefined) {
        defaults[p.name] = p.default;
      }
    }
    return defaults;
  });

  const handleRun = useCallback(
    (runParams: Record<string, unknown>) => {
      execute(method, runParams, apiEndpoint);
    },
    [method, apiEndpoint, execute]
  );

  // Auto-run on mount if no required parameters
  const hasAutoRun = useRef(false);
  useEffect(() => {
    if (hasAutoRun.current) return;
    const hasRequired = method.params.some((p) => p.required);
    if (!hasRequired && apiEndpoint) {
      hasAutoRun.current = true;
      handleRun(params);
    }
  }, [method, apiEndpoint, handleRun, params]);

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
              params={params}
              onParamsChange={setParams}
              onRun={handleRun}
              loading={loading}
            />
          }
        />
      </div>
    </div>
  );
}
