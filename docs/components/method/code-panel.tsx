"use client";

import { useState, useCallback, useEffect } from "react";
import { useTheme } from "next-themes";
import { Play, RotateCcw, ChevronDown, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ParamForm } from "./param-form";
import { MethodDef } from "@/lib/types";
import { generatePythonCode } from "@/lib/execution/code-generator";
import { parsePythonCode } from "@/lib/execution/code-parser";
import dynamic from "next/dynamic";

const MonacoEditor = dynamic(() => import("@monaco-editor/react"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full min-h-[200px] bg-muted/30">
      <span className="text-xs text-muted-foreground">Loading editor...</span>
    </div>
  ),
});

interface CodePanelProps {
  method: MethodDef;
  params: Record<string, unknown>;
  onParamsChange: (params: Record<string, unknown>) => void;
  onRun: (params: Record<string, unknown>) => void;
  loading: boolean;
}

export function CodePanel({
  method,
  params,
  onParamsChange,
  onRun,
  loading,
}: CodePanelProps) {
  const { resolvedTheme } = useTheme();
  const [showParams, setShowParams] = useState(true);
  const [code, setCode] = useState(() =>
    generatePythonCode(method, params)
  );
  const [codeEdited, setCodeEdited] = useState(false);

  // Regenerate code when params change (but not if user hand-edited)
  useEffect(() => {
    if (!codeEdited) {
      setCode(generatePythonCode(method, params));
    }
  }, [method, params, codeEdited]);

  const handleParamChange = useCallback(
    (name: string, value: unknown) => {
      setCodeEdited(false);
      onParamsChange({ ...params, [name]: value });
    },
    [params, onParamsChange]
  );

  const handleCodeChange = useCallback(
    (value: string | undefined) => {
      if (value !== undefined) {
        setCode(value);
        setCodeEdited(true);
      }
    },
    []
  );

  const handleRun = useCallback(() => {
    if (codeEdited) {
      const parsed = parsePythonCode(code, method);
      if (parsed) {
        onRun(parsed.params);
        return;
      }
    }
    onRun(params);
  }, [codeEdited, code, method, params, onRun]);

  const handleReset = useCallback(() => {
    const emptyParams: Record<string, unknown> = {};
    for (const p of method.params) {
      if (p.default !== undefined) {
        emptyParams[p.name] = p.default;
      }
    }
    setCodeEdited(false);
    onParamsChange(emptyParams);
  }, [method, onParamsChange]);

  // Cmd+Enter shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        handleRun();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleRun]);

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={handleRun}
            disabled={loading}
            className="h-7 text-xs gap-1.5"
          >
            <Play className="h-3 w-3" />
            Run
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            className="h-7 text-xs gap-1.5"
          >
            <RotateCcw className="h-3 w-3" />
            Reset
          </Button>
          <span className="text-[10px] text-muted-foreground hidden sm:block">
            {"\u2318"}+Enter
          </span>
        </div>
      </div>

      {/* Monaco Editor */}
      <div className="flex-1 min-h-[150px]">
        <MonacoEditor
          language="python"
          theme={resolvedTheme === "dark" ? "vs-dark" : "light"}
          value={code}
          onChange={handleCodeChange}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineNumbers: "on",
            scrollBeyondLastLine: false,
            wordWrap: "on",
            padding: { top: 12, bottom: 12 },
            renderLineHighlight: "none",
            overviewRulerLanes: 0,
            hideCursorInOverviewRuler: true,
            overviewRulerBorder: false,
            scrollbar: { verticalSliderSize: 6, horizontalSliderSize: 6 },
          }}
        />
      </div>

      {/* Param Form */}
      <div className="border-t border-border">
        <button
          onClick={() => setShowParams(!showParams)}
          className="flex items-center gap-1.5 w-full px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
        >
          {showParams ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          Parameters ({method.params.length})
        </button>
        {showParams && (
          <div className="px-3 pb-3 max-h-[300px] overflow-y-auto">
            <ParamForm
              params={method.params}
              values={params}
              onChange={handleParamChange}
            />
          </div>
        )}
      </div>
    </div>
  );
}
