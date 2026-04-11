"use client";

import { useState, useCallback } from "react";
import { MethodDef } from "@/lib/types";
import { executeApiCall, ExecutionResult } from "@/lib/execution/api-executor";

interface ExecutionState {
  result: ExecutionResult | null;
  loading: boolean;
  error: string | null;
}

export function useExecution() {
  const [state, setState] = useState<ExecutionState>({
    result: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(
    async (
      method: MethodDef,
      params: Record<string, unknown>,
      apiEndpoint: string
    ) => {
      setState({ result: null, loading: true, error: null });

      try {
        const result = await executeApiCall(
          method,
          params,
          apiEndpoint
        );
        setState({ result, loading: false, error: null });
      } catch (err) {
        setState({
          result: null,
          loading: false,
          error: err instanceof Error ? err.message : "Request failed",
        });
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState({ result: null, loading: false, error: null });
  }, []);

  return { ...state, execute, reset };
}
