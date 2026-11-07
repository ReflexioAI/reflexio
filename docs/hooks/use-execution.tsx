"use client";

import { useCallback, useState } from "react";
import { type ExecutionResult, executeApiCall } from "@/lib/execution/api-executor";
import type { MethodDef } from "@/lib/types";

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
			apiEndpoint: string,
			apiKey = "",
		) => {
			setState({ result: null, loading: true, error: null });

			try {
				const result = await executeApiCall(method, params, apiEndpoint, apiKey);
				setState({ result, loading: false, error: null });
			} catch (err) {
				setState({
					result: null,
					loading: false,
					error: err instanceof Error ? err.message : "Request failed",
				});
			}
		},
		[],
	);

	const reset = useCallback(() => {
		setState({ result: null, loading: false, error: null });
	}, []);

	return { ...state, execute, reset };
}
