import { MethodDef } from "../types";

/** Convert Python-style literals (None, True, False) to JSON equivalents */
function pythonToJson(value: string): string {
  return value.replace(/\bNone\b/g, "null").replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false");
}

export interface ExecutionResult {
  data: unknown;
  status: number;
  duration: number;
}

export async function executeApiCall(
  method: MethodDef,
  params: Record<string, unknown>,
  apiEndpoint: string
): Promise<ExecutionResult> {
  const start = performance.now();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };

  let url = `${apiEndpoint.replace(/\/$/, "")}${method.endpoint}`;
  let body: string | undefined;

  if (method.requestStyle === "json_body") {
    // Build JSON body, filtering out empty/undefined values
    const bodyObj: Record<string, unknown> = {};
    for (const param of method.params) {
      const value = params[param.name];
      if (value === undefined || value === null || value === "") continue;

      // Parse JSON strings for json-type params
      if (param.type === "json" && typeof value === "string") {
        try {
          bodyObj[param.name] = JSON.parse(pythonToJson(value));
        } catch {
          bodyObj[param.name] = value;
        }
      } else if (param.type === "number" && typeof value === "string") {
        bodyObj[param.name] = Number(value);
      } else if (param.type === "boolean" && typeof value === "string") {
        bodyObj[param.name] = value === "true" || value === "True";
      } else {
        bodyObj[param.name] = value;
      }
    }

    // When there's a single json param, send its value directly as the body
    // (FastAPI expects the model fields at the top level, not wrapped in a key)
    const jsonParams = method.params.filter((p) => p.type === "json");
    if (jsonParams.length === 1 && bodyObj[jsonParams[0].name] !== undefined) {
      body = JSON.stringify(bodyObj[jsonParams[0].name]);
    } else {
      body = JSON.stringify(bodyObj);
    }
  } else if (method.requestStyle === "query_params") {
    const searchParams = new URLSearchParams();
    for (const param of method.params) {
      const value = params[param.name];
      if (value === undefined || value === null || value === "") continue;
      searchParams.set(param.name, String(value));
    }
    const qs = searchParams.toString();
    if (qs) url += `?${qs}`;
  }

  const response = await fetch(url, {
    method: method.httpMethod,
    headers,
    body: method.requestStyle === "json_body" ? body : undefined,
  });

  const duration = performance.now() - start;
  const data = await response.json();

  return { data, status: response.status, duration };
}
