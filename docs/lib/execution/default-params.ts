import { MethodDef } from "../types";

export function initialParams(
  method: MethodDef,
  sessionId: string
): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const param of method.params) {
    if (param.default !== undefined) params[param.name] = param.default;
    if (
      param.name === "session_id" && sessionId &&
      ["search", "publish_interaction"].includes(method.pythonName)
    ) params.session_id = sessionId;
  }
  return params;
}
