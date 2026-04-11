import { MethodDef } from "../types";

interface ParseResult {
  methodName: string;
  params: Record<string, unknown>;
}

export function parsePythonCode(
  code: string,
  methodDef: MethodDef
): ParseResult | null {
  // Find the client method call line
  const callPattern = new RegExp(
    `client\\.([a-z_]+)\\s*\\(([\\s\\S]*?)\\)\\s*$`,
    "m"
  );
  const match = code.match(callPattern);
  if (!match) return null;

  const methodName = match[1];
  const argsStr = match[2].trim();

  if (!argsStr) {
    return { methodName, params: {} };
  }

  const params: Record<string, unknown> = {};

  // Extract key=value kwargs
  const kwargPattern =
    /(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[\s\S]*?\]|\{[\s\S]*?\}|True|False|None|[\d.]+)/g;
  let kwargMatch;
  while ((kwargMatch = kwargPattern.exec(argsStr)) !== null) {
    const key = kwargMatch[1];
    const rawValue = kwargMatch[2];
    params[key] = parsePythonLiteral(rawValue);
  }

  return { methodName, params };
}

function parsePythonLiteral(value: string): unknown {
  if (value === "True") return true;
  if (value === "False") return false;
  if (value === "None") return null;
  if (/^-?\d+$/.test(value)) return parseInt(value, 10);
  if (/^-?\d+\.\d+$/.test(value)) return parseFloat(value);
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }
  if (value.startsWith("[") || value.startsWith("{")) {
    try {
      return JSON.parse(value.replace(/'/g, '"'));
    } catch {
      return value;
    }
  }
  return value;
}
