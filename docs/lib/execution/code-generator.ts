import { MethodDef } from "../types";

export function generatePythonCode(
  method: MethodDef,
  params: Record<string, unknown>
): string {
  if (method.pythonCodeStyle === "requests") {
    return generateRequestsCode(method, params);
  }

  const lines: string[] = [
    "from reflexio import ReflexioClient",
    "",
    'client = ReflexioClient(url_endpoint="http://localhost:8061")',
    "",
  ];

  const args: string[] = [];
  for (const param of method.params) {
    const value = params[param.name];
    if (value === undefined || value === null || value === "") continue;
    args.push(`    ${param.name}=${formatPythonValue(value, param.type)}`);
  }

  if (args.length > 0) {
    lines.push(`result = client.${method.pythonName}(`);
    lines.push(args.join(",\n"));
    lines.push(")");
  } else {
    lines.push(`result = client.${method.pythonName}()`);
  }

  lines.push("print(result)");
  return lines.join("\n");
}

function generateRequestsCode(
  method: MethodDef,
  params: Record<string, unknown>
): string {
  const lines: string[] = [
    "import requests",
    "",
    'base_url = "http://localhost:8061"',
    `url = f"{base_url}${method.endpoint}"`,
  ];
  const queryEntries = method.params
    .map((param) => {
      const value = params[param.name];
      if (value === undefined || value === null || value === "") return null;
      return `    "${param.name}": ${formatPythonValue(value, param.type)}`;
    })
    .filter((entry): entry is string => entry !== null);

  if (queryEntries.length > 0) {
    lines.push("params = {");
    lines.push(queryEntries.join(",\n"));
    lines.push("}");
  } else {
    lines.push("params = {}");
  }

  lines.push("");
  lines.push(
    `response = requests.${method.httpMethod.toLowerCase()}(url, params=params)`
  );
  lines.push("response.raise_for_status()");
  lines.push("print(response.json())");
  return lines.join("\n");
}

function formatPythonValue(value: unknown, type: string): string {
  if (type === "string" || type === "datetime" || type === "enum") {
    return `"${String(value)}"`;
  }
  if (type === "boolean") {
    return value ? "True" : "False";
  }
  if (type === "number") {
    return String(value);
  }
  if (type === "json" || type === "string[]") {
    if (typeof value === "string") {
      // Normalize Python-style literals to JSON for parsing
      const jsonStr = value.replace(/\bNone\b/g, "null").replace(/\bTrue\b/g, "true").replace(/\bFalse\b/g, "false");
      try {
        JSON.parse(jsonStr);
        // Convert JSON literals back to Python style for display
        return jsonStr.replace(/\bnull\b/g, "None").replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False");
      } catch {
        return `"${value}"`;
      }
    }
    return JSON.stringify(value)
      .replace(/\bnull\b/g, "None")
      .replace(/\btrue\b/g, "True")
      .replace(/\bfalse\b/g, "False");
  }
  return String(value);
}
