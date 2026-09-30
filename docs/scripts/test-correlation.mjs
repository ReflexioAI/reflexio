// Exercise the real explorer definitions through its Python code generator.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));

function load(relative) {
  const source = fs.readFileSync(path.join(scriptDirectory, "../lib", relative), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  });
  const exports = {};
  vm.runInNewContext(outputText, { exports });
  return exports;
}
const { generatePythonCode } = load("execution/code-generator.ts");
const { unifiedSearchMethods } = load("methods/unified-search.ts");
const { interactionMethods } = load("methods/interactions.ts");
const search = unifiedSearchMethods.find((method) => method.pythonName === "search");
const publish = interactionMethods.find((method) => method.pythonName === "publish_interaction");
const defaults = (method) => Object.fromEntries(method.params
  .filter((param) => param.default !== undefined)
  .map((param) => [param.name, param.default]));
const searchDefaults = defaults(search);
const publishDefaults = defaults(publish);
assert.equal(typeof searchDefaults.session_id, "string");
assert.ok(searchDefaults.session_id.length > 0);
assert.equal(searchDefaults.session_id, publishDefaults.session_id);
for (const method of [search, publish]) {
  assert.match(generatePythonCode(method, defaults(method)), /session_id="example-session"/);
}
// A user can override both defaults with the real serving identity.
assert.match(generatePythonCode(search, { query: "task", session_id: "actual-session" }), /session_id="actual-session"/);
console.log("Explorer generated search and publish examples share an active session ID");
