// Exercise the real explorer definitions through its Python code generator.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
const ts = createRequire(import.meta.url)("typescript");

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
assert.ok(search);
assert.ok(publish);
const defaults = (method) => Object.fromEntries(method.params
  .filter((param) => param.default !== undefined)
  .map((param) => [param.name, param.default]));
for (const method of [search, publish]) {
  const session = method.params.find((param) => param.name === "session_id");
  assert.ok(session);
  assert.equal(session.required, false);
  assert.equal(session.default, undefined);
  assert.doesNotMatch(generatePythonCode(method, defaults(method)), /session_id=/);
  // A caller chooses one identity for related calls, without a permanent demo default.
  for (const sessionId of ["first-run-session", "second-run-session"]) {
    assert.ok(generatePythonCode(method, {
      ...defaults(method),
      session_id: sessionId,
    }).includes(`session_id="${sessionId}"`));
  }
}

// Exercise the notebook source without executing its paid model calls.
for (const [file, variable, prefix, retrieval] of [
  ["03_playbook.ipynb", "PLAYBOOK_EVAL_SESSION", "playbook_eval", "client.search("],
  ["06_real_world_simulation.ipynb", "ENHANCED_SESSION", "enhanced", "client.search_user_playbooks("],
]) {
  const notebook = JSON.parse(fs.readFileSync(
    path.join(scriptDirectory, "../../notebooks", file), "utf8",
  ));
  const cells = notebook.cells.filter((cell) => cell.cell_type === "code")
    .map((cell) => cell.source.join(""));
  const source = cells.join("\n");
  assert.match(source, /RUN_ID = uuid\.uuid4\(\)\.hex/);
  const declaration = `${variable} = f"${prefix}_{RUN_ID}"`;
  assert.equal(source.split(declaration).length - 1, 1);
  const searchCell = cells.find((cell) => cell.includes(retrieval));
  assert.ok(searchCell.includes(`session_id=${variable}`));
  assert.ok(source.indexOf(declaration) < source.indexOf(retrieval));
  const publishCell = cells.find((cell) =>
    cell.includes("client.publish_interaction(") && cell.includes(`session_id=${variable}`));
  assert.ok(publishCell);
  assert.ok(cells.some((cell) =>
    cell.includes(`client.grade_on_demand(session_id=${variable}`)));
  assert.match(searchCell, /user_id=USER_ID/);
  assert.match(publishCell, /user_id=USER_ID/);
}
console.log("Explorer preserves caller sessions; notebook retrieval, publish and grading share a fresh run identity");
