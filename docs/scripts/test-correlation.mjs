// Keep authenticated execution covered by the existing docs CI entry point.
import "./test-execution-auth.mjs";

// Exercise the real explorer definitions through its Python code generator.
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
const ts = createRequire(import.meta.url)("typescript");

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));

const nativeRequire = createRequire(import.meta.url);

function load(relative, imports = {}) {
  const source = fs.readFileSync(path.join(scriptDirectory, "../lib", relative), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports,
    require: (name) => imports[name] ?? nativeRequire(name),
  });
  return exports;
}

// The interactive configurator must not advertise or re-send evaluation windows.
const configSchema = load("config-schema.ts");
const evaluation = configSchema.defaultAgentSuccess();
for (const key of ["window_size_override", "stride_size_override"]) {
  assert.equal(key in evaluation, false);
}
const config = configSchema.defaultConfig();
config.agent_success_config = {
  ...evaluation,
  success_definition_prompt: "Keep the session rubric.",
  sampling_rate: 0.25,
  evaluation_only_sampling_rate: 0.75,
  retrieved_learning_sampling_rate: 0.5,
  request_sources_enabled: ["api"],
  window_size_override: 23,
  stride_size_override: 7,
  batch_size_override: 99,
};
config.profile_extractor_config = configSchema.defaultProfileExtractor();
config.profile_extractor_config.extraction_definition_prompt = "Keep";
config.profile_extractor_config.window_size_override = 23;
const serialized = configSchema.serializeConfig(config);
assert.equal("extraction_preset" in serialized, false);
assert.equal("extraction_preset" in configSchema.defaultConfig(), false);
for (const geometry of Object.values(configSchema.PRESET_VALUES)) {
  const edited = configSchema.hydrateConfig({ ...config, extraction_preset: "stale-browser-state" });
  [edited.window_size, edited.stride_size] = geometry;
  const saved = configSchema.serializeConfig(edited);
  assert.equal("extraction_preset" in saved, false);
  assert.equal(saved.window_size, geometry[0]);
  assert.equal(saved.stride_size, geometry[1]);
}

for (const key of ["window_size_override", "stride_size_override", "batch_size_override"]) {
  assert.equal(key in serialized.agent_success_config, false);
}
assert.equal(serialized.agent_success_config.success_definition_prompt, "Keep the session rubric.");
assert.equal(serialized.agent_success_config.sampling_rate, 0.25);
assert.equal(serialized.agent_success_config.evaluation_only_sampling_rate, 0.75);
assert.equal(serialized.agent_success_config.retrieved_learning_sampling_rate, 0.5);
assert.equal(serialized.agent_success_config.request_sources_enabled[0], "api");
assert.equal(serialized.profile_extractor_config.window_size_override, 23);

// A subset editor must preserve the fetched snapshot through full replacement.
const fetched = {
  ...configSchema.defaultConfig(),
  governance_retention: {
    audit_events_max_age_days: 0,
    audit_events_delete_batch_limit: 2000,
  },
  lineage_gc: { enabled: false, tombstone_grace_window_days: 120 },
  future_policy: { enabled: false, budget: 37 },
  storage_config: { db_path: "  ", future_storage_option: "keep" },
  llm_config: { generation_model_name: "keep-model", future_llm_option: 37 },
  api_key_config: {
    openai: { api_key: null, azure_config: {
      api_key: "synthetic-azure-key", endpoint: "https://example.openai.azure.com/",
      api_version: "2024-02-15-preview", deployment_name: "ignored-legacy-deployment",
      extension: { deployment_name: "opaque metadata" },
    } },
    dashscope: { api_key: "test-key", api_base: "", future_provider_option: 37 },
  },
};
const before = JSON.stringify(fetched);
const edited = configSchema.hydrateConfig(fetched);
edited.agent_context_prompt = "An unrelated edit";
const roundtrip = JSON.parse(JSON.stringify(configSchema.serializeConfig(edited)));
assert.deepEqual(roundtrip.governance_retention, fetched.governance_retention);
assert.deepEqual(roundtrip.lineage_gc, fetched.lineage_gc);
assert.deepEqual(roundtrip.future_policy, fetched.future_policy);
const expectedOpenAI = JSON.parse(JSON.stringify(fetched.api_key_config.openai));
delete expectedOpenAI.azure_config.deployment_name;
assert.deepEqual(roundtrip.api_key_config.openai, expectedOpenAI);
assert.equal("deployment_name" in roundtrip.api_key_config.openai.azure_config, false);
assert.equal(roundtrip.agent_context_prompt, "An unrelated edit");
assert.equal(roundtrip.storage_config.db_path, null);
assert.equal(roundtrip.storage_config.future_storage_option, "keep");
assert.equal(roundtrip.llm_config.future_llm_option, 37);
assert.equal(roundtrip.api_key_config.dashscope.future_provider_option, 37);
assert.equal(roundtrip.api_key_config.dashscope.api_base, null);
assert.equal(JSON.stringify(fetched), before);
// A snapshot from another storage implementation must not become SQLite.
const remote = { type: "postgres", db_url: "postgresql://local-test", schema: "tenant" };
assert.deepEqual(
  configSchema.serializeConfig(configSchema.hydrateConfig({ ...fetched, storage_config: remote })).storage_config,
  remote,
);
// Explicitly clearing a represented optional section remains a real clear.
edited.llm_config = null;
assert.equal(configSchema.serializeConfig(edited).llm_config, null);

const { generatePythonCode } = load("execution/code-generator.ts");
const { initialParams } = load("execution/default-params.ts");
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
    const params = initialParams(method, sessionId);
    assert.equal(params.session_id, sessionId);
    assert.ok(generatePythonCode(method, params).includes(`session_id="${sessionId}"`));
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
  if (file === "06_real_world_simulation.ipynb") {
    // Every injected entity type must share the serving identity, not just playbooks.
    execFileSync("python3", ["-c", `
import json, sys
from types import SimpleNamespace
code = json.load(sys.stdin)
for session_id in ["first-run", "second-run"]:
    calls = []
    def profiles(request):
        calls.append(("profiles", request["user_id"], request["session_id"]))
        return SimpleNamespace(user_profiles=[])
    def playbooks(**kwargs):
        calls.append(("playbooks", kwargs["user_id"], kwargs["session_id"]))
        return SimpleNamespace(user_playbooks=[])
    client = SimpleNamespace(search_user_profiles=profiles, search_user_playbooks=playbooks)
    exec(code, {"client": client, "USER_ID": "demo", "ENHANCED_SESSION": session_id, "rprint": lambda *args: None})
    assert calls == [("profiles", "demo", session_id), ("playbooks", "demo", session_id)]
`], { input: JSON.stringify(searchCell) });
  }
  if (file === "03_playbook.ipynb") {
    const selectionCell = cells.find((cell) => cell.includes("selected_playbook ="));
    assert.ok(selectionCell);
    // Execute the actual selection/guard code with an unrelated listing present.
    // Stop before prompting or any paid model calls.
    const selectionCode = selectionCell.split("user_message =")[0];
    execFileSync("python3", ["-c", `
import json, sys
from types import SimpleNamespace
code = json.load(sys.stdin)
served = SimpleNamespace(user_playbook_id="served")
unrelated = SimpleNamespace(user_playbook_id="unrelated")
client = SimpleNamespace(get_user_playbooks=lambda **kwargs: SimpleNamespace(user_playbooks=[unrelated]))
for success, results in [(True, [served]), (True, []), (False, [served])]:
    namespace = {"unified": SimpleNamespace(success=success, user_playbooks=results, msg="search failed"), "client": client, "USER_ID": "demo"}
    try:
        exec(code, namespace)
    except RuntimeError:
        assert not success or not results
        assert "selected_playbook" not in namespace
    else:
        assert success and results
        assert namespace["selected_playbook"] is served
`], { input: JSON.stringify(selectionCode) });
  }
}

// Invoke the real editor Run callback with controlled hook state. Browser checks
// cover DOM/state transitions; this regression pins editor-to-form synchronization.
let hookIndex = 0;
const actions = [];
const editedCode = 'client.search(query="refund", session_id="editor-session")';
const { CodePanel } = load("../components/method/code-panel.tsx", {
  react: {
    useState: () => [[true, editedCode, true][hookIndex++], () => {}],
    useCallback: (callback) => callback,
    useEffect: () => {},
    useMemo: (factory) => factory(),
  },
  "next-themes": { useTheme: () => ({ resolvedTheme: "light" }) },
  "next/dynamic": { default: () => () => null },
  "@/components/ui/button": { Button: "button" },
  "./param-form": { ParamForm: "param-form" },
  "@/hooks/use-settings": { useSettings: () => ({ sessionId: "original-session" }) },
  "@/lib/execution/default-params": { initialParams },
  "@/lib/execution/code-generator": { generatePythonCode },
  "@/lib/execution/code-parser": load("execution/code-parser.ts"),
});
const panel = CodePanel({
  method: search,
  params: initialParams(search, "original-session"),
  onParamsChange: (params) => actions.push(["update", params.session_id]),
  onRun: (params) => actions.push(["execute", params.session_id]),
  loading: false,
});
function findRun(element) {
  if (!element || typeof element !== "object") return null;
  const children = [element.props?.children].flat();
  if (element.type === "button" && children.includes("Run")) return element;
  for (const child of children) {
    const found = findRun(child);
    if (found) return found;
  }
  return null;
}
const run = findRun(panel);
assert.ok(run, "the editor must expose its Run action");
run.props.onClick();
assert.deepEqual(actions, [["update", "editor-session"], ["execute", "editor-session"]]);

console.log("Explorer preserves caller sessions; notebook retrieval, publish and grading share a fresh run identity");

// A credential transport cannot revive the retired model pin or lose a role choice.
const gatewayConfig = configSchema.hydrateConfig({
  ...configSchema.defaultConfig(),
  api_key_config: { custom_endpoint: { api_key: "local-key", api_base: "https://gateway.invalid/v1" } },
  llm_config: { generation_model_name: "openai/chosen", should_run_model_name: "openai/check", embedding_model_name: "local/minilm-l6-v2" },
  opaque: { keep: true },
});
const gatewaySaved = configSchema.serializeConfig(gatewayConfig);
assert.equal("model" in gatewaySaved.api_key_config.custom_endpoint, false);
assert.equal(gatewaySaved.llm_config.generation_model_name, "openai/chosen");
assert.equal(gatewaySaved.llm_config.should_run_model_name, "openai/check");
assert.deepEqual(gatewaySaved.opaque, { keep: true });
assert.throws(() => configSchema.hydrateConfig({
  ...gatewayConfig, api_key_config: { custom_endpoint: { ...gatewayConfig.api_key_config.custom_endpoint, model: "openai/old-pin" } },
}), /retired custom endpoint model/);
