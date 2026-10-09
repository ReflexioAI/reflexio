// Exercise the real executor: credentials belong in headers, never payloads,
// and clearing a key must not reuse an earlier Authorization header.
import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";
import ts from "typescript";

const source = fs.readFileSync(new URL("../lib/execution/api-executor.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
});
const exports = {};
const requests = [];
vm.runInNewContext(outputText, {
  exports,
  performance,
  URLSearchParams,
  fetch: async (url, options) => {
    requests.push({ url, ...options });
    return { status: 200, json: async () => ({ success: true }) };
  },
});
const method = {
  endpoint: "/api/publish_interaction", httpMethod: "POST", requestStyle: "json_body",
  params: [{ name: "user_id", type: "string" }],
};
for (const key of ["", "isolated-test-key", ""]) {
  await exports.executeApiCall(method, { user_id: "probe" }, "http://localhost:8061", key);
}
assert.equal(requests[0].headers.Authorization, undefined);
assert.equal(requests[1].headers.Authorization, "Bearer isolated-test-key");
assert.equal(requests[2].headers.Authorization, undefined);
assert.equal(requests[1].body, JSON.stringify({ user_id: "probe" }));
assert.ok(!requests[1].url.includes("isolated-test-key"));
console.log("Authenticated executor regression passed.");
