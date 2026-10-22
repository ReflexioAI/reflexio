// Pins the exact request bodies the client sends. The same fixture is
// validated against Reflexio's server models by
// tests/integrations/test_openclaw_wire_contract.py in the Python package, so a
// field renamed on either side fails one of the two suites.
//
// Regenerate after an intentional wire change: UPDATE_WIRE_FIXTURE=1 npx vitest run

import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ReflexioClient } from "../src/client.js";
import { resolveSettings } from "../src/config.js";

const FIXTURE = fileURLToPath(new URL("./fixtures/wire-requests.json", import.meta.url));

describe("wire contract", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("matches the committed request fixture", async () => {
    const sent: Record<string, { headers: Record<string, string>; body: unknown }> = {};
    vi.stubGlobal("fetch", async (url: URL, init: RequestInit) => {
      sent[url.pathname] = {
        headers: init.headers as Record<string, string>,
        body: JSON.parse(String(init.body)),
      };
      return new Response(JSON.stringify({ success: true }), { status: 200 });
    });
    const client = new ReflexioClient(resolveSettings({ url: "http://h/", apiKey: "k" }, {}));
    await client.search({ query: "q", userId: "u", sessionId: "s", requestId: "r" });
    await client.publish({
      request_id: "r",
      user_id: "u",
      session_id: "s",
      interactions: [
        { role: "User", content: "q" },
        {
          role: "Assistant",
          content: "a",
          tools_used: [{ tool_name: "exec", status: "success", tool_data: { input: { command: "ls" } } }],
          citations: [{ kind: "profile", real_id: "p", tag: "p1-p", title: "t" }],
          retrieved_learnings: [{ kind: "user_playbook", learning_id: "1", exposure_receipt: "x" }],
        },
      ],
    });
    await client.finalizeSession("u", "s");
    await client.listLearnings("u", 20);

    expect(sent["/api/search"]?.headers.Authorization).toBe("Bearer k");
    const bodies = Object.fromEntries(Object.entries(sent).map(([path, v]) => [path, v.body]));
    if (process.env.UPDATE_WIRE_FIXTURE === "1") {
      writeFileSync(FIXTURE, `${JSON.stringify(bodies, null, 2)}\n`);
    }
    expect(bodies).toEqual(JSON.parse(readFileSync(FIXTURE, "utf8")));
  });
});
