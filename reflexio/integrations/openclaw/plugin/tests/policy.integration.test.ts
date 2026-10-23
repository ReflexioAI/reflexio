import { createServer } from "node:http";
import { expect, it } from "vitest";
import { ReflexioClient } from "../src/client.js";
import { resolveSettings } from "../src/config.js";
import { ReflexioHooks } from "../src/handlers.js";

it("carries retained policy through hooks and real loopback HTTP", async () => {
  const received: { path: string; body: Record<string, unknown>; authorization?: string }[] = [];
  const server = createServer(async (request, response) => {
    const chunks: Buffer[] = [];
    for await (const chunk of request) chunks.push(Buffer.from(chunk));
    received.push({
      path: request.url!,
      body: JSON.parse(Buffer.concat(chunks).toString()),
      authorization: request.headers.authorization,
    });
    response.setHeader("Content-Type", "application/json");
    response.end(
      JSON.stringify({ success: true, profiles: [], user_playbooks: [], agent_playbooks: [] }),
    );
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    if (!address || typeof address === "string") throw new Error("Missing loopback test address");
    const settings = resolveSettings(
      {
        url: `http://127.0.0.1:${address.port}`,
        apiKey: "test-only",
        search: { top_k: 7, min_prompt_length: 1, timeout_ms: 1700 },
        publish: { max_content_length: 4 },
      },
      {},
    );
    const hooks = new ReflexioHooks(settings, new ReflexioClient(settings), {});
    const context = {
      runId: "policy-run",
      sessionId: "policy-session",
      sessionKey: "agent:main:main",
      agentId: "main",
      workspaceDir: "/policy-workspace",
      trigger: "user",
    };
    await hooks.beforePromptBuild({ prompt: "?" }, context);
    await hooks.agentEnd(
      {
        success: true,
        messages: [
          { role: "user", content: "😀😀😀😀x" },
          { role: "assistant", content: "answer" },
        ],
      },
      context,
    );
    expect(received.map((request) => request.path)).toEqual([
      "/api/search",
      "/api/publish_interaction",
    ]);
    expect(received.every((request) => request.authorization === "Bearer test-only")).toBe(true);
    expect(received[0]!.body).toMatchObject({ query: "?", top_k: 7 });
    expect(received[1]!.body.interaction_data_list).toEqual([
      { role: "User", content: "😀😀😀😀…(truncated)" },
      { role: "Assistant", content: "answ…(truncated)" },
    ]);
  } finally {
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) =>
      server.close((error) => (error ? reject(error) : resolve())),
    );
  }
});
