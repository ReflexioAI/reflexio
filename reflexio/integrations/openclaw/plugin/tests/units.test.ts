import { afterEach, describe, expect, it, vi } from "vitest";
import { ReflexioClient, ReflexioError } from "../src/client.js";
import { resolveSettings } from "../src/config.js";
import { requestIdFor, resolveUserId } from "../src/identity.js";
import { renderInjection, resolveCitations } from "../src/render.js";
import { currentTurn } from "../src/transcript.js";

describe("identity", () => {
  it("separates workspaces that share a basename", () => {
    const a = resolveUserId({ agentId: "main", workspaceDir: "/a/workspace" });
    const b = resolveUserId({ agentId: "main", workspaceDir: "/b/workspace" });
    expect(a).toMatch(/^openclaw:main:workspace-[0-9a-f]{8}$/);
    expect(a).not.toBe(b);
  });

  it("separates agents sharing a workspace", () => {
    const main = resolveUserId({ agentId: "main", workspaceDir: "/w" });
    const ops = resolveUserId({ agentId: "ops", workspaceDir: "/w" });
    expect(main).not.toBe(ops);
    expect(resolveUserId({})).toBeUndefined();
  });

  it("derives request ids deterministically per run", () => {
    expect(requestIdFor("agent:main:main", "r1")).toBe(requestIdFor("agent:main:main", "r1"));
    expect(requestIdFor("agent:main:main", "r1")).not.toBe(requestIdFor("agent:main:main", "r2"));
  });
});

describe("settings", () => {
  it("prefers plugin config over env and normalises the URL", () => {
    const s = resolveSettings(
      { url: "https://reflexio.example/api" },
      { REFLEXIO_URL: "http://env/", REFLEXIO_API_KEY: "k" },
    );
    expect(s.url).toBe("https://reflexio.example/api/");
    expect(s.apiKey).toBe("k");
    expect(resolveSettings({}, {}).url).toBe("http://127.0.0.1:8061/");
  });
});

describe("transcript", () => {
  it("takes the last user turn, deduplicates repeated blocks and ignores runtime context", () => {
    const turn = currentTurn([
      { role: "user", content: [{ type: "text", text: "q" }, { type: "text", text: "q" }] },
      { role: "user", content: "host note", runtimeContext: { retained: true } },
      { role: "assistant", content: [{ type: "thinking", thinking: "hmm" }, { type: "text", text: "a" }] },
    ]);
    expect(turn).toEqual({ user: "q", assistant: "a", tools: [], fullAssistant: "a" });
  });

  it("redacts secrets at any depth and in the common shapes", () => {
    const turn = currentTurn([
      { role: "user", content: "deploy with sk-proj-abcdefghijklmnopqrstuv please" },
      {
        role: "assistant",
        content: [
          {
            type: "toolCall",
            id: "1",
            name: "exec",
            arguments: {
              command: "API_TOKEN=Ab3dEf6hIj9kLm2nOp5qRs8t ./go --token=ghp_abcdefghijklmnopqrstuvwxyz0123",
              env: { OPENAI_API_KEY: "sk-proj-abcdefghijklmnopqrstuv" },
              headers: ["Authorization: Bearer 0123456789abcdef0123456789abcdef"],
            },
          },
        ],
      },
    ]);
    const data = JSON.stringify(turn?.tools[0]?.tool_data);
    expect(data).not.toMatch(/Ab3dEf6h|ghp_abc|sk-proj|0123456789abcdef/);
    expect(data).toContain("API_TOKEN=<redacted:24>");
    expect(turn?.user).toBe("deploy with <redacted:30> please");
  });

  it("masks values under credential-named keys, whatever they look like", () => {
    const turn = currentTurn([
      { role: "user", content: "connect" },
      {
        role: "assistant",
        content: [
          {
            type: "toolCall",
            id: "1",
            name: "exec",
            arguments: {
              env: { DATABASE_PASSWORD: "veryprivatepassword123", LOG_LEVEL: "info" },
              headers: { Authorization: "hunter2hunter2", "x-api-key": "abc", Accept: "json" },
              cookie: { session: "s" },
            },
          },
        ],
      },
    ]);
    const input = (turn?.tools[0]?.tool_data as { input: Record<string, Record<string, unknown>> }).input;
    expect(input.env).toEqual({ DATABASE_PASSWORD: "<redacted:22>", LOG_LEVEL: "info" });
    expect(input.headers).toEqual({ Authorization: "<redacted:14>", "x-api-key": "<redacted:3>", Accept: "json" });
    expect(input.cookie).toBe("<redacted>");
  });

  it("never passes an uninspected subtree through", () => {
    let nested: Record<string, unknown> = { apiKey: "sk-abcdefghijklmnopqrstuvwx", note: "plain" };
    for (let i = 0; i < 9; i++) nested = { level: nested };
    const turn = currentTurn([
      { role: "user", content: "x" },
      { role: "assistant", content: [{ type: "toolCall", id: "1", name: "exec", arguments: nested }] },
    ]);
    expect(JSON.stringify(turn?.tools[0]?.tool_data)).not.toContain("sk-abc");
    expect(JSON.stringify(turn?.tools[0]?.tool_data)).toContain("<truncated>");
  });

  it("leaves ordinary names alone", () => {
    const turn = currentTurn([{ role: "user", content: "authorName: Shakespeare, keyboard=qwertyuiop" }]);
    expect(turn?.user).toBe("authorName: Shakespeare, keyboard=qwertyuiop");
  });

  it("bounds tool data size", () => {
    const turn = currentTurn([
      { role: "user", content: "edit" },
      {
        role: "assistant",
        content: [{ type: "toolCall", id: "1", name: "edit", arguments: { edits: Array(40).fill({ newText: "x".repeat(5000) }) } }],
      },
    ]);
    expect(JSON.stringify(turn?.tools[0]?.tool_data).length).toBeLessThan(2300);
  });

  it("does not treat host-authored user messages as the user's turn", () => {
    const turn = currentTurn([
      { role: "user", content: "real question" },
      { role: "user", content: "<<<BEGIN_OPENCLAW_INTERNAL_CONTEXT>>>", runtimeContextCarrier: true },
      { role: "user", content: "operator note", operatorMessage: { kind: "note" } },
      { role: "assistant", content: [{ type: "text", text: "answer" }] },
    ]);
    expect(turn?.user).toBe("real question");
  });
});

describe("render", () => {
  const hits = {
    profiles: [{ profile_id: "abc-123", content: "Prefers metric units. Always." }],
    user_playbooks: [],
    agent_playbooks: [{ agent_playbook_id: 7, content: "Check CI first", trigger: "on failures" }],
  };

  it("tags playbooks before profiles and keeps a registry", () => {
    const { markdown, registry } = renderInjection(hits);
    expect(markdown).toContain("- [oc:s-ca9254] Check CI first _(when: on failures)_");
    expect(markdown).toContain("- [oc:p-89d819] Prefers metric units. Always.");
    expect(registry.map((e) => [e.tag, e.kind, e.title])).toEqual([
      ["s-ca9254", "agent_playbook", "Check CI first"],
      ["p-89d819", "profile", "Prefers metric units"],
    ]);
  });

  it("renders nothing for empty hits", () => {
    expect(renderInjection({ profiles: [], user_playbooks: [], agent_playbooks: [] }).markdown).toBe("");
  });

  it("only honours the final marker line and drops unknown tags", () => {
    const { registry } = renderInjection(hits);
    expect(resolveCitations("I used [oc:s-ca9254] here.", registry)).toEqual([]);
    const text = "ok\n✨ 2 Reflexio learnings applied [oc:s-ca9254, p-000000]";
    expect(resolveCitations(text, registry).map((c) => c.tag)).toEqual(["s-ca9254"]);
  });

  it("gives distinct learnings distinct tags, whatever their rank or kind", () => {
    const tags = (hits: Parameters<typeof renderInjection>[0]) =>
      renderInjection(hits).registry.map((e) => e.tag);
    const [a] = tags({ profiles: [], agent_playbooks: [], user_playbooks: [{ user_playbook_id: 10001, content: "a" }] });
    const [b] = tags({ profiles: [], agent_playbooks: [], user_playbooks: [{ user_playbook_id: 10002, content: "b" }] });
    const [agent, user] = tags({
      profiles: [],
      agent_playbooks: [{ agent_playbook_id: 7, content: "x" }],
      user_playbooks: [{ user_playbook_id: 7, content: "y" }],
    });
    expect(a).not.toBe(b);
    expect(agent).not.toBe(user);
    // Stable: the same learning gets the same tag on a later turn.
    expect(tags({ profiles: [], agent_playbooks: [], user_playbooks: [{ user_playbook_id: 10001, content: "a" }] })[0]).toBe(a);
  });
});

describe("client", () => {
  afterEach(() => vi.unstubAllGlobals());
  const client = () => new ReflexioClient(resolveSettings({ url: "http://h/" }, {}));
  const req = { request_id: "r", user_id: "u", session_id: "s", interactions: [] };
  const respond = (status: number, body: unknown) =>
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify(body), { status }));

  it("reports a request the server already admitted as delivered", async () => {
    respond(200, { success: false, message: "request_id 'r' already exists" });
    await expect(client().publish(req, 1000)).resolves.toBeUndefined();
  });

  it("marks 4xx (but not 408/429) as rejected and other failures as retryable", async () => {
    respond(422, { detail: "bad" });
    await expect(client().publish(req, 1000)).rejects.toSatisfy((e: ReflexioError) => e.rejected);
    respond(429, { detail: "slow down" });
    await expect(client().publish(req, 1000)).rejects.toSatisfy((e: ReflexioError) => !e.rejected);
    respond(200, { success: false, message: "storage unavailable" });
    await expect(client().publish(req, 1000)).rejects.toSatisfy((e: ReflexioError) => !e.rejected);
  });
});

describe("installation policy ownership", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("retains independent search and publication choices", async () => {
    const settings = resolveSettings(
      {
        search: { top_k: 7, min_prompt_length: 1, timeout_ms: 1700 },
        publish: { max_content_length: 23 },
      },
      {},
    );
    expect(settings).toMatchObject({
      topK: 7,
      minPromptLength: 1,
      searchTimeoutMs: 1700,
      maxContentLength: 23,
    });
    const fetch = vi.fn<typeof globalThis.fetch>(
      async () =>
        new Response(
          JSON.stringify({ success: true, profiles: [], user_playbooks: [], agent_playbooks: [] }),
        ),
    );
    vi.stubGlobal("fetch", fetch);
    await new ReflexioClient(settings).search({
      query: "why",
      userId: "u",
      sessionId: "s",
      requestId: "r",
    });
    expect(JSON.parse(fetch.mock.calls[0]![1]!.body as string).top_k).toBe(7);
  });

  it("applies the selected content limit without clipping citation identity", () => {
    const selected = resolveSettings({ publish: { max_content_length: 4 } }, {});
    const turn = currentTurn(
      [
        { role: "user", content: "question" },
        { role: "assistant", content: "answer [oc:p-abcd12]" },
      ],
      undefined,
      selected.maxContentLength,
    )!;
    expect(turn.user).toBe("ques…(truncated)");
    expect(turn.assistant).toBe("answ…(truncated)");
    expect(turn.fullAssistant).toBe("answer [oc:p-abcd12]");
  });

  it.each([-1, 1.5, true, "7", null])("rejects malformed custom policy %s", (value) => {
    expect(() => resolveSettings({ search: { top_k: value } }, {})).toThrow();
  });
});

describe("search timer bounds", () => {
  it.each([0, 1.5, 2147482648, Number.NaN, Number.POSITIVE_INFINITY])("rejects an invalid hook deadline %s", (timeout_ms) => {
    expect(() => resolveSettings({ search: { timeout_ms } }, {})).toThrow("search.timeout_ms");
  });
});
