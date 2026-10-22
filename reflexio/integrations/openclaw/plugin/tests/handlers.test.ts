import { afterEach, describe, expect, it, vi } from "vitest";
import { type PublishRequest, type ReflexioClient, ReflexioError, type SearchHits } from "../src/client.js";
import { resolveSettings } from "../src/config.js";
import { type HookContext, ReflexioHooks } from "../src/handlers.js";

class FakeClient {
  baseUrl = "http://fake/";
  searches: { requestId: string; userId: string; sessionId: string; query: string }[] = [];
  published: PublishRequest[] = [];
  attempts = 0;
  finalized: [string, string][] = [];
  // Next publish outcomes, consumed in order; "ok" when empty.
  publishOutcomes: (Error | "ok")[] = [];
  hits: SearchHits = {
    profiles: [{ profile_id: "prof-1", content: "Answer in one sentence." }],
    user_playbooks: [{ user_playbook_id: 42, content: "Run tests before committing.", exposure_receipt: "rcpt" }],
    agent_playbooks: [],
  };

  async search(args: { requestId: string; userId: string; sessionId: string; query: string }) {
    this.searches.push(args);
    return this.hits;
  }
  async publish(req: PublishRequest) {
    this.attempts++;
    const outcome = this.publishOutcomes.shift() ?? "ok";
    if (outcome !== "ok") throw outcome;
    this.published.push(structuredClone(req));
  }
  async finalizeSession(userId: string, sessionId: string) {
    this.finalized.push([userId, sessionId]);
  }
  async health() {
    return true;
  }
  async listLearnings(): Promise<SearchHits> {
    return this.hits;
  }
}

const ctx = (runId: string, extra: Partial<HookContext> = {}): HookContext => ({
  runId,
  agentId: "main",
  sessionKey: "agent:main:main",
  sessionId: "sess-1",
  workspaceDir: "/home/u/.openclaw/workspace",
  trigger: "user",
  ...extra,
});

const history = [
  { role: "user", content: "earlier question" },
  { role: "assistant", content: [{ type: "text", text: "earlier answer" }] },
];

const turn = (user: string, assistant: string) => [
  ...history,
  { role: "user", content: [{ type: "text", text: user }] },
  {
    role: "assistant",
    content: [{ type: "toolCall", id: "t1", name: "exec", arguments: { command: "npm test" } }],
  },
  { role: "toolResult", toolCallId: "t1", toolName: "exec", content: [{ type: "text", text: "1 failed" }], isError: true },
  { role: "assistant", content: [{ type: "text", text: assistant }] },
];

function setup(isIncognito?: (key: string) => boolean) {
  const client = new FakeClient();
  const hooks = new ReflexioHooks(resolveSettings({}, {}), client as unknown as ReflexioClient, {}, isIncognito);
  return { client, hooks };
}

describe("ReflexioHooks", () => {
  it("publishes a run under the same request_id its search exposure used", async () => {
    const { client, hooks } = setup();
    const injected = await hooks.beforePromptBuild({ prompt: "please fix the tests", messages: history }, ctx("run-1"));
    expect(injected?.prependContext).toContain("[oc:p-104520]");
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("please fix the tests", "Done.") }, ctx("run-1"));

    expect(client.published).toHaveLength(1);
    const req = client.published[0]!;
    expect(req.request_id).toBe(client.searches[0]!.requestId);
    expect(req.user_id).toBe(client.searches[0]!.userId);
    expect(req.session_id).toBe("sess-1");
    expect(req.interactions.map((i) => i.role)).toEqual(["User", "Assistant"]);
    expect(req.interactions[0]!.content).toBe("please fix the tests");
    const assistant = req.interactions[1]!;
    expect(assistant.content).toBe("Done.");
    expect(assistant.tools_used).toEqual([
      { tool_name: "exec", status: "error", tool_data: { input: { command: "npm test" }, output: "1 failed" } },
    ]);
    expect(assistant.retrieved_learnings).toEqual([
      { kind: "user_playbook", learning_id: "42", exposure_receipt: "rcpt" },
      { kind: "profile", learning_id: "prof-1" },
    ]);
  });

  it("searches on the user's own words, not the channel-prefixed prompt", async () => {
    const { client, hooks } = setup();
    await hooks.beforePromptBuild(
      { prompt: "[Discord #ops from alice] deploy it", currentUserMessage: "deploy it" },
      ctx("run-1"),
    );
    expect(client.searches[0]!.query).toBe("deploy it");
  });

  it("keeps steering messages and the run's tool calls in the published turn", async () => {
    const { client, hooks } = setup();
    await hooks.beforePromptBuild({ prompt: "fix the build", messages: history }, ctx("run-1"));
    const messages = [
      ...history,
      { role: "user", content: "fix the build" },
      { role: "assistant", content: [{ type: "toolCall", id: "t1", name: "exec", arguments: { command: "make" } }] },
      { role: "toolResult", toolCallId: "t1", toolName: "exec", content: [{ type: "text", text: "ok" }], isError: false },
      { role: "user", content: "also use clang" },
      { role: "assistant", content: [{ type: "text", text: "Built with clang." }] },
    ];
    await hooks.agentEnd({ runId: "run-1", success: true, messages }, ctx("run-1"));
    const [user, assistant] = client.published[0]!.interactions;
    expect(user!.content).toBe("fix the build\n\nalso use clang");
    expect(assistant!.tools_used?.map((t) => t.tool_name)).toEqual(["exec"]);
    expect(assistant!.retrieved_learnings).toHaveLength(2);
  });

  it("finds the run start when the host's history already holds the prompt", async () => {
    const { client, hooks } = setup();
    const admitted = [...history, { role: "user", content: "fix the build" }];
    await hooks.beforePromptBuild({ prompt: "fix the build", messages: admitted }, ctx("run-1"));
    const messages = [
      ...admitted,
      { role: "assistant", content: [{ type: "toolCall", id: "t1", name: "exec", arguments: { command: "make" } }] },
      { role: "toolResult", toolCallId: "t1", toolName: "exec", content: [{ type: "text", text: "ok" }], isError: false },
      { role: "user", content: "also use clang" },
      { role: "assistant", content: [{ type: "text", text: "Built." }] },
    ];
    await hooks.agentEnd({ runId: "run-1", success: true, messages }, ctx("run-1"));
    const [user, assistant] = client.published[0]!.interactions;
    expect(user!.content).toBe("fix the build\n\nalso use clang");
    expect(assistant!.tools_used?.map((t) => t.tool_name)).toEqual(["exec"]);
  });

  it("does not attribute one run's injections to the next run", async () => {
    const { client, hooks } = setup();
    await hooks.beforePromptBuild({ prompt: "first prompt" }, ctx("run-1"));
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("first prompt", "ok") }, ctx("run-1"));
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("second", "ok") }, ctx("run-2"));
    expect(client.published[1]!.interactions[1]!.retrieved_learnings).toBeUndefined();
  });

  it("resolves a final citation line against injected tags", async () => {
    const { client, hooks } = setup();
    await hooks.beforePromptBuild({ prompt: "fix it now" }, ctx("run-1"));
    const reply = "Fixed.\n✨ 1 Reflexio learning applied [oc:s-12ae19]";
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("fix it now", reply) }, ctx("run-1"));
    expect(client.published[0]!.interactions[1]!.citations).toEqual([
      { kind: "user_playbook", real_id: "42", tag: "s-12ae19", title: "Run tests before committing." },
    ]);
  });

  it("reads the citation from a long reply's final line, past the publish clip", async () => {
    const { client, hooks } = setup();
    await hooks.beforePromptBuild({ prompt: "write it all" }, ctx("run-1"));
    const reply = `${"x".repeat(12_000)}\n✨ 1 Reflexio learning applied [oc:s-12ae19]`;
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("write it all", reply) }, ctx("run-1"));
    const assistant = client.published[0]!.interactions[1]!;
    expect(assistant.content.length).toBeLessThan(reply.length);
    expect(assistant.citations?.map((c) => c.tag)).toEqual(["s-12ae19"]);
  });

  it("queues a failed publish and retries it with the same request_id", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [new Error("ECONNREFUSED")];
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    expect(hooks.stats.failures.publish).toBe(1);
    const queued = hooks.sessions.get("agent:main:main")!.pending[0]!.request_id;
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2"));
    expect(client.published.map((p) => p.interactions[0]!.content)).toEqual(["a", "c"]);
    expect(client.published[0]!.request_id).toBe(queued);
    expect(hooks.sessions.get("agent:main:main")!.pending).toEqual([]);
  });

  it("treats 'already exists' on retry as delivered, so a slow commit does not wedge the queue", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [
      new Error("The operation was aborted due to timeout"), // committed server-side anyway
      new ReflexioError("api/publish_interaction -> request_id 'openclaw-x' already exists"),
    ];
    // The real client maps "already exists" to success; emulate that mapping.
    const real = client.publish.bind(client);
    client.publish = async (req: PublishRequest) => {
      try {
        await real(req);
      } catch (err) {
        if (err instanceof ReflexioError && /already exists/.test(err.message)) return;
        throw err;
      }
    };
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2"));
    expect(hooks.sessions.get("agent:main:main")!.pending).toEqual([]);
    expect(client.attempts).toBe(3);
  });

  it("drops a publish the server rejects instead of retrying it forever", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [new ReflexioError("422 bad request", 422)];
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    expect(hooks.sessions.get("agent:main:main")!.pending).toEqual([]);
    expect(hooks.stats.failures["publish rejected"]).toBe(1);
  });

  it("skips heartbeat runs, failed runs, short prompts and incognito sessions", async () => {
    const { client, hooks } = setup((key) => key.includes(":incognito-"));
    await hooks.beforePromptBuild({ prompt: "check inbox" }, ctx("hb", { trigger: "heartbeat" }));
    await hooks.agentEnd({ runId: "hb", success: true, messages: turn("x", "y") }, ctx("hb", { trigger: "heartbeat" }));
    await hooks.beforePromptBuild({ prompt: "hi" }, ctx("short"));
    await hooks.agentEnd({ runId: "bad", success: false, messages: turn("x", "y") }, ctx("bad"));
    const incognito = ctx("inc", { sessionKey: "agent:main:dashboard:incognito-abc" });
    await hooks.beforePromptBuild({ prompt: "a private question" }, incognito);
    await hooks.agentEnd({ runId: "inc", success: true, messages: turn("x", "y") }, incognito);
    expect(client.searches).toHaveLength(0);
    expect(client.published).toHaveLength(0);
  });

  it("fails open when search throws", async () => {
    const { client, hooks } = setup();
    client.search = async () => {
      throw new Error("ECONNREFUSED");
    };
    await expect(hooks.beforePromptBuild({ prompt: "anything here" }, ctx("r"))).resolves.toBeUndefined();
    expect(hooks.stats.failures.search).toBe(1);
    expect(await hooks.command("status", { sessionKey: "agent:main:main" })).toContain("search=1");
  });

  it("finalizes the session that ended, even when the next one already started", async () => {
    const { client, hooks } = setup();
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    // /new: the next session's first turn is injected before the detached
    // session_end for sess-1 runs.
    await hooks.beforePromptBuild({ prompt: "new topic please" }, ctx("run-2", { sessionId: "sess-2" }));
    await hooks.sessionEnd({ sessionId: "sess-1", reason: "new" }, { sessionKey: "agent:main:main", sessionId: "sess-1" });
    expect(client.finalized).toEqual([[client.published[0]!.user_id, "sess-1"]]);
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("new topic please", "ok") }, ctx("run-2", { sessionId: "sess-2" }));
    const second = client.published[1]!;
    expect(second.session_id).toBe("sess-2");
    expect(second.interactions[1]!.retrieved_learnings).toHaveLength(2);
  });

  it("does not finalize on compaction, and finalizes a deferred session once its turns land", async () => {
    const { client, hooks } = setup();
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    await hooks.sessionEnd({ sessionId: "sess-1", reason: "compaction" }, { sessionKey: "agent:main:main" });
    client.publishOutcomes = [new Error("down"), new Error("down")];
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2"));
    await hooks.sessionEnd({ sessionId: "sess-1", reason: "idle" }, { sessionKey: "agent:main:main" });
    expect(client.finalized).toEqual([]);
    // Next session's first turn delivers the queued turn, then the owed finalize.
    await hooks.agentEnd({ runId: "run-3", success: true, messages: turn("e", "f") }, ctx("run-3", { sessionId: "sess-2" }));
    expect(client.published.map((p) => p.interactions[0]!.content)).toEqual(["a", "c", "e"]);
    expect(client.finalized.map(([, session]) => session)).toEqual(["sess-1"]);
  });

  it("keeps turns in order: a new turn waits behind an undelivered one", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [new Error("down"), new Error("still down")];
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("first", "b") }, ctx("run-1"));
    // The retry of "first" fails again; "second" must not overtake it.
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("second", "d") }, ctx("run-2"));
    expect(client.published).toEqual([]);
    await hooks.agentEnd({ runId: "run-3", success: true, messages: turn("third", "f") }, ctx("run-3"));
    expect(client.published.map((p) => p.interactions[0]!.content)).toEqual(["first", "second", "third"]);
  });

  it("serializes concurrent drains so no queued turn is dropped", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [new Error("down"), new Error("down"), new Error("down")];
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("one", "b") }, ctx("run-1"));
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("two", "b") }, ctx("run-2"));
    expect(hooks.sessions.get("agent:main:main")!.pending).toHaveLength(2);
    // Server recovers but is slow: each publish takes a tick.
    const slow = client.publish.bind(client);
    client.publish = async (req: PublishRequest) => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      return slow(req);
    };
    await Promise.all([
      hooks.agentEnd({ runId: "run-3", success: true, messages: turn("three", "b") }, ctx("run-3")),
      hooks.command("learn", { sessionKey: "agent:main:main" }),
      hooks.sessionEnd({ sessionId: "sess-1", reason: "idle" }, { sessionKey: "agent:main:main" }),
    ]);
    expect(client.published.map((p) => p.interactions[0]!.content)).toEqual(["one", "two", "three"]);
  });

  it("finalizes sessions compaction rotated away from when the conversation ends", async () => {
    const { client, hooks } = setup();
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    await hooks.sessionEnd({ sessionId: "sess-1", reason: "compaction" }, { sessionKey: "agent:main:main" });
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2", { sessionId: "sess-2" }));
    expect(client.finalized).toEqual([]);
    await hooks.sessionEnd({ sessionId: "sess-2", reason: "new" }, { sessionKey: "agent:main:main" });
    expect(client.finalized.map(([, session]) => session).sort()).toEqual(["sess-1", "sess-2"]);
  });

  describe("with a fake clock", () => {
    afterEach(() => vi.useRealTimers());

    it("gives finalize only the budget the queued publishes left", async () => {
      vi.useFakeTimers();
      const { client, hooks } = setup();
      client.publishOutcomes = [new Error("down"), new Error("down")];
      await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
      await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2"));
      const slow = client.publish.bind(client);
      client.publish = async (req: PublishRequest) => {
        vi.setSystemTime(Date.now() + 11_500); // each publish uses most of its timeout
        return slow(req);
      };
      const timeouts: (number | undefined)[] = [];
      client.finalizeSession = async (u: string, sess: string, timeoutMs?: number) => {
        timeouts.push(timeoutMs);
        client.finalized.push([u, sess]);
      };
      await hooks.sessionEnd({ sessionId: "sess-1", reason: "idle" }, { sessionKey: "agent:main:main" });
      // 25 s budget - 2 x 11.5 s of publishing leaves 2 s for finalize.
      expect(timeouts).toHaveLength(1);
      expect(timeouts[0]).toBeLessThanOrEqual(2_000);
    });
  });

  it("finalizes only after the turn an overlapping agent_end is publishing", async () => {
    const { client, hooks } = setup();
    client.publishOutcomes = [new Error("down")];
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("old", "b") }, ctx("run-1"));
    const events: string[] = [];
    const slow = client.publish.bind(client);
    client.publish = async (req: PublishRequest) => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      await slow(req);
      events.push(`publish ${req.interactions[0]!.content}`);
    };
    client.finalizeSession = async (u: string, sess: string) => {
      events.push(`finalize ${sess}`);
      client.finalized.push([u, sess]);
    };
    await Promise.all([
      hooks.agentEnd({ runId: "run-2", success: true, messages: turn("current", "d") }, ctx("run-2")),
      hooks.sessionEnd({ sessionId: "sess-1", reason: "new" }, { sessionKey: "agent:main:main" }),
    ]);
    expect(events).toEqual(["publish old", "publish current", "finalize sess-1"]);
  });

  it("retries a finalize that failed", async () => {
    const { client, hooks } = setup();
    await hooks.agentEnd({ runId: "run-1", success: true, messages: turn("a", "b") }, ctx("run-1"));
    const real = client.finalizeSession.bind(client);
    let fail = true;
    client.finalizeSession = async (u: string, s: string) => {
      if (fail) throw new Error("down");
      return real(u, s);
    };
    await hooks.sessionEnd({ sessionId: "sess-1", reason: "new" }, { sessionKey: "agent:main:main" });
    expect(client.finalized).toEqual([]);
    fail = false;
    await hooks.agentEnd({ runId: "run-2", success: true, messages: turn("c", "d") }, ctx("run-2", { sessionId: "sess-2" }));
    expect(client.finalized.map(([, session]) => session)).toEqual(["sess-1"]);
  });
});
