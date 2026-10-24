import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_URL, type Interaction, ReflexioClient, ReflexioError } from "../src/index.js";

type Sent = { url: URL; init: RequestInit; body: unknown };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

function recorder(respond: (sent: Sent) => Response | Promise<Response> = () => json({ success: true })) {
  const sent: Sent[] = [];
  const fetch = vi.fn(async (input: URL | RequestInfo, init: RequestInit = {}) => {
    const call = { url: new URL(String(input)), init, body: init.body ? JSON.parse(String(init.body)) : undefined };
    sent.push(call);
    return respond(call);
  });
  return { sent, fetch: fetch as unknown as typeof globalThis.fetch };
}

const turns: Interaction[] = [
  { role: "User", content: "I see a charge I do not recognize" },
  { role: "Agent", content: "Let me pull up your recent charges." },
];

afterEach(() => vi.unstubAllEnvs());

describe("configuration", () => {
  it("reads REFLEXIO_API_KEY and REFLEXIO_URL like the Python client", async () => {
    vi.stubEnv("REFLEXIO_API_KEY", "env-key");
    vi.stubEnv("REFLEXIO_URL", "https://self-hosted.example/reflexio");
    const { sent, fetch } = recorder();
    const client = new ReflexioClient({ fetch });
    await client.search({ query: "q" });
    expect(client.url).toBe("https://self-hosted.example/reflexio/");
    expect(sent[0]?.url.href).toBe("https://self-hosted.example/reflexio/api/search");
    expect((sent[0]?.init.headers as Record<string, string>).Authorization).toBe("Bearer env-key");
  });

  it("defaults to the hosted service, and explicit options win over the environment", async () => {
    vi.stubEnv("REFLEXIO_URL", "");
    expect(new ReflexioClient({ fetch: recorder().fetch }).url).toBe(DEFAULT_URL);
    vi.stubEnv("REFLEXIO_API_KEY", "env-key");
    const { sent, fetch } = recorder();
    await new ReflexioClient({ apiKey: "explicit", url: "http://localhost:8081", fetch }).search({ query: "q" });
    expect(sent[0]?.url.href).toBe("http://localhost:8081/api/search");
    expect((sent[0]?.init.headers as Record<string, string>).Authorization).toBe("Bearer explicit");
  });

  it("sends no Authorization header without a key", async () => {
    vi.stubEnv("REFLEXIO_API_KEY", "");
    const { sent, fetch } = recorder();
    await new ReflexioClient({ fetch }).search({ query: "q" });
    expect((sent[0]?.init.headers as Record<string, string>).Authorization).toBeUndefined();
  });
});

describe("publishInteraction", () => {
  it("sends the turns as interaction_data_list with the Python client's default agent_version", async () => {
    const { sent, fetch } = recorder(() => json({ success: true, request_id: "r1", learning_status: "deferred" }));
    const client = new ReflexioClient({ apiKey: "k", url: "http://h/", fetch });
    const res = await client.publishInteraction({ user_id: "u_123", session_id: "s_1", interactions: turns });
    expect(res.request_id).toBe("r1");
    expect(sent[0]?.init.method).toBe("POST");
    expect(sent[0]?.url.pathname).toBe("/api/publish_interaction");
    expect(sent[0]?.url.search).toBe("");
    expect(sent[0]?.body).toEqual({
      agent_version: "agent-v0",
      user_id: "u_123",
      session_id: "s_1",
      interaction_data_list: turns,
    });
  });

  it("passes wait_for_response as the query parameter the server reads", async () => {
    const { sent, fetch } = recorder();
    const client = new ReflexioClient({ url: "http://h/", fetch });
    await client.publishInteraction({
      user_id: "u",
      session_id: "s",
      interactions: turns,
      agent_version: "support-bot",
      wait_for_response: true,
    });
    expect(sent[0]?.url.searchParams.get("wait_for_response")).toBe("true");
    expect(sent[0]?.body).toMatchObject({ agent_version: "support-bot" });
    expect(sent[0]?.body).not.toHaveProperty("wait_for_response");
  });

  it("refuses an empty session_id before sending anything", async () => {
    const { sent, fetch } = recorder();
    const client = new ReflexioClient({ url: "http://h/", fetch });
    await expect(client.publishInteraction({ user_id: "u", session_id: " ", interactions: turns })).rejects.toThrow(
      "session_id is required",
    );
    expect(sent).toHaveLength(0);
  });
});

describe("other calls", () => {
  it("map to their REST endpoints", async () => {
    const { sent, fetch } = recorder(({ url }) =>
      url.pathname === "/api/learning_status" ? json({ status: "done" }) : json({ success: true }),
    );
    const client = new ReflexioClient({ url: "http://h/", fetch });
    await client.search({ query: "refund request", user_id: "u", session_id: "s", top_k: 3 });
    await client.finalizeSession({ user_id: "u", session_id: "s" });
    expect((await client.getLearningStatus("r1")).status).toBe("done");
    await client.whoami();
    await client.getInteractions({ user_id: "u", top_k: 5 });
    await client.gradeOnDemand({ session_id: "s", agent_version: "agent-v0" });
    await client.getProfiles({ user_id: "u" });
    await client.getUserPlaybooks({ user_id: "u" });
    await client.getAgentPlaybooks({ agent_version: "agent-v0" });
    expect(sent.map((s) => `${s.init.method} ${s.url.pathname}${s.url.search}`)).toEqual([
      "POST /api/search",
      "POST /api/finalize_session",
      "GET /api/learning_status?request_id=r1",
      "GET /api/whoami",
      "POST /api/get_interactions",
      "POST /api/evaluations/grade_on_demand",
      "POST /api/get_profiles",
      "POST /api/get_user_playbooks",
      "POST /api/get_agent_playbooks",
    ]);
    expect(sent[0]?.body).toEqual({ query: "refund request", user_id: "u", session_id: "s", top_k: 3 });
    expect(sent[2]?.init.body).toBeUndefined();
  });
});

describe("forSession", () => {
  it("binds one session ID to search, publish and finalize", async () => {
    const { sent, fetch } = recorder();
    const session = new ReflexioClient({ url: "http://h/", fetch }).forSession("s_1");
    await session.search({ query: "q", user_id: "u" });
    await session.publishInteraction({ user_id: "u", interactions: turns });
    await session.finalizeSession({ user_id: "u" });
    expect(sent.map((s) => (s.body as { session_id?: string }).session_id)).toEqual(["s_1", "s_1", "s_1"]);
  });

  it("rejects an empty session and a conflicting one", async () => {
    const client = new ReflexioClient({ url: "http://h/", fetch: recorder().fetch });
    expect(() => client.forSession("")).toThrow(TypeError);
    const session = client.forSession("s_1");
    // A plain-JavaScript caller can still pass one; it must not split the session.
    await expect(session.search({ query: "q", session_id: "other" } as never)).rejects.toThrow("conflicts");
  });
});

describe("failures", () => {
  const call = (respond: () => Response | Promise<Response>) =>
    new ReflexioClient({ url: "http://h/", fetch: recorder(respond).fetch }).search({ query: "q" });

  it("raise ReflexioError with the status and the server's detail", async () => {
    const err = await call(() => json({ detail: "Invalid API key" }, 401)).catch((e) => e);
    expect(err).toBeInstanceOf(ReflexioError);
    expect(err.status).toBe(401);
    expect(err.message).toContain("Invalid API key");
    expect(err.body).toEqual({ detail: "Invalid API key" });
  });

  it("treat a redirect as a misconfigured URL instead of following it", async () => {
    const err = await call(() => new Response(null, { status: 302, headers: { location: "https://reflexio.ai/" } })).catch(
      (e) => e,
    );
    expect(err).toBeInstanceOf(ReflexioError);
    expect(err.message).toContain("Is REFLEXIO_URL pointing at the API host?");
    const { sent, fetch } = recorder();
    await new ReflexioClient({ url: "http://h/", fetch }).whoami();
    expect(sent[0]?.init.redirect).toBe("manual");
  });

  it("reject a 200 that is not JSON, as an HTML page from the wrong host would be", async () => {
    const err = await call(() => new Response("<html>hi</html>", { headers: { "content-type": "text/html" } })).catch(
      (e) => e,
    );
    expect(err).toBeInstanceOf(ReflexioError);
    expect(err.message).toContain("not JSON");
  });

  it("report a timeout as one", async () => {
    const hang = (_: URL | RequestInfo, init: RequestInit = {}) =>
      new Promise<Response>((_, reject) =>
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true }),
      );
    const client = new ReflexioClient({ url: "http://h/", fetch: hang as typeof fetch });
    await expect(client.search({ query: "q" }, { timeoutMs: 20 })).rejects.toThrow("timed out after 20 ms");
  });

  it("honour a caller's abort signal", async () => {
    const hang = (_: URL | RequestInfo, init: RequestInit = {}) =>
      new Promise<Response>((_, reject) =>
        init.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true }),
      );
    const client = new ReflexioClient({ url: "http://h/", fetch: hang as typeof fetch });
    const controller = new AbortController();
    const pending = client.search({ query: "q" }, { signal: controller.signal });
    controller.abort(new Error("user cancelled"));
    await expect(pending).rejects.toThrow("user cancelled");
  });

  it("return an empty object for an empty 2xx", async () => {
    await expect(call(() => new Response(null, { status: 204 }))).resolves.toEqual({});
  });
});
