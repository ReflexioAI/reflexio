// The Reflexio client: the agent-facing calls of the Python ReflexioClient,
// with the same names (camelCased), defaults and environment variables.
// Request and response shapes are the server's own generated schemas.

import { ReflexioError } from "./errors.js";
import type {
  FinalizeSessionRequest,
  FinalizeSessionResponse,
  GetAgentPlaybooksRequest,
  GetInteractionsRequest,
  GetInteractionsResponse,
  GetAgentPlaybooksResponse,
  GetProfilesRequest,
  GetProfilesResponse,
  GetUserPlaybooksRequest,
  GetUserPlaybooksResponse,
  GradeOnDemandRequest,
  GradeOnDemandResponse,
  LearningStatusResponse,
  PublishInteractionParams,
  PublishInteractionResponse,
  RequestOptions,
  SearchRequest,
  SearchResponse,
  WhoamiResponse,
} from "./types.js";

export const DEFAULT_URL = "https://www.reflexio.ai/";
// Matches the Python client's DEFAULT_AGENT_VERSION, so the same agent
// publishing from either language groups its playbooks together.
export const DEFAULT_AGENT_VERSION = "agent-v0";
const DEFAULT_TIMEOUT_MS = 300_000;

export interface ReflexioClientOptions {
  /** API key. Defaults to the `REFLEXIO_API_KEY` environment variable. */
  apiKey?: string;
  /** API base URL. Defaults to `REFLEXIO_URL`, then the hosted service. */
  url?: string;
  /** Default per-request timeout in milliseconds (default 300000, as in Python). */
  timeoutMs?: number;
  /** A `fetch` implementation, for runtimes or tests that supply their own. */
  fetch?: typeof fetch;
}

// Read without depending on Node's types, so the client also loads where
// `process` does not exist (browsers, some edge runtimes).
function env(name: string): string {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[name]?.trim() ?? "";
}

function requireSessionId(sessionId: unknown): string {
  if (typeof sessionId !== "string" || !sessionId.trim()) {
    throw new TypeError("session_id is required and cannot be empty");
  }
  return sessionId;
}

export class ReflexioClient {
  readonly url: string;
  readonly timeoutMs: number;
  private readonly apiKey: string;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ReflexioClientOptions = {}) {
    const url = options.url?.trim() || env("REFLEXIO_URL") || DEFAULT_URL;
    // A trailing slash keeps a base path (https://host/reflexio/) when the
    // endpoint path is resolved against it.
    this.url = url.endsWith("/") ? url : `${url}/`;
    this.apiKey = options.apiKey?.trim() || env("REFLEXIO_API_KEY");
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    const fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof fetchImpl !== "function") {
      throw new TypeError("No fetch implementation available; pass one as options.fetch");
    }
    this.fetchImpl = fetchImpl;
  }

  /**
   * Bind a session ID once. The recommended way to use the client: the search
   * that surfaced a learning and the publish that recorded the turn then
   * always agree on the session, which is what makes retrieval measurable.
   */
  forSession(sessionId: string): SessionScopedClient {
    return new SessionScopedClient(this, requireSessionId(sessionId));
  }

  /**
   * Publish the turns of a conversation. Returns once the server has durably
   * accepted them; extraction runs afterwards unless `wait_for_response` is
   * set. Poll `getLearningStatus(response.request_id)` to follow it.
   */
  async publishInteraction(
    params: PublishInteractionParams,
    options?: RequestOptions,
  ): Promise<PublishInteractionResponse> {
    const { interactions, wait_for_response, ...rest } = params;
    requireSessionId(rest.session_id);
    const body = {
      agent_version: DEFAULT_AGENT_VERSION,
      ...rest,
      interaction_data_list: interactions,
    };
    const query = wait_for_response ? { wait_for_response: "true" } : undefined;
    return this.request("POST", "api/publish_interaction", { body, query, ...options });
  }

  /** Search profiles, user playbooks and agent playbooks for the current turn. */
  async search(params: SearchRequest, options?: RequestOptions): Promise<SearchResponse> {
    return this.request("POST", "api/search", { body: params, ...options });
  }

  /** Tell Reflexio a session is complete, so it can learn without waiting for more turns. */
  async finalizeSession(params: FinalizeSessionRequest, options?: RequestOptions): Promise<FinalizeSessionResponse> {
    requireSessionId(params.session_id);
    return this.request("POST", "api/finalize_session", { body: params, ...options });
  }

  /** Learning status of a published request: pending, processing, done or not_tracked. */
  async getLearningStatus(requestId: string, options?: RequestOptions): Promise<LearningStatusResponse> {
    return this.request("GET", "api/learning_status", { query: { request_id: requestId }, ...options });
  }

  /** The organization and project the API key belongs to. */
  async whoami(options?: RequestOptions): Promise<WhoamiResponse> {
    return this.request("GET", "api/whoami", options);
  }

  /** Grade one published session now instead of waiting for scheduled evaluation. */
  async gradeOnDemand(params: GradeOnDemandRequest, options?: RequestOptions): Promise<GradeOnDemandResponse> {
    requireSessionId(params.session_id);
    return this.request("POST", "api/evaluations/grade_on_demand", { body: params, ...options });
  }

  /** Stored interactions for a user, newest first. `start_time`/`end_time` are ISO 8601 strings. */
  async getInteractions(params: GetInteractionsRequest, options?: RequestOptions): Promise<GetInteractionsResponse> {
    return this.request("POST", "api/get_interactions", { body: params, ...options });
  }

  async getProfiles(params: GetProfilesRequest, options?: RequestOptions): Promise<GetProfilesResponse> {
    return this.request("POST", "api/get_profiles", { body: params, ...options });
  }

  async getUserPlaybooks(params: GetUserPlaybooksRequest, options?: RequestOptions): Promise<GetUserPlaybooksResponse> {
    return this.request("POST", "api/get_user_playbooks", { body: params, ...options });
  }

  async getAgentPlaybooks(params: GetAgentPlaybooksRequest, options?: RequestOptions): Promise<GetAgentPlaybooksResponse> {
    return this.request("POST", "api/get_agent_playbooks", { body: params, ...options });
  }

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    init: RequestOptions & { body?: unknown; query?: Record<string, string> } = {},
  ): Promise<T> {
    const url = new URL(path, this.url);
    for (const [key, value] of Object.entries(init.query ?? {})) url.searchParams.set(key, value);
    const headers: Record<string, string> = { Accept: "application/json" };
    if (init.body !== undefined) headers["Content-Type"] = "application/json";
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const timeoutMs = init.timeoutMs ?? this.timeoutMs;
    const timeout = AbortSignal.timeout(timeoutMs);
    const signal = init.signal ? AbortSignal.any([init.signal, timeout]) : timeout;
    const target = `${method} ${url.origin}${url.pathname}`;
    // The connection can drop, time out or be aborted while the body is still
    // arriving, so reading it is as much a transport step as the fetch itself.
    const transportError = (err: unknown) =>
      timeout.aborted
        ? new ReflexioError(`${target} timed out after ${timeoutMs} ms`, { cause: err })
        : new ReflexioError(`${target} failed: ${err instanceof Error ? err.message : String(err)}`, { cause: err });

    let res: Response;
    try {
      res = await this.fetchImpl(url, {
        method,
        headers,
        body: init.body === undefined ? undefined : JSON.stringify(init.body),
        signal,
        // A redirect is a misconfigured base URL (a marketing host answering
        // with its www twin). Following it would turn a POST into a GET.
        redirect: "manual",
      });
    } catch (err) {
      throw transportError(err);
    }

    if (res.type === "opaqueredirect" || (res.status >= 300 && res.status < 400)) {
      throw new ReflexioError(
        `${target} was redirected${res.headers.get("location") ? ` to ${res.headers.get("location")}` : ""}. ` +
          "Is REFLEXIO_URL pointing at the API host?",
        { status: res.status || undefined },
      );
    }
    let text: string;
    try {
      text = await res.text();
    } catch (err) {
      throw transportError(err);
    }
    const isJson = (res.headers.get("content-type") ?? "").toLowerCase().includes("json");
    let body: unknown = text;
    if (isJson && text) {
      try {
        body = JSON.parse(text);
      } catch (err) {
        throw new ReflexioError(`${target} returned invalid JSON (HTTP ${res.status})`, {
          status: res.status,
          body: text,
          cause: err,
        });
      }
    }
    if (!res.ok) {
      const detail = typeof body === "object" && body !== null && "detail" in body ? (body as { detail: unknown }).detail : body;
      const message = typeof detail === "string" ? detail : JSON.stringify(detail);
      throw new ReflexioError(`${target} -> HTTP ${res.status}: ${message.slice(0, 500)}`, { status: res.status, body });
    }
    if (!text) return {} as T;
    if (!isJson) {
      throw new ReflexioError(
        `${target} returned ${res.headers.get("content-type") || "no content type"}, not JSON (HTTP ${res.status}). ` +
          "Is REFLEXIO_URL pointing at the API host?",
        { status: res.status, body: text.slice(0, 200) },
      );
    }
    return body as T;
  }
}

/** A view of a client with one session ID bound. Created by `forSession`. */
export class SessionScopedClient {
  readonly client: ReflexioClient;
  readonly sessionId: string;

  constructor(client: ReflexioClient, sessionId: string) {
    this.client = client;
    this.sessionId = sessionId;
  }

  async publishInteraction(
    params: Omit<PublishInteractionParams, "session_id">,
    options?: RequestOptions,
  ): Promise<PublishInteractionResponse> {
    return this.client.publishInteraction({ ...params, session_id: this.bound(params) }, options);
  }

  async search(params: Omit<SearchRequest, "session_id">, options?: RequestOptions): Promise<SearchResponse> {
    return this.client.search({ ...params, session_id: this.bound(params) }, options);
  }

  async finalizeSession(
    params: Omit<FinalizeSessionRequest, "session_id">,
    options?: RequestOptions,
  ): Promise<FinalizeSessionResponse> {
    return this.client.finalizeSession({ ...params, session_id: this.bound(params) }, options);
  }

  // The types already omit session_id; this catches plain-JavaScript callers
  // passing a different one, which would split the session silently.
  private bound(params: object): string {
    const given = (params as { session_id?: unknown }).session_id;
    if (given !== undefined && given !== this.sessionId) {
      throw new TypeError(`session_id ${String(given)} conflicts with the bound session ${this.sessionId}`);
    }
    return this.sessionId;
  }
}
