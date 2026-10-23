// Minimal Reflexio HTTP client: only the endpoints this plugin calls.
// Wire shapes mirror reflexio/models/api_schema (UnifiedSearchRequest,
// PublishUserInteractionRequest, FinalizeSessionRequest); the contract test
// in tests/contract.test.ts pins the field names against the server's
// OpenAPI document.

import type { Settings } from "./config.js";
import { characterPrefix } from "./text.js";

export const AGENT_VERSION = "openclaw";
export const SOURCE = "openclaw";
// `null` is Reflexio's CURRENT status in a status_filter list.
const CURRENT = [null];

export interface Profile {
  profile_id?: string;
  content?: string;
}

export interface Playbook {
  user_playbook_id?: number | string;
  agent_playbook_id?: number | string;
  content?: string;
  trigger?: string | null;
  rationale?: string | null;
  playbook_status?: string | null;
  exposure_receipt?: string | null;
}

export interface SearchHits {
  profiles: Profile[];
  user_playbooks: Playbook[];
  agent_playbooks: Playbook[];
}

export interface ToolUsed {
  tool_name: string;
  status: "success" | "error";
  tool_data?: Record<string, unknown>;
}

export interface Citation {
  kind: string;
  real_id: string;
  tag: string;
  title: string;
}

export interface RetrievedLearning {
  kind: "profile" | "user_playbook" | "agent_playbook";
  learning_id: string;
  exposure_receipt?: string;
}

export interface Interaction {
  role: "User" | "Assistant";
  content: string;
  tools_used?: ToolUsed[];
  citations?: Citation[];
  retrieved_learnings?: RetrievedLearning[];
}

export interface PublishRequest {
  request_id: string;
  user_id: string;
  session_id: string;
  interactions: Interaction[];
}

export class ReflexioError extends Error {
  readonly status?: number;

  constructor(
    message: string,
    status?: number,
  ) {
    super(message);
    this.status = status;
  }

  // A 4xx other than timeout/throttling will fail the same way on every
  // retry, so the caller should drop the request instead of queueing it.
  get rejected(): boolean {
    return this.status !== undefined && this.status >= 400 && this.status < 500 && ![408, 429].includes(this.status);
  }
}

// The server refuses a second publish under a request_id it has already
// admitted. A publish that timed out on our side may still have committed
// (the server's own deadline is minutes, ours seconds), so the retry seeing
// this message means the first attempt landed.
const ALREADY_ADMITTED = /request_id .* already exists/;

export class ReflexioClient {
  private readonly settings: Settings;

  constructor(settings: Settings) {
    this.settings = settings;
  }

  get baseUrl(): string {
    return this.settings.url;
  }

  async search(args: {
    query: string;
    userId: string;
    sessionId: string;
    requestId: string;
  }): Promise<SearchHits> {
    const body = await this.post(
      "api/search",
      {
        query: args.query,
        user_id: args.userId,
        agent_version: AGENT_VERSION,
        entity_types: ["profiles", "user_playbooks", "agent_playbooks"],
        agent_playbook_status_filter: ["pending", "approved"],
        enable_agent_answer: false,
        enable_reformulation: false,
        top_k: this.settings.topK,
        search_mode: "hybrid",
        session_id: args.sessionId,
        request_id: args.requestId,
      },
      this.settings.searchTimeoutMs,
    );
    return {
      profiles: list<Profile>(body.profiles),
      user_playbooks: list<Playbook>(body.user_playbooks),
      agent_playbooks: list<Playbook>(body.agent_playbooks).filter(
        (pb) => String(pb.playbook_status ?? "").toLowerCase() !== "rejected",
      ),
    };
  }

  async publish(req: PublishRequest, timeoutMs: number): Promise<void> {
    try {
      await this.post(
        "api/publish_interaction",
        {
          request_id: req.request_id,
          user_id: req.user_id,
          session_id: req.session_id,
          agent_version: AGENT_VERSION,
          source: SOURCE,
          interaction_data_list: req.interactions,
        },
        timeoutMs,
      );
    } catch (err) {
      if (err instanceof ReflexioError && ALREADY_ADMITTED.test(err.message)) return;
      throw err;
    }
  }

  async finalizeSession(
    userId: string,
    sessionId: string,
    timeoutMs: number,
  ): Promise<void> {
    await this.post("api/finalize_session", { user_id: userId, session_id: sessionId }, timeoutMs);
  }

  async listLearnings(userId: string, limit: number): Promise<SearchHits> {
    const timeout = this.settings.searchTimeoutMs;
    const [profiles, userPlaybooks, agentPlaybooks] = await Promise.all([
      this.post(
        "api/get_profiles",
        { user_id: userId, top_k: limit, status_filter: CURRENT },
        timeout,
      ),
      this.post(
        "api/get_user_playbooks",
        { user_id: userId, limit, status_filter: CURRENT },
        timeout,
      ),
      this.post(
        "api/get_agent_playbooks",
        { agent_version: AGENT_VERSION, limit, status_filter: CURRENT },
        timeout,
      ),
    ]);
    return {
      profiles: list<Profile>(profiles.user_profiles),
      user_playbooks: list<Playbook>(userPlaybooks.user_playbooks),
      agent_playbooks: list<Playbook>(agentPlaybooks.agent_playbooks),
    };
  }

  async health(): Promise<boolean> {
    try {
      const res = await this.request("GET", "health", undefined, 2000);
      return res.ok;
    } catch {
      return false;
    }
  }

  private async post(
    path: string,
    payload: unknown,
    timeoutMs: number,
  ): Promise<Record<string, unknown>> {
    const res = await this.request("POST", path, payload, timeoutMs);
    const text = await res.text();
    if (!res.ok) {
      throw new ReflexioError(
        `${path} -> HTTP ${res.status}: ${text.slice(0, characterPrefix(text, 200).end)}`,
        res.status,
      );
    }
    const body = JSON.parse(text) as Record<string, unknown>;
    // Reflexio reports some failures as 200 {"success": false, ...}.
    if (body.success === false) {
      throw new ReflexioError(`${path} -> ${String(body.message ?? "failed")}`);
    }
    return body;
  }

  private request(
    method: string,
    path: string,
    payload: unknown,
    timeoutMs: number,
  ): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json" };
    if (payload !== undefined) headers["Content-Type"] = "application/json";
    if (this.settings.apiKey) headers.Authorization = `Bearer ${this.settings.apiKey}`;
    return fetch(new URL(path, this.settings.url), {
      method,
      headers,
      body: payload === undefined ? undefined : JSON.stringify(payload),
      signal: AbortSignal.timeout(timeoutMs),
    });
  }
}

function list<T>(value: unknown): T[] {
  return Array.isArray(value) ? (value as T[]) : [];
}
