// Hook and command behaviour, independent of the OpenClaw registration API so
// tests can drive it with plain objects.
//
// Every handler fails open: a Reflexio outage must never block or alter the
// user's turn. Failures are counted and logged at warn (throttled) so they are
// visible in the gateway log and in `/reflexio status`, rather than vanishing
// at debug the way the previous subprocess shim's did.
//
// Scope: learnings are per agent workspace, not per human. Everyone talking
// to one agent in a shared channel shares its Reflexio user.

import { type Interaction, type PublishRequest, type ReflexioClient, ReflexioError } from "./client.js";
import type { Settings } from "./config.js";
import { requestIdFor, resolveUserId } from "./identity.js";
import { renderInjection, resolveCitations, retrievedLearnings } from "./render.js";
import { Sessions, type SessionState } from "./sessions.js";
import { currentTurn, runStartIn } from "./transcript.js";

export interface Logger {
  info?: (msg: string) => void;
  warn?: (msg: string) => void;
  debug?: (msg: string) => void;
}

export interface HookContext {
  runId?: string;
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
  trigger?: string;
}

export interface Stats {
  injections: number;
  publishes: number;
  failures: Record<string, number>;
  lastError?: string;
}

export type IncognitoCheck = (sessionKey: string) => boolean;

const WARN_INTERVAL_MS = 60_000;
// agent_end's host budget is 30 s; leave headroom for the host's own work.
const AGENT_END_BUDGET_MS = 25_000;
const MIN_ATTEMPT_MS = 1_000;

export class ReflexioHooks {
  readonly sessions = new Sessions();
  readonly stats: Stats = { injections: 0, publishes: 0, failures: {} };
  private readonly lastWarn = new Map<string, number>();

  constructor(
    private readonly settings: Settings,
    private readonly client: ReflexioClient,
    private readonly log: Logger,
    private readonly isIncognito: IncognitoCheck = () => false,
  ) {}

  async beforePromptBuild(
    event: { prompt?: string; currentUserMessage?: string; messages?: unknown[] },
    ctx: HookContext,
  ): Promise<{ prependContext: string } | undefined> {
    const state = this.stateFor(ctx);
    if (!state || !ctx.sessionKey || !ctx.runId) return undefined;
    const run = this.sessions.run(state, ctx.runId);
    // `prompt` carries inbound channel/sender context ahead of the user's
    // words; search on what the user actually wrote when the host gives it.
    const query = (event.currentUserMessage ?? event.prompt ?? "").trim();
    if (Array.isArray(event.messages)) run.start = runStartIn(event.messages, query);
    if (query.length < this.settings.minPromptLength) return undefined;
    try {
      const hits = await this.client.search({
        query,
        userId: state.userId,
        sessionId: state.reflexioSessionId,
        requestId: requestIdFor(ctx.sessionKey, ctx.runId),
      });
      const { markdown, registry } = renderInjection(hits);
      if (!markdown) return undefined;
      this.sessions.recordInjection(state, ctx.runId, registry);
      this.stats.injections++;
      return { prependContext: markdown };
    } catch (err) {
      this.fail("search", err);
      return undefined;
    }
  }

  async agentEnd(
    event: { runId?: string; messages?: unknown[]; success?: boolean },
    ctx: HookContext,
  ): Promise<void> {
    const deadline = Date.now() + AGENT_END_BUDGET_MS;
    const runId = event.runId ?? ctx.runId;
    const state = this.stateFor(ctx);
    if (!state || !ctx.sessionKey || !runId) return;
    const run = state.runs.get(runId);
    state.runs.delete(runId);
    // A failed run's transcript ends mid-turn; there is nothing to learn from.
    const turn = event.success === false ? undefined : currentTurn(event.messages ?? [], run?.start);
    if (!turn) {
      await this.sessions.exclusive(state, () => this.flush(state, deadline));
      return;
    }

    const assistant: Interaction = { role: "Assistant", content: turn.assistant };
    if (turn.tools.length) assistant.tools_used = turn.tools;
    const citations = resolveCitations(turn.fullAssistant, [...state.tags.values()]);
    if (citations.length) assistant.citations = citations;
    const retrieved = retrievedLearnings(run?.injected ?? []);
    if (retrieved.length) assistant.retrieved_learnings = retrieved;

    const req: PublishRequest = {
      request_id: requestIdFor(ctx.sessionKey, runId),
      user_id: state.userId,
      session_id: state.reflexioSessionId,
      interactions: [{ role: "User", content: turn.user }, assistant],
    };
    await this.sessions.exclusive(state, async () => {
      await this.flush(state, deadline);
      // Behind an undelivered turn, wait in line: the server orders a session
      // by admission time, so sending now would reorder the conversation.
      if (state.pending.length || (await this.send(req, deadline)) === "retry") {
        this.sessions.enqueue(state, req);
      }
    });
  }

  async sessionEnd(
    event: { sessionId?: string; reason?: string },
    ctx: HookContext,
  ): Promise<void> {
    if (!ctx.sessionKey) return;
    const ended = event.sessionId ?? ctx.sessionId;
    const state = this.sessions.get(ctx.sessionKey);
    if (!state || !ended) return;
    // Compaction rotates the sessionId mid-conversation; it is not an end.
    // Remember the old id so the real end finalizes it too.
    if (event.reason === "compaction") {
      state.compacted.add(ended);
      return;
    }
    // Owed until it succeeds. Finalizing before queued turns land would leave
    // them outside the judged prefix, so flush() finalizes once they have.
    // Recorded inside the queue, after work admitted before this session end:
    // an in-flight agent_end must publish its turn before the finalize runs.
    await this.sessions.exclusive(state, () => {
      this.oweFinalize(state, ended);
      return this.flush(state, Date.now() + AGENT_END_BUDGET_MS);
    });
    this.sessions.forget(ctx.sessionKey, ended);
  }

  async command(args: string, ctx: { sessionKey?: string }): Promise<string> {
    const [sub = "status"] = args.trim().split(/\s+/).filter(Boolean);
    const state = ctx.sessionKey ? this.sessions.get(ctx.sessionKey) : undefined;
    switch (sub) {
      case "status":
        return this.statusText(state);
      case "show":
        if (!state) return "Reflexio: no project yet — send a message in this session first.";
        return this.showText(state.userId);
      case "learn":
        if (!state) return "Reflexio: nothing to learn from yet in this session.";
        return this.sessions.exclusive(state, async () => {
          this.oweFinalize(state, state.reflexioSessionId);
          await this.flush(state, Date.now() + AGENT_END_BUDGET_MS);
          if (state.pending.length) {
            return `Reflexio: ${state.pending.length} turn(s) are still waiting to be published; try again shortly.`;
          }
          return state.pendingFinalize.size
            ? `Reflexio: finalize failed — ${this.stats.lastError ?? "out of time"}; it will be retried.`
            : "Reflexio: asked the server to extract learnings from this session.";
        });
      default:
        return "Usage: /reflexio [status|show|learn]";
    }
  }

  private stateFor(ctx: HookContext): SessionState | undefined {
    // Heartbeats and cron runs are not user conversations.
    if (ctx.trigger && ctx.trigger !== "user") return undefined;
    if (!ctx.sessionKey) return undefined;
    // Incognito sessions must not leave the machine: OpenClaw already blanks
    // their transcript for observers, so do not search with their prompts.
    if (this.isIncognito(ctx.sessionKey)) return undefined;
    const userId =
      resolveUserId({ agentId: ctx.agentId, workspaceDir: ctx.workspaceDir }) ??
      this.sessions.get(ctx.sessionKey)?.userId;
    if (!userId) return undefined;
    return this.sessions.touch(ctx.sessionKey, userId, ctx.sessionId ?? ctx.sessionKey);
  }

  // Callers hold the session's queue (Sessions.exclusive).
  private async flush(state: SessionState, deadline: number): Promise<void> {
    // Drain in order while the hook's budget lasts; send() refuses to start
    // an attempt it cannot finish, so a backlog cannot overrun the budget.
    while (state.pending.length) {
      const req = state.pending[0] as PublishRequest;
      if ((await this.send(req, deadline)) === "retry") return; // still down; keep order
      state.pending.shift();
    }
    if (state.pending.length) return;
    for (const sessionId of [...state.pendingFinalize]) {
      // Like send(): never start a request the hook's budget cannot cover.
      const remaining = Math.min(this.settings.publishTimeoutMs, deadline - Date.now());
      if (remaining < MIN_ATTEMPT_MS) return;
      if (!(await this.finalize(state.userId, sessionId, remaining))) return;
      state.pendingFinalize.delete(sessionId);
    }
  }

  // The session plus any ids compaction rotated away from on the way to it.
  private oweFinalize(state: SessionState, sessionId: string): void {
    for (const id of state.compacted) state.pendingFinalize.add(id);
    state.compacted.clear();
    state.pendingFinalize.add(sessionId);
  }

  private async send(req: PublishRequest, deadline: number): Promise<"sent" | "dropped" | "retry"> {
    const remaining = Math.min(this.settings.publishTimeoutMs, deadline - Date.now());
    if (remaining < MIN_ATTEMPT_MS) return "retry";
    try {
      await this.client.publish(req, remaining);
      this.stats.publishes++;
      return "sent";
    } catch (err) {
      if (err instanceof ReflexioError && err.rejected) {
        // Same request, same rejection: retrying would only block the queue.
        this.fail("publish rejected", err);
        return "dropped";
      }
      this.fail("publish", err);
      return "retry";
    }
  }

  private async finalize(userId: string, sessionId: string, timeoutMs: number): Promise<boolean> {
    try {
      await this.client.finalizeSession(userId, sessionId, timeoutMs);
      return true;
    } catch (err) {
      this.fail("finalize", err);
      return false;
    }
  }

  private fail(kind: string, err: unknown): void {
    this.stats.failures[kind] = (this.stats.failures[kind] ?? 0) + 1;
    this.stats.lastError = `${kind}: ${errorText(err)}`;
    const now = Date.now();
    if (now - (this.lastWarn.get(kind) ?? 0) < WARN_INTERVAL_MS) return;
    this.lastWarn.set(kind, now);
    this.log.warn?.(`[reflexio] ${kind} failed (${this.client.baseUrl}): ${errorText(err)}`);
  }

  private async statusText(state?: SessionState): Promise<string> {
    const healthy = await this.client.health();
    const failures = Object.entries(this.stats.failures)
      .map(([k, n]) => `${k}=${n}`)
      .join(", ");
    return [
      `Reflexio ${healthy ? "reachable" : "UNREACHABLE"} at ${this.client.baseUrl}`,
      `project: ${state?.userId ?? "(not resolved yet)"}`,
      `injections: ${this.stats.injections}, publishes: ${this.stats.publishes}` +
        (state?.pending.length ? `, queued: ${state.pending.length}` : ""),
      failures ? `failures: ${failures} — last: ${this.stats.lastError}` : "failures: none",
    ].join("\n");
  }

  private async showText(userId: string): Promise<string> {
    try {
      const hits = await this.client.listLearnings(userId, 20);
      const lines = [`Reflexio learnings for \`${userId}\``];
      const section = (title: string, items: { content?: string }[]) => {
        const rows = items.map((i) => i.content?.trim()).filter(Boolean);
        if (rows.length) lines.push(`\n**${title}**`, ...rows.map((r) => `- ${r}`));
      };
      section("Shared skills", hits.agent_playbooks);
      section("Project skills", hits.user_playbooks);
      section("Preferences", hits.profiles);
      return lines.length > 1 ? lines.join("\n") : `Reflexio: nothing learned yet for \`${userId}\`.`;
    } catch (err) {
      this.fail("show", err);
      return `Reflexio: could not read learnings — ${errorText(err)}`;
    }
  }
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}
