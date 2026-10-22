// In-memory per-session state. The plugin runs inside the gateway process,
// so nothing here needs a file: OpenClaw owns the transcript and replays it
// to agent_end. What we keep is what the transcript does not carry — where
// each run started, which learnings were injected for it, and publishes
// waiting for a retry. The retry queue is lost on a gateway restart; that is
// the accepted cost of not keeping a second copy of the conversation on disk.

import type { PublishRequest } from "./client.js";
import type { RegistryEntry } from "./render.js";

const MAX_SESSIONS = 500;
const MAX_RUNS = 16;
const MAX_TAGS = 200;
const MAX_PENDING = 20;

export interface RunState {
  // Length of the history before this run's prompt (see transcript.ts).
  start?: number;
  injected: RegistryEntry[];
}

export interface SessionState {
  userId: string;
  // OpenClaw's current sessionId under this key. It changes on /new, reset,
  // idle rollover and compaction; runs in flight keep their own state.
  reflexioSessionId: string;
  runs: Map<string, RunState>;
  // Session-wide tag registry so a citation of an earlier injection resolves.
  tags: Map<string, RegistryEntry>;
  pending: PublishRequest[];
  // Ended sessions whose finalize is still owed: it waits for their queued
  // publishes to land, and survives a failed attempt.
  pendingFinalize: Set<string>;
  // Session ids compaction rotated away from. The conversation goes on, so
  // they are finalized together with the session that finally ends it.
  compacted: Set<string>;
  // Tail of the work serialized on this session's queue (see exclusive()).
  queueTail: Promise<void>;
}

function trim<K, V>(map: Map<K, V>, max: number): void {
  while (map.size > max) {
    const oldest = map.keys().next().value as K;
    map.delete(oldest);
  }
}

export class Sessions {
  private readonly byKey = new Map<string, SessionState>();

  get(sessionKey: string): SessionState | undefined {
    return this.byKey.get(sessionKey);
  }

  touch(sessionKey: string, userId: string, reflexioSessionId: string): SessionState {
    const state = this.byKey.get(sessionKey) ?? {
      userId,
      reflexioSessionId,
      runs: new Map(),
      tags: new Map(),
      pending: [],
      pendingFinalize: new Set(),
      compacted: new Set(),
      queueTail: Promise.resolve(),
    };
    // A rotation only moves the session id forward. Run and tag state is
    // keyed by runId / tag, so a session_end for the old id racing the new
    // session's first turn cannot wipe the new run's injections.
    state.userId = userId;
    state.reflexioSessionId = reflexioSessionId;
    // Re-insert so iteration order is least-recently-used first.
    this.byKey.delete(sessionKey);
    this.byKey.set(sessionKey, state);
    trim(this.byKey, MAX_SESSIONS);
    return state;
  }

  run(state: SessionState, runId: string): RunState {
    let run = state.runs.get(runId);
    if (!run) {
      run = { injected: [] };
      state.runs.set(runId, run);
      trim(state.runs, MAX_RUNS);
    }
    return run;
  }

  recordInjection(state: SessionState, runId: string, registry: RegistryEntry[]): void {
    this.run(state, runId).injected = registry;
    for (const entry of registry) {
      state.tags.delete(entry.tag.toLowerCase());
      state.tags.set(entry.tag.toLowerCase(), entry);
    }
    trim(state.tags, MAX_TAGS);
  }

  // Run `work` after every earlier queue operation on this session finishes.
  // Hooks and /reflexio learn run concurrently; two drains reading the same
  // queue head would each shift() it, dropping a turn that was never sent.
  exclusive<T>(state: SessionState, work: () => Promise<T>): Promise<T> {
    const result = state.queueTail.then(work);
    state.queueTail = result.then(
      () => undefined,
      () => undefined,
    );
    return result;
  }

  enqueue(state: SessionState, req: PublishRequest): void {
    if (state.pending.some((p) => p.request_id === req.request_id)) return;
    state.pending.push(req);
    // Oldest publishes are dropped first; they are the least likely to land.
    state.pending.splice(0, Math.max(0, state.pending.length - MAX_PENDING));
  }

  // Forget a session only if it is still the one that ended; a newer session
  // under the same key keeps its state.
  forget(sessionKey: string, endedSessionId: string): void {
    const state = this.byKey.get(sessionKey);
    if (
      state &&
      state.reflexioSessionId === endedSessionId &&
      !state.pending.length &&
      !state.pendingFinalize.size
    ) {
      this.byKey.delete(sessionKey);
    }
  }
}
