"""Shared pure helpers for retrieved-learning evaluation state.

Used by every storage backend (SQLite, Supabase, native Postgres) and by the
group-evaluation runner, so this module must stay free of storage imports.

Concurrency model (no mutation-site instrumentation):

- The **session fingerprint** is a SHA-256 digest over every interaction ID in
  the session plus each interaction's canonical attachment refs. Any
  interaction publish or delete changes it (autoincrement interaction IDs make
  ABA collisions unrealizable), so it covers transcript-only changes as well
  as attachment changes.
- Each evaluation run receives a monotonically increasing **generation**
  allocated in ``_operation_state``. Replacement CASes generation and a
  fingerprint recomputed under the replacement transaction's lock, so an
  older or stale snapshot cannot overwrite newer session state.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

RETRIEVED_LEARNING_STATE_PREFIX = "retrieved_learning_eval"

# Statuses persisted in _operation_state. "complete" and "not_applicable" are
# terminal (the fast path may short-circuit on them); "degraded" and "failed"
# are retried by the next scheduled or forced run.
type RetrievedLearningPersistedStatus = Literal[
    "pending", "in_progress", "complete", "degraded", "failed", "not_applicable"
]
TERMINAL_RETRIEVED_STATUSES: frozenset[str] = frozenset({"complete", "not_applicable"})

# Outcomes of one runner invocation. "stale"/"superseded"/"skipped" are
# invocation outcomes only and are never stored as persisted status.
type RetrievedLearningInvocationStatus = Literal[
    "pending",
    "complete",
    "degraded",
    "failed",
    "not_applicable",
    "stale",
    "superseded",
    "skipped",
]

# Canonical kinds accepted and persisted by retrieved-learning evaluation.
# ``RetrievedLearning.kind`` validates these at the API boundary; storage
# parsers still filter defensively against this set.
CANONICAL_RETRIEVED_KINDS: frozenset[str] = frozenset(
    {"profile", "user_playbook", "agent_playbook"}
)


def build_retrieved_learning_state_key(user_id: str, session_id: str) -> str:
    """Build the ``_operation_state`` key for one session's evaluation state.

    Length-prefixed (following the grade_on_demand cache-key precedent) so a
    crafted user/session pair cannot collide with another pair. Org scoping is
    unnecessary: ``_operation_state`` is already org-scoped on every backend.

    Args:
        user_id (str): Session owner.
        session_id (str): Evaluated session.

    Returns:
        str: The state key.
    """
    return (
        f"{RETRIEVED_LEARNING_STATE_PREFIX}"
        f"::{len(user_id)}:{user_id}::{len(session_id)}:{session_id}"
    )


@dataclass
class SnapshotInteraction:
    """One interaction row in the bounded session projection.

    ``refs`` holds ``(kind, learning_id)`` attachment tuples; interaction
    content is carried only for transcript construction and is never
    persisted in operation state.
    """

    interaction_id: int
    role: str
    content: str
    created_at: int
    refs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class BoundedRetrievedLearningSnapshot:
    """Bounded projection of one session for retrieved-learning evaluation.

    Loaded without constructing full ``Interaction`` objects (no embeddings,
    no image encodings). ``attachment_limit_exceeded`` is set when the raw
    attachment-occurrence cap was hit; the evaluator must then make zero LLM
    calls.
    """

    interactions: list[SnapshotInteraction] = field(default_factory=list)
    earliest_request_created_at: int | None = None
    agent_version: str = ""
    raw_attachment_count: int = 0
    attachment_limit_exceeded: bool = False


def session_fingerprint(snapshot: BoundedRetrievedLearningSnapshot) -> str:
    """Compute the canonical session fingerprint for a bounded snapshot.

    Covers every interaction ID plus each interaction's ``(kind,
    learning_id)`` refs, so any publish or delete in the session invalidates
    it. Only this digest is ever persisted.

    Args:
        snapshot (BoundedRetrievedLearningSnapshot): The session projection.

    Returns:
        str: Hex SHA-256 digest.
    """
    entries = sorted(
        (interaction.interaction_id, sorted(interaction.refs))
        for interaction in snapshot.interactions
    )
    canonical = json.dumps(entries, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class RetrievedLearningCommitResult:
    """Outcome of one atomic result-set replacement.

    ``status``/``committed_count`` are authoritative only when ``disposition``
    is ``"applied"`` (commit-time eligibility can turn a proposed status into
    ``"not_applicable"``).
    """

    disposition: Literal["applied", "stale", "superseded"]
    status: RetrievedLearningPersistedStatus | None = None
    committed_count: int = 0
