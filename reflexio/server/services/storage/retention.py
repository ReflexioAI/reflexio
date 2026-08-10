"""Shared row-retention policy for storage backends."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_ROW_RETENTION_LIMIT = 250_000
ROW_RETENTION_DELETE_FRACTION = 0.20
OPEN_WORLD_EVIDENCE_RETENTION_WINDOW_SECONDS = 14 * 24 * 60 * 60
TOMBSTONE_STATUSES = ("archived", "merged", "superseded", "expired")


@dataclass(frozen=True, slots=True)
class RetentionTarget:
    """A physical storage target eligible for row-count retention."""

    name: str
    table_name: str
    order_column: str
    id_columns: tuple[str, ...]
    priority_statuses: tuple[str, ...] = ()
    minimum_age_seconds: int = 0
    fixed_row_limit: int | None = None


@dataclass(frozen=True, slots=True)
class OptimizationRetentionClass:
    """Fixed owner for one optimization artifact lifetime."""

    artifact_class: str
    owner: str


OPTIMIZATION_RETENTION_CLASSES: tuple[OptimizationRetentionClass, ...] = (
    OptimizationRetentionClass("event", "governance_audit"),
    OptimizationRetentionClass("source_reference", "existing_source_retention"),
    OptimizationRetentionClass("staging", "lease_stale_claim"),
    OptimizationRetentionClass("terminal", "lineage_tombstone_grace"),
)


RETENTION_TARGETS: tuple[RetentionTarget, ...] = (
    RetentionTarget(
        "profiles",
        "profiles",
        "created_at",
        ("profile_id",),
        priority_statuses=TOMBSTONE_STATUSES,
    ),
    RetentionTarget("interactions", "interactions", "created_at", ("interaction_id",)),
    RetentionTarget("requests", "requests", "created_at", ("request_id",)),
    RetentionTarget(
        "user_playbooks",
        "user_playbooks",
        "created_at",
        ("user_playbook_id",),
        priority_statuses=TOMBSTONE_STATUSES,
    ),
    RetentionTarget(
        "agent_playbooks",
        "agent_playbooks",
        "created_at",
        ("agent_playbook_id",),
        priority_statuses=TOMBSTONE_STATUSES,
    ),
    RetentionTarget(
        "agent_success_evaluation_result",
        "agent_success_evaluation_result",
        "created_at",
        ("result_id",),
    ),
    # Grouped session target: keyed on (user_id, session_id) — not result_id —
    # so retention always removes whole session snapshots, never a partial
    # per-learning subset. Rows within a session share created_at (earliest
    # request timestamp), so ordering keeps groups adjacent. The session's
    # retrieved-eval _operation_state row is intentionally left in place: it
    # is content-free (digest + counters) and self-heals on the next
    # publish/forced evaluation.
    RetentionTarget(
        "retrieved_learning_evaluation",
        "retrieved_learning_evaluation",
        "created_at",
        ("user_id", "session_id"),
    ),
    RetentionTarget(
        "offline_tuner_reward_label",
        "offline_tuner_reward_label",
        "label_created_at",
        ("reward_label_id",),
    ),
    RetentionTarget("share_links", "share_links", "created_at", ("id",)),
    RetentionTarget(
        "agent_playbook_source_user_playbooks",
        "agent_playbook_source_user_playbooks",
        "created_at",
        ("agent_playbook_id", "user_playbook_id"),
    ),
    RetentionTarget(
        "playbook_optimization_jobs",
        "playbook_optimization_jobs",
        "created_at",
        ("job_id",),
    ),
    RetentionTarget(
        "playbook_optimization_candidates",
        "playbook_optimization_candidates",
        "created_at",
        ("candidate_id",),
    ),
    RetentionTarget(
        "playbook_optimization_evaluations",
        "playbook_optimization_evaluations",
        "created_at",
        ("evaluation_id",),
    ),
    RetentionTarget(
        "playbook_optimization_events",
        "playbook_optimization_events",
        "created_at",
        ("event_id",),
    ),
    # RETIRED WRITER, LIVE PII. Nothing writes playbook_retrieval_logs any more
    # (the retrieval-capture subsystem is gone), but the table still exists and
    # still holds user_id/session_id until a later release DROPs it — so it must
    # keep being trimmed. Every backend gates on ``_retention_table_exists``
    # before counting/deleting, so this target no-ops cleanly once the table is
    # dropped and an old task never issues a raw DELETE against a missing table.
    RetentionTarget(
        "playbook_retrieval_logs",
        "playbook_retrieval_logs",
        "created_at",
        ("retrieval_log_id",),
    ),
    RetentionTarget(
        "user_playbook_exposure_events",
        "user_playbook_exposure_events",
        "ingested_at",
        ("exposure_event_id",),
        minimum_age_seconds=OPEN_WORLD_EVIDENCE_RETENTION_WINDOW_SECONDS,
        fixed_row_limit=DEFAULT_ROW_RETENTION_LIMIT,
    ),
    RetentionTarget("skills", "skills", "created_at", ("skill_id",)),
)

RETENTION_TARGETS_BY_NAME = {target.name: target for target in RETENTION_TARGETS}


@dataclass(frozen=True, slots=True)
class CascadeRef:
    """A dependent table that must be cleaned when a retention target's rows
    are deleted.

    Attributes:
        table_name (str): Table whose rows depend on the retention target.
        fk_column (str): Column in ``table_name`` holding the parent target's
            primary key. Always references the first column of the parent
            target's ``id_columns`` (single-key parents only).
    """

    table_name: str
    fk_column: str


# Maps a retention target → its dependent tables that must be deleted first
# when retention removes rows. Keep in sync with the storage backends that
# rely on it (Postgres, Supabase, and the SQLite bespoke implementations).
RETENTION_CASCADES: dict[str, tuple[CascadeRef, ...]] = {
    "requests": (CascadeRef("interactions", "request_id"),),
    "user_playbooks": (
        CascadeRef("agent_playbook_source_user_playbooks", "user_playbook_id"),
    ),
    "agent_playbooks": (
        CascadeRef("agent_playbook_source_user_playbooks", "agent_playbook_id"),
    ),
    "playbook_optimization_jobs": (
        CascadeRef("playbook_optimization_evaluations", "job_id"),
        CascadeRef("playbook_optimization_events", "job_id"),
        CascadeRef("playbook_optimization_candidates", "job_id"),
    ),
    "playbook_optimization_candidates": (
        CascadeRef("playbook_optimization_evaluations", "candidate_id"),
    ),
    # Retired writer, live PII — see the RETENTION_TARGETS note above. The
    # cascade only runs when the parent target yielded keys, which requires the
    # parent table to exist, so a dropped pair no-ops without a raw DELETE.
    "playbook_retrieval_logs": (
        CascadeRef("playbook_retrieval_log_items", "retrieval_log_id"),
    ),
    "offline_tuner_reward_label": (
        CascadeRef("offline_tuner_reward_label_target", "reward_label_id"),
    ),
}


def get_row_retention_limits() -> dict[str, int]:
    """Return per-target row limits from env with code defaults.

    ``REFLEXIO_ROW_LIMIT_<TARGET>`` takes precedence for targets without a
    ``fixed_row_limit``. Fixed targets explicitly reject that override path.
    ``INTERACTION_CLEANUP_THRESHOLD`` remains the legacy override for
    interactions when the new variable is not present.
    """
    limits: dict[str, int] = {}
    for target in RETENTION_TARGETS:
        if target.fixed_row_limit is not None:
            limits[target.name] = target.fixed_row_limit
            continue
        env_name = f"REFLEXIO_ROW_LIMIT_{target.name.upper()}"
        default = DEFAULT_ROW_RETENTION_LIMIT
        if target.name == "interactions":
            default = _get_int_env("INTERACTION_CLEANUP_THRESHOLD", default)
        limits[target.name] = _get_int_env(env_name, default)
    return limits


def delete_count_for_retention(current_count: int) -> int:
    """Return how many rows to delete when a target exceeds its limit."""
    if current_count <= 0:
        return 0
    return max(1, math.ceil(current_count * ROW_RETENTION_DELETE_FRACTION))


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default
