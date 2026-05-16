"""Shared row-retention policy for storage backends."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

DEFAULT_ROW_RETENTION_LIMIT = 250_000
ROW_RETENTION_DELETE_FRACTION = 0.20


@dataclass(frozen=True, slots=True)
class RetentionTarget:
    """A physical storage target eligible for row-count retention."""

    name: str
    table_name: str
    order_column: str
    id_columns: tuple[str, ...]


RETENTION_TARGETS: tuple[RetentionTarget, ...] = (
    RetentionTarget("profiles", "profiles", "created_at", ("profile_id",)),
    RetentionTarget("interactions", "interactions", "created_at", ("interaction_id",)),
    RetentionTarget("requests", "requests", "created_at", ("request_id",)),
    RetentionTarget(
        "user_playbooks", "user_playbooks", "created_at", ("user_playbook_id",)
    ),
    RetentionTarget(
        "agent_playbooks", "agent_playbooks", "created_at", ("agent_playbook_id",)
    ),
    RetentionTarget(
        "agent_success_evaluation_result",
        "agent_success_evaluation_result",
        "created_at",
        ("result_id",),
    ),
    RetentionTarget("profile_change_logs", "profile_change_logs", "created_at", ("id",)),
    RetentionTarget(
        "playbook_aggregation_change_logs",
        "playbook_aggregation_change_logs",
        "created_at",
        ("id",),
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
# rely on it (Postgres, Supabase, and the SQLite/Disk bespoke implementations).
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
}


def get_row_retention_limits() -> dict[str, int]:
    """Return per-target row limits from env with code defaults.

    ``REFLEXIO_ROW_LIMIT_<TARGET>`` takes precedence for every target.
    ``INTERACTION_CLEANUP_THRESHOLD`` remains the legacy override for
    interactions when the new variable is not present.
    """
    limits: dict[str, int] = {}
    for target in RETENTION_TARGETS:
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
