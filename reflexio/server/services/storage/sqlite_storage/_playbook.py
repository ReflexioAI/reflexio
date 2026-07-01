"""Playbook CRUD + search methods for SQLite storage."""

import logging
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)

from ._lineage import _append_event_stmt


def _emit_hard_delete_playbook(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_type: str,
    entity_id: str,
    request_id: str,
    actor: str = "api",
) -> None:
    """Emit a single hard_delete lineage event for a playbook entity."""
    _append_event_stmt(
        conn,
        org_id=org_id,
        entity_type=entity_type,
        entity_id=entity_id,
        op="hard_delete",
        prov="wasInvalidatedBy",
        source_ids=[],
        actor=actor,
        request_id=request_id,
        reason="erasure",
    )


def _build_tags_sql(alias: str, tags: list[str] | None) -> tuple[str, list[Any]]:
    if not tags:
        return "", []
    placeholders = ",".join("?" for _ in tags)
    return (
        f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE value IN ({placeholders}))",
        list(tags),
    )
