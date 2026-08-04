"""SQLite governance rebuild-hide store methods.

Extracted verbatim from ``_governance.py`` (the RebuildHide bucket): the two
public methods (``hide_governance_agent_playbooks_for_rebuild``,
``apply_governance_agent_playbook_rebuild``) plus the three RebuildHide-owned
privates (``_replace_agent_playbook_source_windows_locked``,
``_delete_agent_playbook_search_rows_locked``,
``_upsert_agent_playbook_search_rows_locked``) and the module-level
``_build_agent_playbook_source_window_rows`` helper.

``apply_governance_agent_playbook_rebuild`` preserves its ordering byte-for-byte:
verify the planned rebuild target + hide_for_rebuild complete, then either
UPDATE the playbook and refresh FTS/vec (windows remain) OR hard-delete the
playbook and emit the ``hard_delete`` lineage event (no windows remain), then
record the ``rebuild_without_erased_sources`` target complete, all committing
together. ``_emit_hard_delete_playbook`` is function-imported LOCAL-in-method
(from ``.._playbook``) to avoid an import cycle.

The residual ``SQLiteGovernanceMixin`` stays composed alongside this mixin and
permanently holds the shared infra (``conn``, ``_lock``, ``org_id``,
``_deps()``), the delete-target / hide-for-rebuild / prepared-snapshot
validators, and the module-level helpers (``_json_dumps``, ``_json_loads``),
which are imported here rather than duplicated. ``_record_purge_target_locked``
(PurgeOperationStore) resolves through the composed MRO.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from reflexio.models.api_schema.domain import AgentPlaybookSourceWindow
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.embedding_text import playbook_trigger_embedding_text
from reflexio.server.services.storage.governance_claims import PurgeExecutionClaim
from reflexio.server.services.storage.governance_validation import (
    _canonicalize_governance_windows,
    _parse_governance_window_list,
    _validate_governance_purge_id,
)

from .._governance import _json_dumps, _json_loads

if TYPE_CHECKING:
    from .._governance import _SQLiteGovernanceDeps


def _build_agent_playbook_source_window_rows(
    agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
) -> list[tuple[int, int, str]]:
    by_id: dict[int, list[int]] = {}
    for window in windows:
        ids = by_id.setdefault(window.user_playbook_id, [])
        seen = set(ids)
        for source_id in window.source_interaction_ids:
            if source_id not in seen:
                ids.append(source_id)
                seen.add(source_id)
    return [
        (
            agent_playbook_id,
            user_playbook_id,
            _json_dumps(source_interaction_ids) or "[]",
        )
        for user_playbook_id, source_interaction_ids in by_id.items()
    ]


class RebuildHideMixin:
    """SQLite governance rebuild-hide store primitives."""

    # Type hints for instance attributes/methods provided via MRO by the
    # co-composed residual SQLiteGovernanceMixin / SQLiteStorageBase.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str
    _deps: Callable[[], _SQLiteGovernanceDeps]

    # Provided via MRO by the co-composed PurgeOperationStoreMixin (purge bucket);
    # reached here by the cross-bucket rebuild-hide method.
    _record_purge_target_locked: Callable[..., None]
    _assert_purge_operation_execution_claim_locked: Callable[
        [str, PurgeExecutionClaim | None], None
    ]

    def _replace_agent_playbook_source_windows_locked(
        self, agent_playbook_id: int, windows: list[AgentPlaybookSourceWindow]
    ) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )
        source_window_rows = _build_agent_playbook_source_window_rows(
            agent_playbook_id, windows
        )
        if source_window_rows:
            self.conn.executemany(
                """INSERT OR IGNORE INTO agent_playbook_source_user_playbooks
                   (agent_playbook_id, user_playbook_id, source_interaction_ids)
                   VALUES (?, ?, ?)""",
                source_window_rows,
            )

    def _delete_agent_playbook_search_rows_locked(self, agent_playbook_id: int) -> None:
        self.conn.execute(
            "DELETE FROM agent_playbooks_fts WHERE rowid = ?",
            (agent_playbook_id,),
        )
        if self._deps()._has_sqlite_vec:
            self.conn.execute(
                "DELETE FROM agent_playbooks_vec WHERE rowid = ?",
                (agent_playbook_id,),
            )

    def _upsert_agent_playbook_search_rows_locked(
        self,
        *,
        agent_playbook_id: int,
        trigger: str | None,
        content: str,
        expanded_terms: str | None,
        embedding: list[float],
    ) -> None:
        self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
        fts_parts = [trigger or "", content]
        if expanded_terms:
            fts_parts.append(expanded_terms)
        self.conn.execute(
            "INSERT INTO agent_playbooks_fts(rowid, search_text) VALUES (?, ?)",
            (
                agent_playbook_id,
                " ".join(part for part in fts_parts if part) or "",
            ),
        )
        if self._deps()._has_sqlite_vec and embedding:
            self.conn.execute(
                "INSERT INTO agent_playbooks_vec(rowid, embedding) VALUES (?, ?)",
                (agent_playbook_id, json.dumps(embedding)),
            )

    def hide_governance_agent_playbooks_for_rebuild(
        self,
        purge_id: str,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> list[int]:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                target_rows = self.conn.execute(
                    """SELECT target_ref
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ?
                         AND target_name = 'agent_playbook'
                         AND phase = 'rebuild_without_erased_sources'
                         AND target_ref != ''
                         AND status != 'complete'
                       ORDER BY CAST(target_ref AS INTEGER) ASC""",
                    (self.org_id, purge_id),
                ).fetchall()
                agent_playbook_ids = [int(row["target_ref"]) for row in target_rows]
                if not agent_playbook_ids:
                    self.conn.commit()
                    return []
                placeholders = ",".join("?" for _ in agent_playbook_ids)
                self.conn.execute(
                    f"""UPDATE agent_playbooks
                        SET status = ?
                        WHERE agent_playbook_id IN ({placeholders})""",
                    [Status.ARCHIVE_IN_PROGRESS.value, *agent_playbook_ids],
                )
                for agent_playbook_id in agent_playbook_ids:
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="hide_for_rebuild",
                        status="complete",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                    self._record_purge_target_locked(
                        purge_id=purge_id,
                        target_name="agent_playbook",
                        target_ref=str(agent_playbook_id),
                        phase="rebuild_without_erased_sources",
                        status="running",
                        detail=None,
                        deleted_count=0,
                        error_detail=None,
                    )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
        return agent_playbook_ids

    def apply_governance_agent_playbook_rebuild(
        self,
        purge_id: str,
        agent_playbook_id: int,
        remaining_source_windows: list[dict[str, object]],
        content: str | None,
        trigger: str | None,
        rationale: str | None,
        blocking_issue: dict[str, object] | None,
        expanded_terms: str | None,
        tags: list[str] | None,
        *,
        execution_claim: PurgeExecutionClaim,
    ) -> None:
        purge_id = _validate_governance_purge_id("purge_id", purge_id)
        windows = _parse_governance_window_list(
            "remaining_source_windows", remaining_source_windows
        )
        canonical_remaining_windows = [window.model_dump() for window in windows]
        content_value = content or ""
        trigger_value = trigger or None
        with self._lock:
            try:
                self.conn.execute("BEGIN IMMEDIATE")
                self._assert_purge_operation_execution_claim_locked(
                    purge_id, execution_claim
                )
                embedding_text = playbook_trigger_embedding_text(trigger_value)
                embedding = (
                    self._deps()._get_embedding(embedding_text)
                    if embedding_text
                    else []
                )
                rebuild_target_row = self.conn.execute(
                    """SELECT status, detail
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'rebuild_without_erased_sources'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if rebuild_target_row is None:
                    raise ValueError("planned rebuild target does not exist")
                if rebuild_target_row["status"] == "complete":
                    raise ValueError("planned rebuild target is already complete")
                rebuild_detail = _json_loads(rebuild_target_row["detail"])
                if not isinstance(rebuild_detail, dict) or not {
                    "original_source_windows",
                    "previous_lifecycle_status",
                    "remaining_source_windows",
                }.issubset(rebuild_detail):
                    raise ValueError(
                        "planned rebuild target is missing source window detail"
                    )
                planned_remaining_windows = _canonicalize_governance_windows(
                    "planned remaining_source_windows",
                    cast(
                        list[dict[str, object]],
                        rebuild_detail["remaining_source_windows"],
                    ),
                )
                if planned_remaining_windows != canonical_remaining_windows:
                    raise ValueError(
                        "remaining_source_windows must match the planned rebuild target"
                    )
                previous_lifecycle_status = cast(
                    str | None, rebuild_detail["previous_lifecycle_status"]
                )
                hide_target_row = self.conn.execute(
                    """SELECT status
                       FROM purge_operation_targets
                       WHERE org_id = ? AND purge_id = ? AND target_name = 'agent_playbook'
                         AND target_ref = ? AND phase = 'hide_for_rebuild'""",
                    (self.org_id, purge_id, str(agent_playbook_id)),
                ).fetchone()
                if hide_target_row is None or hide_target_row["status"] != "complete":
                    raise ValueError("hide_for_rebuild target must be complete")
                if windows:
                    cur = self.conn.execute(
                        """UPDATE agent_playbooks
                           SET content = ?, trigger = ?, rationale = ?, blocking_issue = ?,
                               embedding = ?, expanded_terms = ?, tags = ?, status = ?
                           WHERE agent_playbook_id = ?""",
                        (
                            content_value,
                            trigger_value,
                            rationale,
                            json.dumps(blocking_issue)
                            if blocking_issue is not None
                            else None,
                            _json_dumps(embedding),
                            expanded_terms,
                            _json_dumps(tags),
                            previous_lifecycle_status,
                            agent_playbook_id,
                        ),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    self._replace_agent_playbook_source_windows_locked(
                        agent_playbook_id, windows
                    )
                    self._upsert_agent_playbook_search_rows_locked(
                        agent_playbook_id=agent_playbook_id,
                        trigger=trigger_value,
                        content=content_value,
                        expanded_terms=expanded_terms,
                        embedding=embedding,
                    )
                else:
                    from .._playbook import _emit_hard_delete_playbook

                    self._delete_agent_playbook_search_rows_locked(agent_playbook_id)
                    self.conn.execute(
                        "DELETE FROM agent_playbook_source_user_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    cur = self.conn.execute(
                        "DELETE FROM agent_playbooks WHERE agent_playbook_id = ?",
                        (agent_playbook_id,),
                    )
                    if cur.rowcount == 0:
                        raise ValueError(
                            f"Agent playbook with ID {agent_playbook_id} not found"
                        )
                    _emit_hard_delete_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(agent_playbook_id),
                        request_id=purge_id,
                    )
                self._record_purge_target_locked(
                    purge_id=purge_id,
                    target_name="agent_playbook",
                    target_ref=str(agent_playbook_id),
                    phase="rebuild_without_erased_sources",
                    status="complete",
                    detail=None,
                    deleted_count=0,
                    error_detail=None,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
