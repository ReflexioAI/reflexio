"""Agent playbook CRUD + search methods for SQLite storage."""

import json
import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

from reflexio.models.api_schema.common import BlockingIssue
from reflexio.models.api_schema.domain.entities import LineageContext
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    Status,
)
from reflexio.models.config_schema import SearchMode, SearchOptions
from reflexio.server.services.embedding_text import (
    embedding_text,
    resolve_retrieval_threshold,
)
from reflexio.server.services.storage.lifecycle_filters import (
    validate_include_inactive,
)

from .._base import (
    _TOMBSTONE_STATUS_VALUES,
    SQLiteStorageBase,
    _build_status_sql,
    _effective_search_mode,
    _epoch_now,
    _epoch_to_iso,
    _json_dumps,
    _json_loads,
    _row_to_agent_playbook,
    _sanitize_fts_query,
    _true_rrf_merge,
    _uses_unicode_lexical_fallback,
    _vector_rank_rows,
)
from .._lineage import _append_event_stmt
from .._playbook import _build_tags_sql, _emit_hard_delete_playbook

_AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES = (
    *_TOMBSTONE_STATUS_VALUES,
    Status.ARCHIVE_IN_PROGRESS.value,
)


def _emit_supersede_playbook(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_id: str,
    old_status: str | None,
    request_id: str,
) -> None:
    """Emit a single status_change->superseded lineage event for an agent playbook."""
    _append_event_stmt(
        conn,
        org_id=org_id,
        entity_type="agent_playbook",
        entity_id=entity_id,
        op="status_change",
        prov="wasInvalidatedBy",
        source_ids=[],
        actor="aggregator",
        request_id=request_id,
        reason=f"{old_status or 'None'}->superseded",
        from_status=old_status,
        to_status=Status.SUPERSEDED.value,
        status_namespace="lifecycle_status",
    )


class AgentPlaybookStoreMixin:
    """Mixin providing agent playbook CRUD + search for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    embedding_model_name: str
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _should_expand_documents: Any
    _expand_document: Any
    _fts_upsert: Any
    _vec_upsert: Any
    _delete_playbook_search_rows: Any
    _own_transaction: Any
    commit_scope: Any

    def _index_agent_playbook_fts_vec(self, ap: AgentPlaybook) -> None:
        """Update the FTS and vector indexes for a single agent playbook row.

        Must be called AFTER the row's transaction has been committed.  The FTS
        and vec helpers self-commit, so they must never be interleaved inside a
        transaction that still has pending mutations.

        Args:
            ap (AgentPlaybook): The already-saved playbook (``agent_playbook_id``
                must be set).
        """
        fts_parts = [ap.trigger or "", ap.content or ""]
        if ap.expanded_terms:
            fts_parts.append(ap.expanded_terms)
        self._fts_upsert(
            "agent_playbooks_fts",
            ap.agent_playbook_id,
            search_text=" ".join(p for p in fts_parts if p) or "",
        )
        if ap.embedding:
            self._vec_upsert("agent_playbooks_vec", ap.agent_playbook_id, ap.embedding)

    def _insert_agent_playbook_row(
        self, conn: "sqlite3.Connection", ap: AgentPlaybook, created_at_iso: str
    ) -> "sqlite3.Cursor":
        """Execute the agent_playbooks INSERT and populate ``ap.agent_playbook_id``.

        Runs the INSERT inside the caller's connection context; does NOT commit.
        The caller is responsible for committing (or rolling back) the transaction.

        Args:
            conn: The open SQLite connection to execute against.
            ap: The playbook to insert; ``agent_playbook_id`` is set on return.
            created_at_iso: ISO-8601 timestamp string for ``created_at``.

        Returns:
            sqlite3.Cursor: The cursor from the INSERT (``lastrowid`` is the new PK).
        """
        cur = conn.execute(
            """INSERT INTO agent_playbooks
               (playbook_name, created_at, agent_version, content,
                trigger, rationale, blocking_issue,
                playbook_status, playbook_metadata, embedding,
                expanded_terms, tags, status,
                merged_into, superseded_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ap.playbook_name,
                created_at_iso,
                ap.agent_version,
                ap.content,
                ap.trigger,
                ap.rationale,
                json.dumps(ap.blocking_issue.model_dump())
                if ap.blocking_issue
                else None,
                ap.playbook_status.value
                if isinstance(ap.playbook_status, PlaybookStatus)
                else ap.playbook_status,
                ap.playbook_metadata,
                _json_dumps(ap.embedding),
                ap.expanded_terms,
                _json_dumps(ap.tags),
                ap.status.value if ap.status else None,
                ap.merged_into,
                ap.superseded_by,
            ),
        )
        ap.agent_playbook_id = cur.lastrowid or 0
        return cur

    @SQLiteStorageBase.handle_exceptions
    def save_agent_playbooks(
        self,
        agent_playbooks: list[AgentPlaybook],
        *,
        lineage_contexts: list[LineageContext] | None = None,
    ) -> list[AgentPlaybook]:
        if lineage_contexts is not None and len(lineage_contexts) != len(
            agent_playbooks
        ):
            raise ValueError("lineage_contexts must match agent_playbooks length")
        if any(
            context.op_kind not in {"create", "aggregate"}
            for context in lineage_contexts or []
        ):
            raise ValueError(
                "agent playbook lineage context must use op_kind='create' or 'aggregate'"
            )
        if any(
            context.op_kind == "aggregate"
            and not (context.request_id and context.request_id.strip())
            for context in lineage_contexts or []
        ):
            raise ValueError("agent playbook aggregate lineage requires request_id")

        contexts = lineage_contexts or [
            LineageContext(op_kind="create") for _ap in agent_playbooks
        ]
        rows: list[tuple[AgentPlaybook, LineageContext, str]] = []
        for ap, context in zip(agent_playbooks, contexts, strict=True):
            text = embedding_text(ap)
            if text and self._should_expand_documents():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    emb_future = executor.submit(self._get_embedding, text)
                    exp_future = executor.submit(self._expand_document, text)
                    ap.embedding = emb_future.result(timeout=15)
                    ap.expanded_terms = exp_future.result(timeout=15)
            elif text:
                ap.embedding = self._get_embedding(text)
                ap.expanded_terms = None
            else:
                ap.embedding = []
                ap.expanded_terms = None
            rows.append((ap, context, _epoch_to_iso(ap.created_at)))

        with self.commit_scope():
            for ap, context, created_at_iso in rows:
                with self._lock:
                    self._insert_agent_playbook_row(self.conn, ap, created_at_iso)
                    is_aggregate = context.op_kind == "aggregate"
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(ap.agent_playbook_id),
                        op=context.op_kind,
                        prov="wasDerivedFrom" if is_aggregate else "wasGeneratedBy",
                        source_ids=context.source_ids,
                        actor=context.actor,
                        request_id=context.request_id
                        or f"{context.op_kind}_{ap.agent_playbook_id}",
                        reason=context.reason,
                        model_name=context.model_name,
                        provider=context.provider,
                    )

        for ap, _context, _created_at_iso in rows:
            try:
                self._index_agent_playbook_fts_vec(ap)
            except Exception:
                logger.exception(
                    "FTS/vec indexing failed for agent_playbook %s "
                    "(row committed, index skipped)",
                    ap.agent_playbook_id,
                )
        return agent_playbooks

    @SQLiteStorageBase.handle_exceptions
    def get_agent_playbooks(
        self,
        limit: int = 100,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
        tags: list[str] | None = None,
        agent_playbook_id: int | None = None,
        query: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        offset: int = 0,
        max_agent_playbook_id: int | None = None,
    ) -> list[AgentPlaybook]:
        sql = "SELECT * FROM agent_playbooks WHERE 1=1"
        params: list[Any] = []

        if agent_playbook_id is not None:
            sql += " AND agent_playbook_id = ?"
            params.append(agent_playbook_id)
        if max_agent_playbook_id is not None:
            sql += " AND agent_playbook_id <= ?"
            params.append(max_agent_playbook_id)
        if query:
            like = f"%{query.lower()}%"
            sql += (
                " AND (LOWER(content) LIKE ? OR LOWER(trigger) LIKE ? "
                "OR LOWER(rationale) LIKE ? OR LOWER(playbook_name) LIKE ? "
                "OR LOWER(playbook_metadata) LIKE ?)"
            )
            params.extend([like, like, like, like, like])
        if playbook_name:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)

        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if start_time is not None:
            sql += " AND created_at >= ?"
            params.append(_epoch_to_iso(start_time))
        if end_time is not None:
            sql += " AND created_at <= ?"
            params.append(_epoch_to_iso(end_time))

        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            sql += f" AND {frag}"
            params.extend(sparams)
        else:
            sql += " AND status IS NULL"

        if playbook_status_filter:
            ph = ",".join("?" for _ in playbook_status_filter)
            sql += f" AND playbook_status IN ({ph})"
            params.extend(ps.value for ps in playbook_status_filter)
        tag_frag, tag_params = _build_tags_sql("agent_playbooks", tags)
        if tag_frag:
            sql += f" AND {tag_frag}"
            params.extend(tag_params)

        if max_agent_playbook_id is not None:
            sql += " ORDER BY agent_playbook_id DESC LIMIT ?"
            params.append(limit)
        else:
            sql += " ORDER BY created_at DESC, agent_playbook_id DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self._fetchall(sql, params)
        return [_row_to_agent_playbook(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_agent_playbook_by_id(
        self, agent_playbook_id: int, *, include_tombstones: bool = False
    ) -> AgentPlaybook | None:
        sql = "SELECT * FROM agent_playbooks WHERE agent_playbook_id = ?"
        if not include_tombstones:
            _ph = ",".join("?" * len(_AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES))
            sql += f" AND (status IS NULL OR status NOT IN ({_ph}))"
            row = self._fetchone(
                sql,
                (agent_playbook_id, *_AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES),
            )
        else:
            row = self._fetchone(sql, (agent_playbook_id,))
        return _row_to_agent_playbook(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_agent_playbooks_by_ids(
        self,
        agent_playbook_ids: list[int],
        *,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
        include_inactive: bool = False,
        include_embedding: bool = False,
    ) -> list[AgentPlaybook]:
        validate_include_inactive(
            include_inactive=include_inactive,
            status_filter=status_filter,
            playbook_status_filter=playbook_status_filter,
        )
        if not agent_playbook_ids:
            return []
        unique_ids = list(dict.fromkeys(agent_playbook_ids))
        rows: list[sqlite3.Row] = []
        for start in range(0, len(unique_ids), 900):
            chunk = unique_ids[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT * FROM agent_playbooks "
                f"WHERE agent_playbook_id IN ({placeholders})"
            )
            params: list[Any] = list(chunk)
            if not include_inactive:
                if status_filter is not None:
                    fragment, status_params = _build_status_sql(status_filter)
                    query += f" AND {fragment}"
                    params.extend(status_params)
                else:
                    query += " AND status IS NULL"
                if playbook_status_filter:
                    status_placeholders = ",".join("?" for _ in playbook_status_filter)
                    query += f" AND playbook_status IN ({status_placeholders})"
                    params.extend(status.value for status in playbook_status_filter)
            rows.extend(self._fetchall(query, params))
        result = [_row_to_agent_playbook(row) for row in rows]
        if include_embedding:
            for playbook, row in zip(result, rows, strict=True):
                playbook.embedding = _json_loads(row["embedding"]) or []
        return result

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_playbooks(self) -> None:
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            ids = [
                r["agent_playbook_id"]
                for r in self.conn.execute(
                    "SELECT agent_playbook_id FROM agent_playbooks"
                ).fetchall()
            ]
            self.conn.execute("DELETE FROM agent_playbooks")
            for apid in ids:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(apid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
        self._delete_playbook_search_rows("agent", ids)

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_playbook(self, agent_playbook_id: int) -> None:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM agent_playbooks WHERE agent_playbook_id = ?",
                (agent_playbook_id,),
            )
            if cur.rowcount > 0:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(agent_playbook_id),
                    request_id=uuid.uuid4().hex,
                )
            self._delete_playbook_search_rows(
                "agent", [agent_playbook_id], commit=False
            )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "SELECT agent_playbook_id FROM agent_playbooks WHERE playbook_name = ?"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            ids = [
                r["agent_playbook_id"]
                for r in self.conn.execute(sql, params).fetchall()
            ]
            if not ids:
                return
            ph = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM agent_playbooks WHERE agent_playbook_id IN ({ph})", ids
            )
            for apid in ids:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(apid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
        self._delete_playbook_search_rows("agent", ids)

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int], *, emit_hard_delete: bool = True
    ) -> None:
        if not agent_playbook_ids:
            return
        ph = ",".join("?" for _ in agent_playbook_ids)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            existing = [
                r["agent_playbook_id"]
                for r in self.conn.execute(
                    f"SELECT agent_playbook_id FROM agent_playbooks WHERE agent_playbook_id IN ({ph})",
                    agent_playbook_ids,
                ).fetchall()
            ]
            self.conn.execute(
                f"DELETE FROM agent_playbooks WHERE agent_playbook_id IN ({ph})",
                agent_playbook_ids,
            )
            if emit_hard_delete:
                for apid in existing:
                    _emit_hard_delete_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(apid),
                        request_id=batch_request_id,
                        actor="system",
                    )
            self.conn.commit()
        self._delete_playbook_search_rows("agent", agent_playbook_ids)

    @SQLiteStorageBase.handle_exceptions
    def update_agent_playbook_status(
        self, agent_playbook_id: int, playbook_status: PlaybookStatus
    ) -> None:
        """Update an agent playbook's status and emit a status_change lineage event.

        Each call generates a fresh request_id so every status change is recorded as
        a distinct audit event (not collapsed by the idempotency key).
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT playbook_status FROM agent_playbooks WHERE agent_playbook_id = ?",
                (agent_playbook_id,),
            ).fetchone()
            if not row:
                raise ValueError(
                    f"Agent playbook with ID {agent_playbook_id} not found"
                )
            prior_playbook_status = row["playbook_status"]
            old_status = prior_playbook_status or "None"
            cur = self.conn.execute(
                "UPDATE agent_playbooks SET playbook_status = ? WHERE agent_playbook_id = ?",
                (playbook_status.value, agent_playbook_id),
            )
            if cur.rowcount > 0:
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(agent_playbook_id),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=uuid.uuid4().hex,
                    reason=f"{old_status}->{playbook_status.value}",
                    from_status=prior_playbook_status,
                    to_status=playbook_status.value,
                    status_namespace="playbook_status",
                )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def update_agent_playbook(
        self,
        agent_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        trigger: str | None = None,
        rationale: str | None = None,
        blocking_issue: BlockingIssue | None = None,
        playbook_status: PlaybookStatus | None = None,
        tags: list[str] | None = None,
    ) -> None:
        updates: list[str] = []
        params: list[Any] = []
        if playbook_name is not None:
            updates.append("playbook_name = ?")
            params.append(playbook_name)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if trigger is not None:
            updates.append("trigger = ?")
            params.append(trigger)
        if rationale is not None:
            updates.append("rationale = ?")
            params.append(rationale)
        if blocking_issue is not None:
            updates.append("blocking_issue = ?")
            params.append(json.dumps(blocking_issue.model_dump()))
        if playbook_status is not None:
            updates.append("playbook_status = ?")
            params.append(playbook_status.value)
        if tags is not None:
            updates.append("tags = ?")
            params.append(_json_dumps(tags))
        if updates:
            params.append(agent_playbook_id)
            op = "revise" if content is not None else "status_change"
            prov = "wasRevisionOf" if op == "revise" else "wasInvalidatedBy"
            with self._lock:
                prior_row = self.conn.execute(
                    "SELECT playbook_status FROM agent_playbooks WHERE agent_playbook_id = ?",
                    (agent_playbook_id,),
                ).fetchone()
                if not prior_row:
                    raise ValueError(
                        f"Agent playbook with ID {agent_playbook_id} not found"
                    )
                prior_playbook_status = prior_row["playbook_status"]
                cur = self.conn.execute(
                    f"UPDATE agent_playbooks SET {', '.join(updates)} WHERE agent_playbook_id = ?",
                    tuple(params),
                )
                if cur.rowcount > 0:
                    # Populate structured status fields only when playbook_status is
                    # among the updated fields and the op is status_change (not revise).
                    if op == "status_change" and playbook_status is not None:
                        from_status = prior_playbook_status
                        to_status = playbook_status.value
                        status_namespace: str | None = "playbook_status"
                    else:
                        from_status = None
                        to_status = None
                        status_namespace = None
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="agent_playbook",
                        entity_id=str(agent_playbook_id),
                        op=op,
                        prov=prov,
                        source_ids=[],
                        actor="api",
                        request_id=uuid.uuid4().hex,
                        reason="in-place update",
                        from_status=from_status,
                        to_status=to_status,
                        status_namespace=status_namespace,
                    )
                self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def archive_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        where = (
            "playbook_name = ? AND playbook_status != ?"
            " AND (status IS NULL OR status != 'archived')"
        )
        params: list[Any] = [playbook_name, PlaybookStatus.APPROVED.value]
        if agent_version is not None:
            where += " AND agent_version = ?"
            params.append(agent_version)
        now_ts = _epoch_now()
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = self.conn.execute(
                f"SELECT agent_playbook_id, status FROM agent_playbooks WHERE {where}",
                params,
            ).fetchall()
            self.conn.execute(
                f"UPDATE agent_playbooks SET status = 'archived', retired_at = ? WHERE {where}",
                [now_ts, *params],
            )
            for row in affected:
                prior = row["status"] or "None"
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(row["agent_playbook_id"]),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=batch_request_id,
                    reason=f"{prior}->archived",
                    from_status=row["status"],
                    to_status="archived",
                    status_namespace="lifecycle_status",
                )
            if self._own_transaction():
                self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def archive_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        if not agent_playbook_ids:
            return
        ph = ",".join("?" for _ in agent_playbook_ids)
        now_ts = _epoch_now()
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = self.conn.execute(
                f"SELECT agent_playbook_id, status FROM agent_playbooks"
                f" WHERE agent_playbook_id IN ({ph}) AND playbook_status != ?"
                f" AND (status IS NULL OR status != 'archived')",
                [*agent_playbook_ids, PlaybookStatus.APPROVED.value],
            ).fetchall()
            self.conn.execute(
                f"UPDATE agent_playbooks SET status = 'archived', retired_at = ?"
                f" WHERE agent_playbook_id IN ({ph}) AND playbook_status != ?"
                f" AND (status IS NULL OR status != 'archived')",
                [now_ts, *agent_playbook_ids, PlaybookStatus.APPROVED.value],
            )
            for row in affected:
                prior = row["status"] or "None"
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(row["agent_playbook_id"]),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=batch_request_id,
                    reason=f"{prior}->archived",
                    from_status=row["status"],
                    to_status="archived",
                    status_namespace="lifecycle_status",
                )
            if self._own_transaction():
                self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def supersede_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int], request_id: str
    ) -> int:
        """Soft-delete agent playbooks by setting status to SUPERSEDED, emitting set-based lineage.

        For each eligible id (not APPROVED, not already tombstoned), updates status to
        SUPERSEDED and emits one ``status_change`` event under the shared ``request_id``.
        Atomic: one ``conn.commit()`` at the end, guarded on rowcount per id.
        FTS/vec rows are NOT removed — reads exclude tombstones by status filter.

        Args:
            agent_playbook_ids (list[int]): Agent playbook ids to supersede.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of agent playbooks actually updated.
        """
        if not agent_playbook_ids:
            return 0
        if not request_id:
            raise ValueError("request_id must be non-empty for supersede")
        now_ts = _epoch_now()
        updated = 0
        with self._lock:
            for apid in agent_playbook_ids:
                row = self.conn.execute(
                    "SELECT status FROM agent_playbooks WHERE agent_playbook_id = ?",
                    (apid,),
                ).fetchone()
                if row is None:
                    continue
                old_status = row["status"]
                # NOTE (M3): The model supersede_profiles_by_ids adds an eligible-check
                # `if old_status_val not in eligible: continue` before the UPDATE. For
                # agent_playbooks the ineligible condition spans two columns (status in
                # _TOMBSTONE_STATUS_VALUES OR playbook_status == APPROVED), so aligning
                # would require adding playbook_status to the SELECT. The UPDATE WHERE
                # clause already excludes those rows atomically; the extra continue here
                # would be a no-op and not worth the added complexity.
                _ph = ",".join("?" * len(_TOMBSTONE_STATUS_VALUES))
                cur = self.conn.execute(
                    "UPDATE agent_playbooks SET status = ?, retired_at = ?"
                    " WHERE agent_playbook_id = ? AND playbook_status != ?"
                    f" AND (status IS NULL OR status NOT IN ({_ph}))",
                    (
                        Status.SUPERSEDED.value,
                        now_ts,
                        apid,
                        PlaybookStatus.APPROVED.value,
                        *_TOMBSTONE_STATUS_VALUES,
                    ),
                )
                if cur.rowcount > 0:
                    _emit_supersede_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_id=str(apid),
                        old_status=old_status,
                        request_id=request_id,
                    )
                    updated += 1
            if self._own_transaction():
                self.conn.commit()
        return updated

    @SQLiteStorageBase.handle_exceptions
    def supersede_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None, request_id: str
    ) -> int:
        """Soft-delete archived agent playbooks by name/version via SUPERSEDED status.

        Mirrors ``delete_archived_agent_playbooks_by_playbook_name`` but converts
        the hard-delete to a soft-supersede with status_change lineage events.
        Atomic: one ``conn.commit()`` at the end.
        FTS/vec rows are NOT removed.

        Args:
            playbook_name (str): Playbook name to supersede.
            agent_version (str | None): Agent version filter. None matches all versions.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of agent playbooks actually updated.
        """
        if not request_id:
            raise ValueError("request_id must be non-empty for supersede")
        sql = (
            "SELECT agent_playbook_id, status FROM agent_playbooks"
            " WHERE playbook_name = ? AND status = 'archived'"
        )
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        updated = 0
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
            if not rows:
                return 0
            now_ts = _epoch_now()
            for row in rows:
                apid = row["agent_playbook_id"]
                old_status = row["status"]
                cur = self.conn.execute(
                    "UPDATE agent_playbooks SET status = ?, retired_at = ?"
                    " WHERE agent_playbook_id = ? AND playbook_status != ?"
                    " AND status = 'archived'",
                    (
                        Status.SUPERSEDED.value,
                        now_ts,
                        apid,
                        PlaybookStatus.APPROVED.value,
                    ),
                )
                if cur.rowcount > 0:
                    _emit_supersede_playbook(
                        self.conn,
                        org_id=self.org_id,
                        entity_id=str(apid),
                        old_status=old_status,
                        request_id=request_id,
                    )
                    updated += 1
            if self._own_transaction():
                self.conn.commit()
        return updated

    @SQLiteStorageBase.handle_exceptions
    def restore_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "UPDATE agent_playbooks SET status = NULL, retired_at = NULL WHERE playbook_name = ? AND status = 'archived'"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        self._execute(sql, params)

    @SQLiteStorageBase.handle_exceptions
    def restore_archived_agent_playbooks_by_ids(
        self, agent_playbook_ids: list[int]
    ) -> None:
        if not agent_playbook_ids:
            return
        ph = ",".join("?" for _ in agent_playbook_ids)
        self._execute(
            f"UPDATE agent_playbooks SET status = NULL, retired_at = NULL WHERE agent_playbook_id IN ({ph}) AND status = 'archived'",
            agent_playbook_ids,
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "SELECT agent_playbook_id FROM agent_playbooks WHERE playbook_name = ? AND status = 'archived'"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            ids = [
                r["agent_playbook_id"]
                for r in self.conn.execute(sql, params).fetchall()
            ]
            if not ids:
                return
            ph = ",".join("?" for _ in ids)
            self.conn.execute(
                f"DELETE FROM agent_playbooks WHERE agent_playbook_id IN ({ph})", ids
            )
            for apid in ids:
                _emit_hard_delete_playbook(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="agent_playbook",
                    entity_id=str(apid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
        self._delete_playbook_search_rows("agent", ids)

    @SQLiteStorageBase.handle_exceptions
    def search_agent_playbooks(  # noqa: C901
        self,
        request: SearchAgentPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[AgentPlaybook]:
        query = request.query
        agent_version = request.agent_version
        playbook_name = request.playbook_name
        start_time = int(request.start_time.timestamp()) if request.start_time else None
        end_time = int(request.end_time.timestamp()) if request.end_time else None
        status_filter = request.status_filter
        playbook_status_filter = request.playbook_status_filter
        match_count = request.top_k or 10
        query_embedding = options.query_embedding if options else None
        mode = _effective_search_mode(
            request.search_mode, query_embedding, request.query
        )
        threshold = resolve_retrieval_threshold(
            request.threshold,
            model_name=self.embedding_model_name,
        )
        rrf_k = options.rrf_k if options else 60
        vector_weight = options.vector_weight if options else 1.0
        fts_weight = options.fts_weight if options else 1.0

        conditions: list[str] = []
        params: list[Any] = []

        if agent_version:
            conditions.append("ap.agent_version = ?")
            params.append(agent_version)
        if playbook_name:
            conditions.append("ap.playbook_name = ?")
            params.append(playbook_name)
        if start_time:
            conditions.append("ap.created_at >= ?")
            params.append(_epoch_to_iso(start_time))
        if end_time:
            conditions.append("ap.created_at <= ?")
            params.append(_epoch_to_iso(end_time))
        if playbook_status_filter is not None:
            if isinstance(playbook_status_filter, list):
                if not playbook_status_filter:
                    conditions.append("1=0")
                else:
                    placeholders = ",".join("?" for _ in playbook_status_filter)
                    conditions.append(f"ap.playbook_status IN ({placeholders})")
                    params.extend(s.value for s in playbook_status_filter)
            else:
                conditions.append("ap.playbook_status = ?")
                params.append(playbook_status_filter.value)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)
        else:
            _ph = ",".join("?" * len(_AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES))
            conditions.append(f"(ap.status IS NULL OR ap.status NOT IN ({_ph}))")
            params.extend(_AGENT_PLAYBOOK_DEFAULT_EXCLUDED_STATUSES)
        tag_frag, tag_params = _build_tags_sql("ap", request.tags)
        if tag_frag:
            conditions.append(tag_frag)
            params.extend(tag_params)

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM agent_playbooks ap
                      {base_where}
                      ORDER BY ap.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(
                rows,
                query_embedding,
                match_count,
                threshold=threshold,
            )
            return [_row_to_agent_playbook(r) for r in rows]

        if query:
            fts_query = _sanitize_fts_query(query)
            sql = f"""SELECT ap.* FROM agent_playbooks ap
                      JOIN agent_playbooks_fts f ON ap.agent_playbook_id = f.rowid
                      WHERE agent_playbooks_fts MATCH ?{where_extra}
                      ORDER BY bm25(agent_playbooks_fts, 1.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, [fts_query, *params, overfetch])
            if _uses_unicode_lexical_fallback(query):
                base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
                unicode_sql = f"""SELECT ranked.* FROM (
                              SELECT ap.*,
                                reflexio_unicode_lexical_score(
                                  COALESCE(ap.trigger, '') || ' ' ||
                                  COALESCE(ap.content, '') || ' ' ||
                                  COALESCE(ap.rationale, ''), ?
                                ) AS unicode_score
                              FROM agent_playbooks ap
                              {base_where}
                            ) AS ranked
                            WHERE ranked.unicode_score > 0
                            ORDER BY ranked.unicode_score DESC
                            LIMIT ?"""
                unicode_rows = self._fetchall(
                    unicode_sql,
                    [query, *params, overfetch],
                )
                fts_rows = _true_rrf_merge(
                    fts_rows,
                    unicode_rows,
                    "agent_playbook_id",
                    overfetch,
                )

            if mode == SearchMode.HYBRID and query_embedding:
                base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
                vec_limit = match_count * 10
                vec_sql = f"""SELECT * FROM agent_playbooks ap
                              {base_where}
                              ORDER BY ap.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, [*params, vec_limit])
                vec_rows = _vector_rank_rows(
                    vec_candidates,
                    query_embedding,
                    overfetch,
                    threshold=threshold,
                )
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "agent_playbook_id",
                    match_count,
                    rrf_k,
                    vector_weight,
                    fts_weight,
                )
                return [_row_to_agent_playbook(r) for r in rows]
            return [_row_to_agent_playbook(r) for r in fts_rows[:match_count]]

        # HYBRID without query text: rank by embedding only
        if query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM agent_playbooks ap
                      {base_where}
                      ORDER BY ap.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(
                rows,
                query_embedding,
                match_count,
                threshold=threshold,
            )
            return [_row_to_agent_playbook(r) for r in rows]

        # No query text, no embedding -- recency fallback
        base_where = "WHERE " + " AND ".join(conditions) if conditions else "WHERE 1=1"
        sql = f"""SELECT * FROM agent_playbooks ap
                  {base_where}
                  ORDER BY ap.created_at DESC LIMIT ?"""
        params.append(match_count)
        rows = self._fetchall(sql, params)
        return [_row_to_agent_playbook(r) for r in rows]
