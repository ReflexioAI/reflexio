"""Playbook CRUD + search methods for SQLite storage."""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchUserPlaybookRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    AgentSuccessEvaluationResult,
    PlaybookStatus,
    Status,
    StructuredData,
    UserPlaybook,
)
from reflexio.models.config_schema import SearchMode, SearchOptions

from ._base import (
    SQLiteStorageBase,
    _build_status_sql,
    _effective_search_mode,
    _epoch_to_iso,
    _json_dumps,
    _row_to_agent_playbook,
    _row_to_eval_result,
    _row_to_user_playbook,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
)

logger = logging.getLogger(__name__)


class PlaybookMixin:
    """Mixin providing user playbook, agent playbook, and evaluation CRUD + search."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    _execute: Any
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _should_expand_documents: Any
    _expand_document: Any
    _fts_upsert: Any
    _fts_delete: Any
    _vec_upsert: Any
    _vec_delete: Any
    _has_sqlite_vec: bool

    # ------------------------------------------------------------------
    # User Playbook methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_user_playbooks(self, user_playbooks: list[UserPlaybook]) -> None:
        for up in user_playbooks:
            sd = up.structured_data
            embedding_text = sd.embedding_text or sd.trigger or up.content
            if embedding_text:
                # Embeddings are best-effort. When no embedding provider is
                # configured (e.g. the LLM-free OpenClaw setup), vector
                # ranking is unavailable but FTS5 still works for retrieval.
                # On failure, leave ``up.embedding`` as the empty list so the
                # in-memory model stays valid, and rely on ``up.embedding or
                # None`` at INSERT time to store SQL NULL — that lets a
                # future re-embed migration target the row via
                # ``WHERE embedding IS NULL``.
                if self._should_expand_documents():
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        emb_future = executor.submit(
                            self._get_embedding, embedding_text
                        )
                        exp_future = executor.submit(
                            self._expand_document, embedding_text
                        )
                        try:
                            up.embedding = emb_future.result(timeout=15)
                        except Exception as exc:  # noqa: BLE001
                            logger.warning(
                                "Embedding generation failed for user "
                                "playbook; saving without vector (FTS only): %s",
                                exc,
                            )
                            up.embedding = []
                        up.expanded_terms = exp_future.result(timeout=15)
                else:
                    try:
                        up.embedding = self._get_embedding(embedding_text)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Embedding generation failed for user "
                            "playbook; saving without vector (FTS only): %s",
                            exc,
                        )
                        up.embedding = []

            created_at_iso = _epoch_to_iso(up.created_at)
            with self._lock:
                cur = self.conn.execute(
                    """INSERT INTO user_playbooks
                       (user_id, playbook_name, created_at, request_id, agent_version,
                        content, structured_data, source_interaction_ids,
                        status, source, embedding, expanded_terms)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        up.user_id,
                        up.playbook_name,
                        created_at_iso,
                        up.request_id,
                        up.agent_version,
                        up.content,
                        json.dumps(up.structured_data.model_dump(exclude_none=True)),
                        _json_dumps(up.source_interaction_ids or None),
                        up.status.value if up.status else None,
                        up.source,
                        _json_dumps(up.embedding or None),
                        up.expanded_terms,
                    ),
                )
                upid = cur.lastrowid or 0
                up.user_playbook_id = upid
                self.conn.commit()

            fts_parts = [sd.trigger or "", up.content or ""]
            if up.expanded_terms:
                fts_parts.append(up.expanded_terms)
            self._fts_upsert(
                "user_playbooks_fts",
                upid,
                search_text=" ".join(p for p in fts_parts if p) or "",
            )
            if up.embedding:
                self._vec_upsert("user_playbooks_vec", upid, up.embedding)

    @SQLiteStorageBase.handle_exceptions
    def get_user_playbooks(
        self,
        limit: int = 100,
        user_id: str | None = None,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        include_embedding: bool = False,
    ) -> list[UserPlaybook]:
        sql = "SELECT * FROM user_playbooks WHERE 1=1"
        params: list[Any] = []

        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
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

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [
            _row_to_user_playbook(r, include_embedding=include_embedding) for r in rows
        ]

    @SQLiteStorageBase.handle_exceptions
    def count_user_playbooks(
        self,
        user_id: str | None = None,
        playbook_name: str | None = None,
        min_user_playbook_id: int | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
    ) -> int:
        sql = "SELECT COUNT(*) as cnt FROM user_playbooks WHERE 1=1"
        params: list[Any] = []

        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        if playbook_name:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)
        if min_user_playbook_id is not None:
            sql += " AND user_playbook_id > ?"
            params.append(min_user_playbook_id)
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            sql += f" AND {frag}"
            params.extend(sparams)

        row = self._fetchone(sql, params)
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def count_user_playbooks_by_session(self, session_id: str) -> int:
        row = self._fetchone(
            """SELECT COUNT(*) as cnt FROM user_playbooks up
               JOIN requests r ON up.request_id = r.request_id
               WHERE r.session_id = ?""",
            (session_id,),
        )
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM user_playbooks_fts")
            self.conn.execute("DELETE FROM user_playbooks")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_user_playbook(self, user_playbook_id: int) -> None:
        self._fts_delete("user_playbooks_fts", user_playbook_id)
        self._vec_delete("user_playbooks_vec", user_playbook_id)
        self._execute(
            "DELETE FROM user_playbooks WHERE user_playbook_id = ?", (user_playbook_id,)
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "SELECT user_playbook_id FROM user_playbooks WHERE playbook_name = ?"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        ids = [r["user_playbook_id"] for r in self._fetchall(sql, params)]
        if ids:
            ph = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM user_playbooks_fts WHERE rowid IN ({ph})", ids
                )
                self.conn.commit()

        del_sql = "DELETE FROM user_playbooks WHERE playbook_name = ?"
        del_params: list[Any] = [playbook_name]
        if agent_version is not None:
            del_sql += " AND agent_version = ?"
            del_params.append(agent_version)
        self._execute(del_sql, del_params)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_playbooks_by_ids(self, user_playbook_ids: list[int]) -> int:
        if not user_playbook_ids:
            return 0
        ph = ",".join("?" for _ in user_playbook_ids)
        with self._lock:
            self.conn.execute(
                f"DELETE FROM user_playbooks_fts WHERE rowid IN ({ph})",
                user_playbook_ids,
            )
            cur = self.conn.execute(
                f"DELETE FROM user_playbooks WHERE user_playbook_id IN ({ph})",
                user_playbook_ids,
            )
            self.conn.commit()
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def update_all_user_playbooks_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        new_val = new_status.value if new_status else None
        params: list[Any] = [new_val]

        if old_status is None or (
            hasattr(old_status, "value") and old_status.value is None
        ):
            where = "status IS NULL"
        else:
            where = "status = ?"
            params.append(old_status.value)

        if agent_version is not None:
            where += " AND agent_version = ?"
            params.append(agent_version)
        if playbook_name is not None:
            where += " AND playbook_name = ?"
            params.append(playbook_name)

        cur = self._execute(
            f"UPDATE user_playbooks SET status = ? WHERE {where}", params
        )
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def delete_all_user_playbooks_by_status(
        self,
        status: Status,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> int:
        where = "status = ?"
        params: list[Any] = [status.value]
        if agent_version is not None:
            where += " AND agent_version = ?"
            params.append(agent_version)
        if playbook_name is not None:
            where += " AND playbook_name = ?"
            params.append(playbook_name)

        # Clean up FTS
        ids = [
            r["user_playbook_id"]
            for r in self._fetchall(
                f"SELECT user_playbook_id FROM user_playbooks WHERE {where}", params
            )
        ]
        if ids:
            ph = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM user_playbooks_fts WHERE rowid IN ({ph})", ids
                )
                self.conn.commit()

        cur = self._execute(f"DELETE FROM user_playbooks WHERE {where}", params)
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def has_user_playbooks_with_status(
        self,
        status: Status | None,
        agent_version: str | None = None,
        playbook_name: str | None = None,
    ) -> bool:
        sql = "SELECT 1 FROM user_playbooks WHERE "
        params: list[Any] = []

        if status is None or (hasattr(status, "value") and status.value is None):
            sql += "status IS NULL"
        else:
            sql += "status = ?"
            params.append(status.value)

        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        if playbook_name is not None:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)

        sql += " LIMIT 1"
        row = self._fetchone(sql, params)
        return row is not None

    @SQLiteStorageBase.handle_exceptions
    def search_user_playbooks(  # noqa: C901
        self,
        request: SearchUserPlaybookRequest,
        options: SearchOptions | None = None,
    ) -> list[UserPlaybook]:
        query = request.query
        user_id = request.user_id
        agent_version = request.agent_version
        playbook_name = request.playbook_name
        start_time = int(request.start_time.timestamp()) if request.start_time else None
        end_time = int(request.end_time.timestamp()) if request.end_time else None
        status_filter = request.status_filter
        match_count = request.top_k or 10
        query_embedding = options.query_embedding if options else None
        mode = _effective_search_mode(request.search_mode, query_embedding)
        rrf_k = options.rrf_k if options else 60
        vector_weight = options.vector_weight if options else 1.0
        fts_weight = options.fts_weight if options else 1.0

        conditions: list[str] = []
        params: list[Any] = []

        if user_id:
            conditions.append("up.user_id = ?")
            params.append(user_id)
        if agent_version:
            conditions.append("up.agent_version = ?")
            params.append(agent_version)
        if playbook_name:
            conditions.append("up.playbook_name = ?")
            params.append(playbook_name)
        if start_time:
            conditions.append("up.created_at >= ?")
            params.append(_epoch_to_iso(start_time))
        if end_time:
            conditions.append("up.created_at <= ?")
            params.append(_epoch_to_iso(end_time))
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM user_playbooks up
                      {base_where}
                      ORDER BY up.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(rows, query_embedding, match_count)
            return [_row_to_user_playbook(r) for r in rows]

        if query:
            fts_query = _sanitize_fts_query(query)
            sql = f"""SELECT up.* FROM user_playbooks up
                      JOIN user_playbooks_fts f ON up.user_playbook_id = f.rowid
                      WHERE user_playbooks_fts MATCH ?{where_extra}
                      ORDER BY bm25(user_playbooks_fts, 1.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, [fts_query, *params, overfetch])

            if mode == SearchMode.HYBRID and query_embedding:
                base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
                vec_limit = match_count * 10
                vec_sql = f"""SELECT * FROM user_playbooks up
                              {base_where}
                              ORDER BY up.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, [*params, vec_limit])
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "user_playbook_id",
                    match_count,
                    rrf_k,
                    vector_weight,
                    fts_weight,
                )
                return [_row_to_user_playbook(r) for r in rows]
            return [_row_to_user_playbook(r) for r in fts_rows[:match_count]]

        # HYBRID without query text: rank by embedding only
        if query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM user_playbooks up
                      {base_where}
                      ORDER BY up.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(rows, query_embedding, match_count)
            return [_row_to_user_playbook(r) for r in rows]

        # No query text, no embedding -- recency fallback
        base_where = "WHERE " + " AND ".join(conditions) if conditions else "WHERE 1=1"
        sql = f"""SELECT * FROM user_playbooks up
                  {base_where}
                  ORDER BY up.created_at DESC LIMIT ?"""
        params.append(match_count)
        rows = self._fetchall(sql, params)
        return [_row_to_user_playbook(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent Playbook methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_agent_playbooks(
        self, agent_playbooks: list[AgentPlaybook]
    ) -> list[AgentPlaybook]:
        saved: list[AgentPlaybook] = []
        for ap in agent_playbooks:
            sd = ap.structured_data
            embedding_text = sd.embedding_text or sd.trigger or ap.content
            if self._should_expand_documents():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    emb_future = executor.submit(self._get_embedding, embedding_text)
                    exp_future = executor.submit(self._expand_document, embedding_text)
                    ap.embedding = emb_future.result(timeout=15)
                    ap.expanded_terms = exp_future.result(timeout=15)
            else:
                ap.embedding = self._get_embedding(embedding_text)

            created_at_iso = _epoch_to_iso(ap.created_at)
            with self._lock:
                cur = self.conn.execute(
                    """INSERT INTO agent_playbooks
                       (playbook_name, created_at, agent_version, content,
                        structured_data,
                        playbook_status, playbook_metadata, embedding,
                        expanded_terms, status)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        ap.playbook_name,
                        created_at_iso,
                        ap.agent_version,
                        ap.content,
                        json.dumps(ap.structured_data.model_dump(exclude_none=True)),
                        ap.playbook_status.value
                        if isinstance(ap.playbook_status, PlaybookStatus)
                        else ap.playbook_status,
                        ap.playbook_metadata,
                        _json_dumps(ap.embedding),
                        ap.expanded_terms,
                        ap.status.value if ap.status else None,
                    ),
                )
                ap.agent_playbook_id = cur.lastrowid or 0
                self.conn.commit()

            fts_parts = [sd.trigger or "", ap.content or ""]
            if ap.expanded_terms:
                fts_parts.append(ap.expanded_terms)
            self._fts_upsert(
                "agent_playbooks_fts",
                ap.agent_playbook_id,
                search_text=" ".join(p for p in fts_parts if p) or "",
            )
            if ap.embedding:
                self._vec_upsert(
                    "agent_playbooks_vec", ap.agent_playbook_id, ap.embedding
                )
            saved.append(ap)
        return saved

    @SQLiteStorageBase.handle_exceptions
    def get_agent_playbooks(
        self,
        limit: int = 100,
        playbook_name: str | None = None,
        agent_version: str | None = None,
        status_filter: list[Status | None] | None = None,
        playbook_status_filter: list[PlaybookStatus] | None = None,
    ) -> list[AgentPlaybook]:
        sql = "SELECT * FROM agent_playbooks WHERE 1=1"
        params: list[Any] = []

        if playbook_name:
            sql += " AND playbook_name = ?"
            params.append(playbook_name)

        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)

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

        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_agent_playbook(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_playbooks(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM agent_playbooks_fts")
            self.conn.execute("DELETE FROM agent_playbooks")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_playbook(self, agent_playbook_id: int) -> None:
        self._fts_delete("agent_playbooks_fts", agent_playbook_id)
        self._vec_delete("agent_playbooks_vec", agent_playbook_id)
        self._execute(
            "DELETE FROM agent_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "SELECT agent_playbook_id FROM agent_playbooks WHERE playbook_name = ?"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        ids = [r["agent_playbook_id"] for r in self._fetchall(sql, params)]
        if ids:
            ph = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM agent_playbooks_fts WHERE rowid IN ({ph})", ids
                )
                self.conn.commit()

        del_sql = "DELETE FROM agent_playbooks WHERE playbook_name = ?"
        del_params: list[Any] = [playbook_name]
        if agent_version is not None:
            del_sql += " AND agent_version = ?"
            del_params.append(agent_version)
        self._execute(del_sql, del_params)

    @SQLiteStorageBase.handle_exceptions
    def delete_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        if not agent_playbook_ids:
            return
        ph = ",".join("?" for _ in agent_playbook_ids)
        with self._lock:
            self.conn.execute(
                f"DELETE FROM agent_playbooks_fts WHERE rowid IN ({ph})",
                agent_playbook_ids,
            )
            self.conn.execute(
                f"DELETE FROM agent_playbooks WHERE agent_playbook_id IN ({ph})",
                agent_playbook_ids,
            )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def update_agent_playbook_status(
        self, agent_playbook_id: int, playbook_status: PlaybookStatus
    ) -> None:
        row = self._fetchone(
            "SELECT agent_playbook_id FROM agent_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )
        if not row:
            raise ValueError(f"Agent playbook with ID {agent_playbook_id} not found")
        self._execute(
            "UPDATE agent_playbooks SET playbook_status = ? WHERE agent_playbook_id = ?",
            (playbook_status.value, agent_playbook_id),
        )

    @SQLiteStorageBase.handle_exceptions
    def update_agent_playbook(
        self,
        agent_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        structured_data: StructuredData | None = None,
        playbook_status: PlaybookStatus | None = None,
    ) -> None:
        row = self._fetchone(
            "SELECT agent_playbook_id FROM agent_playbooks WHERE agent_playbook_id = ?",
            (agent_playbook_id,),
        )
        if not row:
            raise ValueError(f"Agent playbook with ID {agent_playbook_id} not found")
        updates: list[str] = []
        params: list[Any] = []
        if playbook_name is not None:
            updates.append("playbook_name = ?")
            params.append(playbook_name)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if structured_data is not None:
            updates.append("structured_data = ?")
            params.append(json.dumps(structured_data.model_dump(exclude_none=True)))
        if playbook_status is not None:
            updates.append("playbook_status = ?")
            params.append(playbook_status.value)
        if updates:
            params.append(agent_playbook_id)
            self._execute(
                f"UPDATE agent_playbooks SET {', '.join(updates)} WHERE agent_playbook_id = ?",
                tuple(params),
            )

    @SQLiteStorageBase.handle_exceptions
    def update_user_playbook(
        self,
        user_playbook_id: int,
        playbook_name: str | None = None,
        content: str | None = None,
        structured_data: StructuredData | None = None,
    ) -> None:
        row = self._fetchone(
            "SELECT user_playbook_id FROM user_playbooks WHERE user_playbook_id = ?",
            (user_playbook_id,),
        )
        if not row:
            raise ValueError(f"User playbook with ID {user_playbook_id} not found")
        updates: list[str] = []
        params: list[Any] = []
        if playbook_name is not None:
            updates.append("playbook_name = ?")
            params.append(playbook_name)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if structured_data is not None:
            updates.append("structured_data = ?")
            params.append(json.dumps(structured_data.model_dump(exclude_none=True)))
        if updates:
            params.append(user_playbook_id)
            self._execute(
                f"UPDATE user_playbooks SET {', '.join(updates)} WHERE user_playbook_id = ?",
                tuple(params),
            )

    @SQLiteStorageBase.handle_exceptions
    def archive_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "UPDATE agent_playbooks SET status = 'archived' WHERE playbook_name = ? AND playbook_status != ?"
        params: list[Any] = [playbook_name, PlaybookStatus.APPROVED.value]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        self._execute(sql, params)

    @SQLiteStorageBase.handle_exceptions
    def archive_agent_playbooks_by_ids(self, agent_playbook_ids: list[int]) -> None:
        if not agent_playbook_ids:
            return
        ph = ",".join("?" for _ in agent_playbook_ids)
        self._execute(
            f"UPDATE agent_playbooks SET status = 'archived' WHERE agent_playbook_id IN ({ph}) AND playbook_status != ?",
            [*agent_playbook_ids, PlaybookStatus.APPROVED.value],
        )

    @SQLiteStorageBase.handle_exceptions
    def restore_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        sql = "UPDATE agent_playbooks SET status = NULL WHERE playbook_name = ? AND status = 'archived'"
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
            f"UPDATE agent_playbooks SET status = NULL WHERE agent_playbook_id IN ({ph}) AND status = 'archived'",
            agent_playbook_ids,
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_archived_agent_playbooks_by_playbook_name(
        self, playbook_name: str, agent_version: str | None = None
    ) -> None:
        # Get IDs for FTS cleanup
        sql = "SELECT agent_playbook_id FROM agent_playbooks WHERE playbook_name = ? AND status = 'archived'"
        params: list[Any] = [playbook_name]
        if agent_version is not None:
            sql += " AND agent_version = ?"
            params.append(agent_version)
        ids = [r["agent_playbook_id"] for r in self._fetchall(sql, params)]
        if ids:
            ph = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM agent_playbooks_fts WHERE rowid IN ({ph})", ids
                )
                self.conn.commit()

        del_sql = "DELETE FROM agent_playbooks WHERE playbook_name = ? AND status = 'archived'"
        del_params: list[Any] = [playbook_name]
        if agent_version is not None:
            del_sql += " AND agent_version = ?"
            del_params.append(agent_version)
        self._execute(del_sql, del_params)

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
        mode = _effective_search_mode(request.search_mode, query_embedding)
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
        if playbook_status_filter:
            conditions.append("ap.playbook_status = ?")
            params.append(playbook_status_filter.value)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)

        where_extra = (" AND " + " AND ".join(conditions)) if conditions else ""
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
            sql = f"""SELECT * FROM agent_playbooks ap
                      {base_where}
                      ORDER BY ap.created_at DESC"""
            rows = self._fetchall(sql, params)
            rows = _vector_rank_rows(rows, query_embedding, match_count)
            return [_row_to_agent_playbook(r) for r in rows]

        if query:
            fts_query = _sanitize_fts_query(query)
            sql = f"""SELECT ap.* FROM agent_playbooks ap
                      JOIN agent_playbooks_fts f ON ap.agent_playbook_id = f.rowid
                      WHERE agent_playbooks_fts MATCH ?{where_extra}
                      ORDER BY bm25(agent_playbooks_fts, 1.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, [fts_query, *params, overfetch])

            if mode == SearchMode.HYBRID and query_embedding:
                base_where = "WHERE " + " AND ".join(conditions) if conditions else ""
                vec_limit = match_count * 10
                vec_sql = f"""SELECT * FROM agent_playbooks ap
                              {base_where}
                              ORDER BY ap.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, [*params, vec_limit])
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
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
            rows = _vector_rank_rows(rows, query_embedding, match_count)
            return [_row_to_agent_playbook(r) for r in rows]

        # No query text, no embedding -- recency fallback
        base_where = "WHERE " + " AND ".join(conditions) if conditions else "WHERE 1=1"
        sql = f"""SELECT * FROM agent_playbooks ap
                  {base_where}
                  ORDER BY ap.created_at DESC LIMIT ?"""
        params.append(match_count)
        rows = self._fetchall(sql, params)
        return [_row_to_agent_playbook(r) for r in rows]

    # ------------------------------------------------------------------
    # Agent Success Evaluation methods
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def save_agent_success_evaluation_results(
        self, results: list[AgentSuccessEvaluationResult]
    ) -> None:
        for result in results:
            embedding_text = f"{result.failure_type} {result.failure_reason}"
            if embedding_text.strip():
                result.embedding = self._get_embedding(embedding_text)
            else:
                result.embedding = []

            created_at_iso = _epoch_to_iso(result.created_at)
            self._execute(
                """INSERT INTO agent_success_evaluation_result
                   (session_id, agent_version, evaluation_name, is_success,
                    failure_type, failure_reason, regular_vs_shadow,
                    number_of_correction_per_session, user_turns_to_resolution,
                    is_escalated, embedding, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    result.session_id,
                    result.agent_version,
                    result.evaluation_name,
                    int(result.is_success),
                    result.failure_type,
                    result.failure_reason,
                    result.regular_vs_shadow.value
                    if result.regular_vs_shadow
                    else None,
                    result.number_of_correction_per_session,
                    result.user_turns_to_resolution,
                    int(result.is_escalated),
                    _json_dumps(result.embedding) if result.embedding else None,
                    created_at_iso,
                ),
            )

    @SQLiteStorageBase.handle_exceptions
    def get_agent_success_evaluation_results(
        self, limit: int = 100, agent_version: str | None = None
    ) -> list[AgentSuccessEvaluationResult]:
        sql = "SELECT * FROM agent_success_evaluation_result"
        params: list[Any] = []
        if agent_version is not None:
            sql += " WHERE agent_version = ?"
            params.append(agent_version)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._fetchall(sql, params)
        return [_row_to_eval_result(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_all_agent_success_evaluation_results(self) -> None:
        self._execute("DELETE FROM agent_success_evaluation_result")
