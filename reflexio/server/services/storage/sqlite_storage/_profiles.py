"""Profile and interaction CRUD + search mixins for SQLite storage."""

import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)

from reflexio.models.api_schema.retriever_schema import (
    SearchInteractionRequest,
    SearchUserProfileRequest,
)
from reflexio.models.api_schema.service_schemas import (
    DeleteUserInteractionRequest,
    DeleteUserProfileRequest,
    Interaction,
    Status,
    UserProfile,
)
from reflexio.models.config_schema import SearchMode

from ._base import (
    SQLiteStorageBase,
    _build_status_sql,
    _effective_search_mode,
    _epoch_now,
    _epoch_to_iso,
    _iso_now,
    _json_dumps,
    _row_to_interaction,
    _row_to_profile,
    _sanitize_fts_query,
    _true_rrf_merge,
    _vector_rank_rows,
)


class ProfileMixin:
    """Mixin providing profile and interaction CRUD + search."""

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
    _fts_upsert_profile: Any
    _fts_delete_profile: Any
    _vec_upsert: Any
    _vec_delete: Any
    _has_sqlite_vec: bool
    llm_client: Any
    embedding_model_name: str
    embedding_dimensions: int

    # ------------------------------------------------------------------
    # CRUD — Profiles
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]
        frag, params = _build_status_sql(status_filter)
        sql = f"SELECT * FROM profiles WHERE {frag} ORDER BY last_modified_timestamp DESC LIMIT ?"
        params.append(limit)
        return [_row_to_profile(r) for r in self._fetchall(sql, params)]

    @SQLiteStorageBase.handle_exceptions
    def get_user_profile(
        self,
        user_id: str,
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]
        current_ts = _epoch_now()
        frag, params = _build_status_sql(status_filter)
        sql = f"SELECT * FROM profiles WHERE user_id = ? AND expiration_timestamp >= ? AND {frag}"
        all_params: list[Any] = [user_id, current_ts, *params]
        return [_row_to_profile(r) for r in self._fetchall(sql, all_params)]

    @SQLiteStorageBase.handle_exceptions
    def add_user_profile(self, user_id: str, user_profiles: list[UserProfile]) -> None:  # noqa: ARG002
        for profile in user_profiles:
            embedding_text = "\n".join([profile.content, str(profile.custom_features)])
            if self._should_expand_documents():
                with ThreadPoolExecutor(max_workers=2) as executor:
                    emb_future = executor.submit(self._get_embedding, embedding_text)
                    exp_future = executor.submit(self._expand_document, profile.content)
                    profile.embedding = emb_future.result(timeout=15)
                    profile.expanded_terms = exp_future.result(timeout=15)
            else:
                profile.embedding = self._get_embedding(embedding_text)
            embedding = profile.embedding
            self._execute(
                """INSERT OR REPLACE INTO profiles
                   (profile_id, user_id, content, last_modified_timestamp,
                    generated_from_request_id, profile_time_to_live,
                    expiration_timestamp, custom_features, embedding, source,
                    status, extractor_names, expanded_terms, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    profile.profile_id,
                    profile.user_id,
                    profile.content,
                    profile.last_modified_timestamp,
                    profile.generated_from_request_id,
                    profile.profile_time_to_live.value,
                    profile.expiration_timestamp,
                    _json_dumps(profile.custom_features),
                    _json_dumps(profile.embedding),
                    profile.source,
                    profile.status.value if profile.status else None,
                    _json_dumps(profile.extractor_names),
                    profile.expanded_terms,
                    _iso_now(),
                ),
            )
            fts_parts = [profile.content or ""]
            if profile.custom_features:
                fts_parts.extend(str(v) for v in profile.custom_features.values() if v)
            if profile.expanded_terms:
                fts_parts.append(profile.expanded_terms)
            self._fts_upsert_profile(profile.profile_id, " ".join(fts_parts))
            # Sync vec table — look up implicit rowid via primary key
            row = self._fetchone(
                "SELECT rowid FROM profiles WHERE profile_id = ?",
                (profile.profile_id,),
            )
            if row and embedding:
                self._vec_upsert("profiles_vec", row["rowid"], embedding)

    @SQLiteStorageBase.handle_exceptions
    def update_user_profile_by_id(
        self, user_id: str, profile_id: str, new_profile: UserProfile
    ) -> None:
        current_ts = _epoch_now()
        row = self._fetchone(
            "SELECT profile_id FROM profiles WHERE user_id = ? AND profile_id = ? AND expiration_timestamp >= ?",
            (user_id, profile_id, current_ts),
        )
        if not row:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("User profile not found for user id: %s", user_id)
            return
        embedding = self._get_embedding(
            "\n".join([new_profile.content, str(new_profile.custom_features)])
        )
        new_profile.embedding = embedding
        self._execute(
            """UPDATE profiles SET content=?, last_modified_timestamp=?,
               generated_from_request_id=?, profile_time_to_live=?,
               expiration_timestamp=?, custom_features=?, embedding=?,
               source=?, status=?, extractor_names=?, expanded_terms=?
               WHERE profile_id=?""",
            (
                new_profile.content,
                new_profile.last_modified_timestamp,
                new_profile.generated_from_request_id,
                new_profile.profile_time_to_live.value,
                new_profile.expiration_timestamp,
                _json_dumps(new_profile.custom_features),
                _json_dumps(new_profile.embedding),
                new_profile.source,
                new_profile.status.value if new_profile.status else None,
                _json_dumps(new_profile.extractor_names),
                new_profile.expanded_terms,
                profile_id,
            ),
        )
        fts_parts = [new_profile.content or ""]
        if new_profile.custom_features:
            fts_parts.extend(str(v) for v in new_profile.custom_features.values() if v)
        if new_profile.expanded_terms:
            fts_parts.append(new_profile.expanded_terms)
        self._fts_upsert_profile(profile_id, " ".join(fts_parts))
        rowid_row = self._fetchone(
            "SELECT rowid FROM profiles WHERE profile_id = ?", (profile_id,)
        )
        if rowid_row and embedding:
            self._vec_upsert("profiles_vec", rowid_row["rowid"], embedding)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_profile(self, request: DeleteUserProfileRequest) -> None:
        rowid_row = self._fetchone(
            "SELECT rowid FROM profiles WHERE profile_id = ?",
            (request.profile_id,),
        )
        self._fts_delete_profile(request.profile_id)
        if rowid_row:
            self._vec_delete("profiles_vec", rowid_row["rowid"])
        self._execute(
            "DELETE FROM profiles WHERE user_id = ? AND profile_id = ?",
            (request.user_id, request.profile_id),
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles_for_user(self, user_id: str) -> None:
        pids = [
            r["profile_id"]
            for r in self._fetchall(
                "SELECT profile_id FROM profiles WHERE user_id = ?", (user_id,)
            )
        ]
        for pid in pids:
            self._fts_delete_profile(pid)
        self._execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM profiles_fts")
            self.conn.execute("DELETE FROM profiles")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def count_all_profiles(self) -> int:
        row = self._fetchone("SELECT COUNT(*) as cnt FROM profiles")
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def update_all_profiles_status(
        self,
        old_status: Status | None,
        new_status: Status | None,
        user_ids: list[str] | None = None,
    ) -> int:
        new_val = new_status.value if new_status else None
        now_ts = _epoch_now()
        params: list[Any] = [new_val, now_ts]

        if old_status is None or (
            hasattr(old_status, "value") and old_status.value is None
        ):
            where = "status IS NULL"
        else:
            where = "status = ?"
            params.append(old_status.value)

        if user_ids is not None:
            placeholders = ",".join("?" for _ in user_ids)
            where += f" AND user_id IN ({placeholders})"
            params.extend(user_ids)

        cur = self._execute(
            f"UPDATE profiles SET status = ?, last_modified_timestamp = ? WHERE {where}",
            params,
        )
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles_by_status(self, status: Status) -> int:
        # Clean up FTS for profiles being deleted
        pids = [
            r["profile_id"]
            for r in self._fetchall(
                "SELECT profile_id FROM profiles WHERE status = ?", (status.value,)
            )
        ]
        for pid in pids:
            self._fts_delete_profile(pid)
        cur = self._execute("DELETE FROM profiles WHERE status = ?", (status.value,))
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def get_user_ids_with_status(self, status: Status | None) -> list[str]:
        if status is None or (hasattr(status, "value") and status.value is None):
            rows = self._fetchall(
                "SELECT DISTINCT user_id FROM profiles WHERE status IS NULL"
            )
        else:
            rows = self._fetchall(
                "SELECT DISTINCT user_id FROM profiles WHERE status = ?",
                (status.value,),
            )
        return [r["user_id"] for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def delete_profiles_by_ids(self, profile_ids: list[str]) -> int:
        if not profile_ids:
            return 0
        for pid in profile_ids:
            self._fts_delete_profile(pid)
        ph = ",".join("?" for _ in profile_ids)
        cur = self._execute(
            f"DELETE FROM profiles WHERE profile_id IN ({ph})", profile_ids
        )
        return cur.rowcount

    # ------------------------------------------------------------------
    # CRUD — Interactions
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def get_all_interactions(self, limit: int = 100) -> list[Interaction]:
        rows = self._fetchall(
            "SELECT * FROM interactions ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [_row_to_interaction(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_user_interaction(self, user_id: str) -> list[Interaction]:
        rows = self._fetchall(
            "SELECT * FROM interactions WHERE user_id = ?", (user_id,)
        )
        return [_row_to_interaction(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def add_user_interaction(self, user_id: str, interaction: Interaction) -> None:  # noqa: ARG002
        embedding = self._get_embedding(
            f"{interaction.content}\n{interaction.user_action_description}"
        )
        interaction.embedding = embedding
        self._insert_interaction(interaction)

    def _insert_interaction(self, interaction: Interaction) -> int:
        created_at_iso = _epoch_to_iso(interaction.created_at)
        with self._lock:
            if interaction.interaction_id:
                self.conn.execute(
                    """INSERT OR REPLACE INTO interactions
                       (interaction_id, user_id, content, request_id, created_at,
                        role, user_action, user_action_description,
                        interacted_image_url, shadow_content, expert_content,
                        tools_used, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        interaction.interaction_id,
                        interaction.user_id,
                        interaction.content,
                        interaction.request_id,
                        created_at_iso,
                        interaction.role,
                        interaction.user_action.value,
                        interaction.user_action_description,
                        interaction.interacted_image_url,
                        interaction.shadow_content,
                        interaction.expert_content,
                        _json_dumps([t.model_dump() for t in interaction.tools_used]),
                        _json_dumps(interaction.embedding),
                    ),
                )
                iid = interaction.interaction_id
            else:
                cur = self.conn.execute(
                    """INSERT INTO interactions
                       (user_id, content, request_id, created_at,
                        role, user_action, user_action_description,
                        interacted_image_url, shadow_content, expert_content,
                        tools_used, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        interaction.user_id,
                        interaction.content,
                        interaction.request_id,
                        created_at_iso,
                        interaction.role,
                        interaction.user_action.value,
                        interaction.user_action_description,
                        interaction.interacted_image_url,
                        interaction.shadow_content,
                        interaction.expert_content,
                        _json_dumps([t.model_dump() for t in interaction.tools_used]),
                        _json_dumps(interaction.embedding),
                    ),
                )
                iid = cur.lastrowid or 0
                interaction.interaction_id = iid
            self.conn.commit()
        # Update FTS and vec
        self._fts_upsert(
            "interactions_fts",
            iid,
            content=interaction.content,
            user_action_description=interaction.user_action_description,
        )
        if interaction.embedding:
            self._vec_upsert("interactions_vec", iid, interaction.embedding)
        return iid

    @SQLiteStorageBase.handle_exceptions
    def add_user_interactions_bulk(
        self,
        user_id: str,  # noqa: ARG002
        interactions: list[Interaction],
    ) -> None:
        if not interactions:
            return
        texts = [
            "\n".join([i.content or "", i.user_action_description or ""])
            for i in interactions
        ]
        embeddings = self.llm_client.get_embeddings(
            texts, self.embedding_model_name, self.embedding_dimensions
        )
        for interaction, embedding in zip(interactions, embeddings, strict=False):
            interaction.embedding = embedding
            self._insert_interaction(interaction)

    @SQLiteStorageBase.handle_exceptions
    def delete_user_interaction(self, request: DeleteUserInteractionRequest) -> None:
        self._fts_delete("interactions_fts", request.interaction_id)
        self._execute(
            "DELETE FROM interactions WHERE user_id = ? AND interaction_id = ?",
            (request.user_id, request.interaction_id),
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_all_interactions_for_user(self, user_id: str) -> None:
        # Delete FTS entries for this user's interactions
        ids = [
            r["interaction_id"]
            for r in self._fetchall(
                "SELECT interaction_id FROM interactions WHERE user_id = ?", (user_id,)
            )
        ]
        if ids:
            placeholders = ",".join("?" for _ in ids)
            with self._lock:
                self.conn.execute(
                    f"DELETE FROM interactions_fts WHERE rowid IN ({placeholders})", ids
                )
                self.conn.commit()
        self._execute("DELETE FROM interactions WHERE user_id = ?", (user_id,))

    @SQLiteStorageBase.handle_exceptions
    def delete_all_interactions(self) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM interactions_fts")
            self.conn.execute("DELETE FROM interactions")
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def count_all_interactions(self) -> int:
        row = self._fetchone("SELECT COUNT(*) as cnt FROM interactions")
        return row["cnt"] if row else 0

    @SQLiteStorageBase.handle_exceptions
    def delete_oldest_interactions(self, count: int) -> int:
        if count <= 0:
            return 0
        rows = self._fetchall(
            "SELECT interaction_id FROM interactions ORDER BY created_at ASC LIMIT ?",
            (count,),
        )
        if not rows:
            return 0
        ids = [r["interaction_id"] for r in rows]
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            self.conn.execute(
                f"DELETE FROM interactions_fts WHERE rowid IN ({placeholders})", ids
            )
            self.conn.execute(
                f"DELETE FROM interactions WHERE interaction_id IN ({placeholders})",
                ids,
            )
            self.conn.commit()
        return len(ids)

    # ------------------------------------------------------------------
    # Search — Interactions & Profiles
    # ------------------------------------------------------------------

    @SQLiteStorageBase.handle_exceptions
    def search_interaction(
        self,
        search_interaction_request: SearchInteractionRequest,
        query_embedding: list[float] | None = None,
    ) -> list[Interaction]:
        req = search_interaction_request
        has_query = bool(req.query)
        match_count = req.most_recent_k or 10
        mode = _effective_search_mode(req.search_mode, query_embedding)

        conditions: list[str] = ["i.user_id = ?"]
        params: list[str | int | float] = [req.user_id]

        if req.request_id:
            conditions.append("i.request_id = ?")
            params.append(req.request_id)
        if req.start_time:
            conditions.append("i.created_at >= ?")
            params.append(req.start_time.timestamp())
        if req.end_time:
            conditions.append("i.created_at <= ?")
            params.append(req.end_time.timestamp())

        where_clause = " AND ".join(conditions)
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Vector-only: rank by embedding similarity
        if (
            mode in (SearchMode.VECTOR, SearchMode.HYBRID)
            and query_embedding
            and not has_query
        ):
            vector_limit = match_count * 10
            sql = f"""SELECT i.* FROM interactions i
                      WHERE {where_clause}
                      ORDER BY i.created_at DESC
                      LIMIT ?"""
            rows = self._fetchall(sql, (*params, vector_limit))
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        elif has_query:
            # FTS search (with optional HYBRID re-ranking)
            fts_query = _sanitize_fts_query(req.query)  # type: ignore[arg-type]
            fts_conditions = ["interactions_fts MATCH ?", *conditions]
            fts_where = " AND ".join(fts_conditions)
            fts_params: list[str | int | float] = [fts_query, *params, overfetch]
            sql = f"""SELECT i.* FROM interactions i
                      JOIN interactions_fts f ON i.interaction_id = f.rowid
                      WHERE {fts_where}
                      ORDER BY bm25(interactions_fts, 1.0, 2.0)
                      LIMIT ?"""
            fts_rows = self._fetchall(sql, tuple(fts_params))

            if mode == SearchMode.HYBRID and query_embedding:
                vec_limit = match_count * 10
                vec_sql = f"""SELECT i.* FROM interactions i
                              WHERE {where_clause}
                              ORDER BY i.created_at DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, (*params, vec_limit))
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "interaction_id",
                    match_count,
                )
            else:
                rows = fts_rows[:match_count]
        else:
            if req.most_recent_k:
                # No query — just fetch most recent interactions by time
                sql = f"""SELECT i.* FROM interactions i
                          WHERE {where_clause}
                          ORDER BY i.created_at DESC
                          LIMIT ?"""
                rows = self._fetchall(sql, (*params, req.most_recent_k))
                return [_row_to_interaction(r) for r in reversed(rows)]
            return []

        interactions = [_row_to_interaction(r) for r in rows]
        if req.most_recent_k:
            sorted_ints = sorted(interactions, key=lambda x: x.created_at, reverse=True)
            return list(reversed(sorted_ints[: req.most_recent_k]))
        return interactions

    @SQLiteStorageBase.handle_exceptions
    def search_user_profile(  # noqa: C901
        self,
        search_user_profile_request: SearchUserProfileRequest,
        status_filter: list[Status | None] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]

        req = search_user_profile_request
        match_count = req.top_k or 10
        current_ts = _epoch_now()
        has_query = bool(req.query)
        mode = _effective_search_mode(req.search_mode, query_embedding)
        has_embedding = query_embedding is not None
        logger.info(
            "Profile search: requested_mode=%s, effective_mode=%s, has_query=%s, has_embedding=%s, user_id=%s",
            req.search_mode,
            mode,
            has_query,
            has_embedding,
            req.user_id,
        )

        conditions: list[str] = ["p.expiration_timestamp >= ?"]
        params: list[object] = [current_ts]

        if req.user_id:
            conditions.append("p.user_id = ?")
            params.append(req.user_id)
        if req.start_time:
            conditions.append("p.last_modified_timestamp >= ?")
            params.append(int(req.start_time.timestamp()))
        if req.end_time:
            conditions.append("p.last_modified_timestamp <= ?")
            params.append(int(req.end_time.timestamp()))
        if req.source:
            conditions.append("LOWER(p.source) = LOWER(?)")
            params.append(req.source)
        if status_filter is not None:
            frag, sparams = _build_status_sql(status_filter)
            conditions.append(frag)
            params.extend(sparams)

        where_clause = " AND ".join(conditions)
        overfetch = match_count * 5 if mode != SearchMode.FTS else match_count

        # Pure vector search: fetch all candidates, rank by cosine similarity
        if mode == SearchMode.VECTOR and query_embedding:
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC"""
            rows = self._fetchall(sql, tuple(params))
            logger.info(
                "VECTOR search: %d candidates fetched, ranking by embedding", len(rows)
            )
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        elif has_query:
            fts_query = _sanitize_fts_query(req.query)  # type: ignore[arg-type]
            sql = f"""SELECT p.* FROM profiles p
                      JOIN profiles_fts f ON p.profile_id = f.profile_id
                      WHERE profiles_fts MATCH ?
                      AND {where_clause}
                      ORDER BY bm25(profiles_fts, 0.0, 1.0)
                      LIMIT ?"""
            params_list: list[object] = [fts_query, *params, overfetch]
            fts_rows = self._fetchall(sql, tuple(params_list))
            logger.info("FTS search: %d results from BM25", len(fts_rows))

            if mode == SearchMode.HYBRID and query_embedding:
                logger.info("HYBRID merging FTS + vector results via RRF")
                vec_limit = match_count * 10
                vec_sql = f"""SELECT p.* FROM profiles p
                              WHERE {where_clause}
                              ORDER BY p.last_modified_timestamp DESC
                              LIMIT ?"""
                vec_candidates = self._fetchall(vec_sql, (*params, vec_limit))
                vec_rows = _vector_rank_rows(vec_candidates, query_embedding, overfetch)
                rows = _true_rrf_merge(
                    fts_rows,
                    vec_rows,
                    "profile_id",
                    match_count,
                )
            else:
                rows = fts_rows
        elif query_embedding:
            # HYBRID without query text: rank by embedding only
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC"""
            rows = self._fetchall(sql, tuple(params))
            logger.info(
                "HYBRID (no query text) search: %d candidates, ranking by embedding",
                len(rows),
            )
            rows = _vector_rank_rows(rows, query_embedding, match_count)
        else:
            if req.generated_from_request_id:
                conditions.append("p.generated_from_request_id = ?")
                params.append(req.generated_from_request_id)
                where_clause = " AND ".join(conditions)
            sql = f"""SELECT p.* FROM profiles p
                      WHERE {where_clause}
                      ORDER BY p.last_modified_timestamp DESC
                      LIMIT ?"""
            params_list = [*params, overfetch]
            rows = self._fetchall(sql, tuple(params_list))

        profiles = [_row_to_profile(r) for r in rows]
        logger.info("Profile search: %d profiles before post-filtering", len(profiles))

        # Apply filters that can't easily go into SQL
        filtered: list[UserProfile] = []
        for profile in profiles:
            if req.custom_feature and (
                req.custom_feature.lower() not in str(profile.custom_features).lower()
            ):
                continue
            if req.extractor_name and (
                not profile.extractor_names
                or req.extractor_name not in profile.extractor_names
            ):
                continue
            filtered.append(profile)
            if len(filtered) >= match_count:
                break
        return filtered
