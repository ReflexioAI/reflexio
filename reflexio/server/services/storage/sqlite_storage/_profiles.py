"""Profile and interaction CRUD + search mixins for SQLite storage."""

import logging
import sqlite3
import uuid
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
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
)

from ._base import (
    _TOMBSTONE_STATUS_VALUES,
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
from ._lineage import _append_event_stmt


def _emit_hard_delete_profile(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_id: str,
    request_id: str,
    actor: str = "api",
) -> None:
    """Emit a single hard_delete lineage event for a profile entity."""
    _append_event_stmt(
        conn,
        org_id=org_id,
        entity_type="profile",
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


class ProfileMixin:
    """Mixin providing profile and interaction CRUD + search."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
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
    _delete_profile_search_rows: Any
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
        tags: list[str] | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]
        current_ts = _epoch_now()
        frag, params = _build_status_sql(status_filter)
        conditions = ["user_id = ?", "expiration_timestamp >= ?", frag]
        all_params: list[Any] = [user_id, current_ts, *params]
        tag_frag, tag_params = _build_tags_sql("profiles", tags)
        if tag_frag:
            conditions.append(tag_frag)
            all_params.extend(tag_params)
        sql = f"SELECT * FROM profiles WHERE {' AND '.join(conditions)}"
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
                    status, extractor_names, expanded_terms,
                    source_span, notes, reader_angle, tags, created_at,
                    merged_into, superseded_by)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    profile.source_span,
                    profile.notes,
                    profile.reader_angle,
                    _json_dumps(profile.tags),
                    _iso_now(),
                    profile.merged_into,
                    profile.superseded_by,
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
        """Replace a profile's content in-place and emit a revise lineage event.

        Each call generates a fresh request_id so every edit is a distinct audit
        event (not collapsed by the idempotency key).  The UPDATE, lineage event,
        FTS sync, and vec sync are all executed inside a single lock acquisition;
        self._lock is an RLock so the inner _fts_upsert_profile/_vec_upsert calls
        that re-acquire it are safe.
        """
        current_ts = _epoch_now()
        row = self._fetchone(
            "SELECT profile_id FROM profiles WHERE user_id = ? AND profile_id = ? AND expiration_timestamp >= ?",
            (user_id, profile_id, current_ts),
        )
        if not row:
            logger.warning("User profile not found for user id: %s", user_id)
            return
        embedding = self._get_embedding(
            "\n".join([new_profile.content, str(new_profile.custom_features)])
        )
        new_profile.embedding = embedding
        with self._lock:
            cur = self.conn.execute(
                """UPDATE profiles SET content=?, last_modified_timestamp=?,
                   generated_from_request_id=?, profile_time_to_live=?,
                   expiration_timestamp=?, custom_features=?, embedding=?,
                   source=?, status=?, extractor_names=?, expanded_terms=?,
                   source_span=?, notes=?, reader_angle=?, tags=?
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
                    new_profile.source_span,
                    new_profile.notes,
                    new_profile.reader_angle,
                    _json_dumps(new_profile.tags),
                    profile_id,
                ),
            )
            if cur.rowcount > 0:
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="profile",
                    entity_id=str(profile_id),
                    op="revise",
                    prov="wasRevisionOf",
                    source_ids=[],
                    actor="api",
                    request_id=uuid.uuid4().hex,
                    reason="in-place update",
                )
            self.conn.commit()
            fts_parts = [new_profile.content or ""]
            if new_profile.custom_features:
                fts_parts.extend(
                    str(v) for v in new_profile.custom_features.values() if v
                )
            if new_profile.expanded_terms:
                fts_parts.append(new_profile.expanded_terms)
            self._fts_upsert_profile(profile_id, " ".join(fts_parts))
            rowid_row = self._fetchone(
                "SELECT rowid FROM profiles WHERE profile_id = ?", (profile_id,)
            )
            if rowid_row and embedding:
                self._vec_upsert("profiles_vec", rowid_row["rowid"], embedding)

    @SQLiteStorageBase.handle_exceptions
    def update_user_profile_tags(
        self, user_id: str, profile_id: str, tags: list[str]
    ) -> None:
        self._execute(
            "UPDATE profiles SET tags=? WHERE user_id=? AND profile_id=?",
            (_json_dumps(tags), user_id, profile_id),
        )

    @SQLiteStorageBase.handle_exceptions
    def delete_user_profile(self, request: DeleteUserProfileRequest) -> None:
        with self._lock:
            rowid_row = self.conn.execute(
                "SELECT rowid FROM profiles WHERE user_id = ? AND profile_id = ?",
                (request.user_id, request.profile_id),
            ).fetchone()
            self._fts_delete_profile(request.profile_id)
            if rowid_row:
                self._vec_delete("profiles_vec", rowid_row["rowid"])
            cur = self.conn.execute(
                "DELETE FROM profiles WHERE user_id = ? AND profile_id = ?",
                (request.user_id, request.profile_id),
            )
            if cur.rowcount > 0:
                _emit_hard_delete_profile(
                    self.conn,
                    org_id=self.org_id,
                    entity_id=str(request.profile_id),
                    request_id=uuid.uuid4().hex,
                )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles_for_user(self, user_id: str) -> None:
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            pids = [
                r["profile_id"]
                for r in self.conn.execute(
                    "SELECT profile_id FROM profiles WHERE user_id = ?", (user_id,)
                ).fetchall()
            ]
            if not pids:
                return
            self._delete_profile_search_rows(pids)
            self.conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            for pid in pids:
                _emit_hard_delete_profile(
                    self.conn,
                    org_id=self.org_id,
                    entity_id=str(pid),
                    request_id=batch_request_id,
                )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles(self) -> None:
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            pids = [
                r["profile_id"]
                for r in self.conn.execute("SELECT profile_id FROM profiles").fetchall()
            ]
            for pid in pids:
                _emit_hard_delete_profile(
                    self.conn,
                    org_id=self.org_id,
                    entity_id=str(pid),
                    request_id=batch_request_id,
                )
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
        old_val_str = old_status.value if old_status else "None"
        new_val_str = new_status.value if new_status else "None"
        reason = f"{old_val_str}->{new_val_str}"

        if old_status is None or (
            hasattr(old_status, "value") and old_status.value is None
        ):
            where = "status IS NULL"
            select_params: list[Any] = []
        else:
            where = "status = ?"
            select_params = [old_status.value]

        extra_params: list[Any] = []
        if user_ids is not None:
            placeholders = ",".join("?" for _ in user_ids)
            where += f" AND user_id IN ({placeholders})"
            extra_params.extend(user_ids)

        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = [
                r["profile_id"]
                for r in self.conn.execute(
                    f"SELECT profile_id FROM profiles WHERE {where}",
                    select_params + extra_params,
                ).fetchall()
            ]
            cur = self.conn.execute(
                f"UPDATE profiles SET status = ?, last_modified_timestamp = ? WHERE {where}",
                [new_val, now_ts] + select_params + extra_params,
            )
            from_val = old_status.value if old_status else None
            to_val = new_status.value if new_status else None
            for pid in affected:
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="profile",
                    entity_id=str(pid),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=batch_request_id,
                    reason=reason,
                    from_status=from_val,
                    to_status=to_val,
                    status_namespace="lifecycle_status",
                )
            self.conn.commit()
        return cur.rowcount

    @SQLiteStorageBase.handle_exceptions
    def get_profiles_by_ids(
        self,
        user_id: str,
        profile_ids: list[str],
        status_filter: list[Status | None] | None = None,
    ) -> list[UserProfile]:
        if not profile_ids:
            return []
        if status_filter is None:
            status_filter = [None]
        current_ts = _epoch_now()
        frag, sparams = _build_status_sql(status_filter)
        ph = ",".join("?" for _ in profile_ids)
        sql = (
            f"SELECT * FROM profiles "
            f"WHERE user_id = ? AND profile_id IN ({ph}) "
            f"AND expiration_timestamp >= ? AND {frag}"
        )
        params: list[Any] = [user_id, *profile_ids, current_ts, *sparams]
        return [_row_to_profile(r) for r in self._fetchall(sql, params)]

    @SQLiteStorageBase.handle_exceptions
    def get_profile_by_id(
        self, profile_id: str, *, include_tombstones: bool = False
    ) -> UserProfile | None:
        """Fetch a single profile by primary key.

        Args:
            profile_id: The profile's primary key.
            include_tombstones: When False (default), MERGED/SUPERSEDED profiles
                return None. Set to True for lineage resolution (resolve_current).

        Returns:
            The UserProfile if found and not filtered, otherwise None.
        """
        sql = "SELECT * FROM profiles WHERE profile_id = ?"
        if not include_tombstones:
            sql += " AND (status IS NULL OR status NOT IN (?, ?))"
            row = self._fetchone(sql, (profile_id, *_TOMBSTONE_STATUS_VALUES))
        else:
            row = self._fetchone(sql, (profile_id,))
        return _row_to_profile(row) if row else None

    @SQLiteStorageBase.handle_exceptions
    def get_distinct_generated_from_request_ids(self) -> list[str]:
        """Return DISTINCT non-empty generated_from_request_id values, including tombstones.

        Returns:
            list[str]: Distinct non-empty ``generated_from_request_id`` values.
        """
        rows = self._fetchall(
            "SELECT DISTINCT generated_from_request_id FROM profiles"
            " WHERE generated_from_request_id IS NOT NULL"
            " AND generated_from_request_id != ''",
            (),
        )
        return [row[0] for row in rows]

    @SQLiteStorageBase.handle_exceptions
    def get_profiles_by_generated_from_request_id(
        self,
        request_id: str,
    ) -> list[UserProfile]:
        """Return all profiles for a generated_from_request_id, including tombstones.

        Args:
            request_id (str): The generated_from_request_id to filter on.

        Returns:
            list[UserProfile]: All matching profiles (any status).
        """
        rows = self._fetchall(
            "SELECT * FROM profiles WHERE generated_from_request_id = ?",
            (request_id,),
        )
        return [_row_to_profile(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def archive_profile_by_id(self, user_id: str, profile_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE profiles SET status = ?, last_modified_timestamp = ? "
                "WHERE profile_id = ? AND user_id = ? AND status IS NULL",
                (Status.ARCHIVED.value, _epoch_now(), profile_id, user_id),
            )
            if cur.rowcount > 0:
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="profile",
                    entity_id=str(profile_id),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="api",
                    request_id=uuid.uuid4().hex,
                    reason="None->archived",
                    from_status=None,
                    to_status="archived",
                    status_namespace="lifecycle_status",
                )
            self.conn.commit()
        return cur.rowcount > 0

    @SQLiteStorageBase.handle_exceptions
    def supersede_profiles_by_ids(
        self,
        user_id: str,
        profile_ids: list[str],
        request_id: str,
    ) -> int:
        """Soft-delete profiles by setting status to SUPERSEDED, emitting set-based lineage.

        For each matching id (user_id scoped, currently CURRENT), updates status to
        SUPERSEDED and emits one ``status_change`` event under the shared ``request_id``.
        Atomic: one ``conn.commit()`` at the end, guarded on rowcount per id.
        FTS/vec rows are NOT removed — reads exclude tombstones by status filter.

        Args:
            user_id (str): Owning user id.
            profile_ids (list[str]): Profile ids to supersede.
            request_id (str): Shared request id for all emitted lineage events.

        Returns:
            int: Number of profiles actually updated.
        """
        if not profile_ids:
            return 0
        now_ts = _epoch_now()
        # Eligibility: CURRENT (NULL) or PENDING — the two live statuses dedup can target.
        eligible = (None, Status.PENDING.value)
        updated = 0
        with self._lock:
            for pid in profile_ids:
                # Read current status for from_status derivation (user_id scoped)
                row = self.conn.execute(
                    "SELECT status FROM profiles WHERE profile_id = ? AND user_id = ?",
                    (pid, user_id),
                ).fetchone()
                if row is None:
                    continue
                old_status_val = (
                    row[0] if isinstance(row, (tuple, list)) else row["status"]
                )
                if old_status_val not in eligible:
                    continue
                cur = self.conn.execute(
                    "UPDATE profiles SET status = ?, last_modified_timestamp = ? "
                    "WHERE profile_id = ? AND user_id = ? "
                    "AND (status IS NULL OR status = ?)",
                    (
                        Status.SUPERSEDED.value,
                        now_ts,
                        pid,
                        user_id,
                        Status.PENDING.value,
                    ),
                )
                if cur.rowcount > 0:
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type="profile",
                        entity_id=str(pid),
                        op="status_change",
                        prov="wasInvalidatedBy",
                        source_ids=[],
                        actor="dedup",
                        request_id=request_id,
                        reason=f"{old_status_val}->superseded",
                        from_status=old_status_val,
                        to_status=Status.SUPERSEDED.value,
                        status_namespace="lifecycle_status",
                    )
                    updated += 1
            self.conn.commit()
        return updated

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles_by_status(self, status: Status) -> int:
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            pids = [
                r["profile_id"]
                for r in self.conn.execute(
                    "SELECT profile_id FROM profiles WHERE status = ?", (status.value,)
                ).fetchall()
            ]
            if not pids:
                return 0
            self._delete_profile_search_rows(pids)
            ph = ",".join("?" for _ in pids)
            cur = self.conn.execute(
                f"DELETE FROM profiles WHERE profile_id IN ({ph})", pids
            )
            for pid in pids:
                _emit_hard_delete_profile(
                    self.conn,
                    org_id=self.org_id,
                    entity_id=str(pid),
                    request_id=batch_request_id,
                )
            self.conn.commit()
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
    def delete_profiles_by_ids(
        self, profile_ids: list[str], *, emit_hard_delete: bool = True
    ) -> int:
        if not profile_ids:
            return 0
        ph = ",".join("?" for _ in profile_ids)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            existing = [
                r["profile_id"]
                for r in self.conn.execute(
                    f"SELECT profile_id FROM profiles WHERE profile_id IN ({ph})",
                    profile_ids,
                ).fetchall()
            ]
            if not existing:
                return 0
            self._delete_profile_search_rows(existing)
            cur = self.conn.execute(
                f"DELETE FROM profiles WHERE profile_id IN ({ph})", profile_ids
            )
            if emit_hard_delete:
                for pid in existing:
                    _emit_hard_delete_profile(
                        self.conn,
                        org_id=self.org_id,
                        entity_id=str(pid),
                        request_id=batch_request_id,
                        actor="system",
                    )
            self.conn.commit()
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
                        tools_used, citations, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        _json_dumps([c.model_dump() for c in interaction.citations]),
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
                        tools_used, citations, embedding)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                        _json_dumps([c.model_dump() for c in interaction.citations]),
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
        try:
            embeddings = self.llm_client.get_embeddings(
                texts, self.embedding_model_name, self.embedding_dimensions
            )
        except EmbeddingUnavailableError as exc:
            logger.warning(
                "Embedding unavailable for interaction bulk insert; "
                "continuing without vectors: %s",
                exc,
            )
            embeddings = [[] for _ in texts]
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
        mode = _effective_search_mode(req.search_mode, query_embedding, req.query)

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
        mode = _effective_search_mode(req.search_mode, query_embedding, req.query)
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
        tag_frag, tag_params = _build_tags_sql("p", req.tags)
        if tag_frag:
            conditions.append(tag_frag)
            params.extend(tag_params)

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
            filtered.append(profile)
            if len(filtered) >= match_count:
                break
        return filtered
