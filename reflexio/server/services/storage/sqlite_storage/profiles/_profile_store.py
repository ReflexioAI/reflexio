"""Profile store CRUD methods for SQLite storage.

Extracted verbatim from ``_profiles.py`` (the ProfileStore bucket). Interaction
methods live in ``profiles._interaction_store`` (``InteractionStoreMixin``);
search methods live in ``profiles._search`` (``ProfileSearchMixin``).
"""

import logging
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from reflexio.models.api_schema.service_schemas import (
    DeleteUserProfileRequest,
    Status,
    UserProfile,
)

from .._base import (
    _PROFILE_TOMBSTONE_STATUS_VALUES,
    SQLiteStorageBase,
    _build_status_sql,
    _epoch_now,
    _iso_now,
    _json_dumps,
    _row_to_profile,
)
from .._lineage import _GC_ELIGIBLE_STATUSES, _append_event_stmt
from .._profiles import _build_tags_sql

logger = logging.getLogger(__name__)


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


def _escape_like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


class ProfileStoreMixin:
    """Mixin providing profile-store CRUD for SQLite storage."""

    # Type hints for instance attributes/methods provided by SQLiteStorageBase via MRO
    _lock: Any
    conn: sqlite3.Connection
    org_id: str
    _fetchone: Any
    _fetchall: Any
    _get_embedding: Any
    _should_expand_documents: Any
    _expand_document: Any
    _fts_upsert_profile: Any
    _vec_upsert: Any
    _delete_in_chunks: Any
    _has_sqlite_vec: bool
    _subject_ref_for_user_id: Any
    _assert_subject_writable_locked: Any

    def _subject_ref_from_profile_row(self, row: sqlite3.Row) -> str:
        subject_ref = row["governance_subject_ref"]
        return (
            str(subject_ref)
            if subject_ref
            else self._subject_ref_for_user_id(row["user_id"])
        )

    def _assert_profile_writable_locked(
        self,
        profile_id: str,
        *,
        user_id: str | None = None,
    ) -> sqlite3.Row | None:
        if user_id is None:
            row = self.conn.execute(
                "SELECT user_id, governance_subject_ref FROM profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT user_id, governance_subject_ref FROM profiles WHERE profile_id = ? AND user_id = ?",
                (profile_id, user_id),
            ).fetchone()
        if row is None:
            return None
        self._assert_subject_writable_locked(self._subject_ref_from_profile_row(row))
        return row

    @SQLiteStorageBase.handle_exceptions
    def get_all_profiles(
        self,
        limit: int = 100,
        status_filter: list[Status | None] | None = None,
        user_id: str | None = None,
        profile_id: str | None = None,
        query: str | None = None,
        source: str | None = None,
        profile_time_to_live: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]
        frag, params = _build_status_sql(status_filter)
        conditions = [frag]
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if profile_id:
            conditions.append("LOWER(profile_id) = LOWER(?)")
            params.append(profile_id)
        if query:
            like = f"%{_escape_like_pattern(query.lower())}%"
            conditions.append(
                "(LOWER(content) LIKE ? ESCAPE '\\' OR LOWER(profile_id) LIKE ? ESCAPE '\\' OR LOWER(user_id) LIKE ? ESCAPE '\\')"
            )
            params.extend([like, like, like])
        if source is not None:
            conditions.append("source = ?")
            params.append(source)
        if profile_time_to_live:
            conditions.append("profile_time_to_live = ?")
            params.append(profile_time_to_live)
        if start_time is not None:
            conditions.append("last_modified_timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("last_modified_timestamp <= ?")
            params.append(end_time)
        sql = (
            f"SELECT * FROM profiles WHERE {' AND '.join(conditions)} "
            "ORDER BY last_modified_timestamp DESC LIMIT ?"
        )
        params.append(limit)
        return [_row_to_profile(r) for r in self._fetchall(sql, params)]

    @SQLiteStorageBase.handle_exceptions
    def get_user_profile(
        self,
        user_id: str,
        status_filter: list[Status | None] | None = None,
        tags: list[str] | None = None,
        profile_id: str | None = None,
        query: str | None = None,
        source: str | None = None,
        profile_time_to_live: str | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        include_expired: bool = False,
    ) -> list[UserProfile]:
        if status_filter is None:
            status_filter = [None]
        frag, params = _build_status_sql(status_filter)
        conditions: list[str] = ["user_id = ?"]
        all_params: list[Any] = [user_id]
        if not include_expired:
            conditions.append("expiration_timestamp >= ?")
            all_params.append(_epoch_now())
        conditions.append(frag)
        all_params.extend(params)
        if profile_id:
            conditions.append("LOWER(profile_id) = LOWER(?)")
            all_params.append(profile_id)
        if query:
            like = f"%{_escape_like_pattern(query.lower())}%"
            conditions.append(
                "(LOWER(content) LIKE ? ESCAPE '\\' OR LOWER(profile_id) LIKE ? ESCAPE '\\' OR LOWER(user_id) LIKE ? ESCAPE '\\')"
            )
            all_params.extend([like, like, like])
        if source is not None:
            conditions.append("source = ?")
            all_params.append(source)
        if profile_time_to_live:
            conditions.append("profile_time_to_live = ?")
            all_params.append(profile_time_to_live)
        if start_time is not None:
            conditions.append("last_modified_timestamp >= ?")
            all_params.append(start_time)
        if end_time is not None:
            conditions.append("last_modified_timestamp <= ?")
            all_params.append(end_time)
        tag_frag, tag_params = _build_tags_sql("profiles", tags)
        if tag_frag:
            conditions.append(tag_frag)
            all_params.extend(tag_params)
        sql = f"SELECT * FROM profiles WHERE {' AND '.join(conditions)}"
        return [_row_to_profile(r) for r in self._fetchall(sql, all_params)]

    @SQLiteStorageBase.handle_exceptions
    def add_user_profile(self, user_id: str, user_profiles: list[UserProfile]) -> None:  # noqa: ARG002
        for profile in user_profiles:
            subject_ref = self._subject_ref_for_user_id(profile.user_id)
            with self._lock:
                self._assert_subject_writable_locked(subject_ref)
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
            with self._lock:
                try:
                    self.conn.execute("BEGIN IMMEDIATE")
                    self._assert_subject_writable_locked(subject_ref)
                    self.conn.execute(
                        """INSERT OR REPLACE INTO profiles
                           (profile_id, user_id, content, last_modified_timestamp,
                            generated_from_request_id, profile_time_to_live,
                            expiration_timestamp, custom_features, embedding, source,
                            status, extractor_names, expanded_terms,
                            source_span, notes, reader_angle, tags, source_interaction_ids, created_at,
                            merged_into, superseded_by, governance_subject_ref)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                            _json_dumps(profile.source_interaction_ids),
                            _iso_now(),
                            profile.merged_into,
                            profile.superseded_by,
                            subject_ref,
                        ),
                    )
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise
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
        with self._lock:
            row = self.conn.execute(
                "SELECT user_id, governance_subject_ref FROM profiles WHERE user_id = ? AND profile_id = ? AND expiration_timestamp >= ?",
                (user_id, profile_id, current_ts),
            ).fetchone()
            if not row:
                logger.warning("User profile not found for user id: %s", user_id)
                return
            self._assert_subject_writable_locked(
                self._subject_ref_from_profile_row(row)
            )
        embedding = self._get_embedding(
            "\n".join([new_profile.content, str(new_profile.custom_features)])
        )
        new_profile.embedding = embedding
        with self._lock:
            if (
                self._assert_profile_writable_locked(profile_id, user_id=user_id)
                is None
            ):
                logger.warning("User profile not found for user id: %s", user_id)
                return
            cur = self.conn.execute(
                """UPDATE profiles SET content=?, last_modified_timestamp=?,
                   generated_from_request_id=?, profile_time_to_live=?,
                   expiration_timestamp=?, custom_features=?, embedding=?,
                   source=?, status=?, extractor_names=?, expanded_terms=?,
                   source_span=?, notes=?, reader_angle=?, tags=?, source_interaction_ids=?
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
                    _json_dumps(new_profile.source_interaction_ids),
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
        with self._lock:
            if (
                self._assert_profile_writable_locked(profile_id, user_id=user_id)
                is None
            ):
                return
            self.conn.execute(
                "UPDATE profiles SET tags=? WHERE user_id=? AND profile_id=?",
                (_json_dumps(tags), user_id, profile_id),
            )
            self.conn.commit()

    @SQLiteStorageBase.handle_exceptions
    def delete_user_profile(self, request: DeleteUserProfileRequest) -> None:
        # Atomic: fts + vec + row + lineage in ONE lock/commit to prevent rowid reuse
        # race. profiles uses implicit (reusable) rowid keyed by TEXT PK — a cleanup
        # running after commit could race with a concurrent INSERT reusing the freed
        # rowid and delete the NEW profile's vec row. (#196)
        with self._lock:
            rowid_row = self.conn.execute(
                "SELECT rowid FROM profiles WHERE user_id = ? AND profile_id = ?",
                (request.user_id, request.profile_id),
            ).fetchone()
            if rowid_row is None:
                return
            self.conn.execute(
                "DELETE FROM profiles_fts WHERE profile_id = ?",
                (request.profile_id,),
            )
            if self._has_sqlite_vec and rowid_row:
                self.conn.execute(
                    "DELETE FROM profiles_vec WHERE rowid = ?",
                    (rowid_row["rowid"],),
                )
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
        # Atomic: fts + vec + row + lineage in ONE lock/commit — rowid reuse race
        # prevention (see delete_user_profile comment, #196).
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid, profile_id FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchall()
            if not rows:
                return
            pids = [r["profile_id"] for r in rows]
            rowids = [r["rowid"] for r in rows]
            self._delete_in_chunks("profiles_fts", "profile_id", pids)
            if self._has_sqlite_vec and rowids:
                self._delete_in_chunks("profiles_vec", "rowid", rowids)
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
        # Also wipe profiles_vec (full-wipe variant of the rowid-race fix, #196).
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
            if self._has_sqlite_vec:
                self.conn.execute("DELETE FROM profiles_vec")
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

        # Set retired_at = now when transitioning to a GC-eligible status; clear to NULL otherwise.
        retired_at_val = now_ts if new_val in _GC_ELIGIBLE_STATUSES else None

        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = list(
                self.conn.execute(
                    f"SELECT profile_id, user_id, governance_subject_ref FROM profiles WHERE {where}",
                    select_params + extra_params,
                ).fetchall()
            )
            for row in affected:
                self._assert_subject_writable_locked(
                    self._subject_ref_from_profile_row(row)
                )
            cur = self.conn.execute(
                f"UPDATE profiles SET status = ?, last_modified_timestamp = ?, retired_at = ? WHERE {where}",
                [new_val, now_ts, retired_at_val] + select_params + extra_params,
            )
            from_val = old_status.value if old_status else None
            to_val = new_status.value if new_status else None
            for row in affected:
                pid = row["profile_id"]
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
    def expire_active_profiles(self, *, now: int, limit: int = 1000) -> int:
        """Tombstone active profiles whose TTL has elapsed.

        Args:
            now: Current epoch timestamp (seconds). Profiles with
                ``expiration_timestamp < now`` are tombstoned.
            limit: Maximum number of profiles to tombstone in one call (default 1000).

        Returns:
            int: Number of profiles tombstoned.
        """
        if limit <= 0:
            return 0
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            affected = list(
                self.conn.execute(
                    "SELECT profile_id, user_id, governance_subject_ref FROM profiles "
                    "WHERE status IS NULL AND expiration_timestamp < ? "
                    "ORDER BY expiration_timestamp ASC LIMIT ?",
                    (now, limit),
                ).fetchall()
            )
            if not affected:
                return 0
            for row in affected:
                self._assert_subject_writable_locked(
                    self._subject_ref_from_profile_row(row)
                )
            ids = [r["profile_id"] for r in affected]
            ph = ",".join("?" * len(ids))
            cur = self.conn.execute(
                f"UPDATE profiles SET status = ?, retired_at = ?, last_modified_timestamp = ? "  # noqa: S608
                f"WHERE profile_id IN ({ph}) AND status IS NULL",
                [Status.EXPIRED.value, now, now, *ids],
            )
            for row in affected:
                _append_event_stmt(
                    self.conn,
                    org_id=self.org_id,
                    entity_type="profile",
                    entity_id=str(row["profile_id"]),
                    op="status_change",
                    prov="wasInvalidatedBy",
                    source_ids=[],
                    actor="system",
                    request_id=batch_request_id,
                    reason="ttl-expired",
                    from_status=None,
                    to_status=Status.EXPIRED.value,
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
            include_tombstones: When False (default), MERGED/SUPERSEDED/EXPIRED profiles
                return None. Set to True for lineage resolution (resolve_current).

        Returns:
            The UserProfile if found and not filtered, otherwise None.
        """
        sql = "SELECT * FROM profiles WHERE profile_id = ?"
        if not include_tombstones:
            ph = ",".join("?" * len(_PROFILE_TOMBSTONE_STATUS_VALUES))
            sql += f" AND (status IS NULL OR status NOT IN ({ph}))"
            row = self._fetchone(sql, (profile_id, *_PROFILE_TOMBSTONE_STATUS_VALUES))
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

    def get_all_generated_profiles(self) -> list[UserProfile]:
        """All profiles (any status) with a non-empty generated_from_request_id."""
        rows = self._fetchall(
            "SELECT * FROM profiles "
            "WHERE generated_from_request_id IS NOT NULL "
            "AND generated_from_request_id <> ''",
        )
        return [_row_to_profile(r) for r in rows]

    @SQLiteStorageBase.handle_exceptions
    def archive_profile_by_id(self, user_id: str, profile_id: str) -> bool:
        with self._lock:
            if (
                self._assert_profile_writable_locked(profile_id, user_id=user_id)
                is None
            ):
                return False
            now_ts = _epoch_now()
            cur = self.conn.execute(
                "UPDATE profiles SET status = ?, last_modified_timestamp = ?, retired_at = ? "
                "WHERE profile_id = ? AND user_id = ? AND status IS NULL",
                (Status.ARCHIVED.value, now_ts, now_ts, profile_id, user_id),
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
    ) -> list[str]:
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
            list[str]: The profile ids actually superseded by this call, in input order
                (already-superseded or absent ids are omitted).
        """
        if not profile_ids:
            return []
        if not request_id:
            raise ValueError("request_id must be non-empty for supersede")
        now_ts = _epoch_now()
        # Eligibility: CURRENT (NULL) or PENDING — the two live statuses dedup can target.
        eligible = (None, Status.PENDING.value)
        committed_ids: list[str] = []
        with self._lock:
            for pid in profile_ids:
                # Read current status for from_status derivation (user_id scoped)
                row = self.conn.execute(
                    "SELECT status, user_id, governance_subject_ref FROM profiles WHERE profile_id = ? AND user_id = ?",
                    (pid, user_id),
                ).fetchone()
                if row is None:
                    continue
                self._assert_subject_writable_locked(
                    self._subject_ref_from_profile_row(row)
                )
                old_status_val = (
                    row[0] if isinstance(row, (tuple, list)) else row["status"]
                )
                if old_status_val not in eligible:
                    continue
                cur = self.conn.execute(
                    "UPDATE profiles SET status = ?, last_modified_timestamp = ?, retired_at = ? "
                    "WHERE profile_id = ? AND user_id = ? "
                    "AND (status IS NULL OR status = ?)",
                    (
                        Status.SUPERSEDED.value,
                        now_ts,
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
                    committed_ids.append(pid)
            self.conn.commit()
        return committed_ids

    @SQLiteStorageBase.handle_exceptions
    def delete_all_profiles_by_status(self, status: Status) -> int:
        # Atomic: fts + vec + row + lineage in ONE lock/commit — rowid reuse race
        # prevention (see delete_user_profile comment, #196).
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            rows = self.conn.execute(
                "SELECT rowid, profile_id FROM profiles WHERE status = ?",
                (status.value,),
            ).fetchall()
            if not rows:
                return 0
            pids = [r["profile_id"] for r in rows]
            rowids = [r["rowid"] for r in rows]
            self._delete_in_chunks("profiles_fts", "profile_id", pids)
            if self._has_sqlite_vec and rowids:
                self._delete_in_chunks("profiles_vec", "rowid", rowids)
            ph = ",".join("?" for _ in pids)
            cur = self.conn.execute(
                f"DELETE FROM profiles WHERE profile_id IN ({ph})",
                pids,  # noqa: S608
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
        # Atomic: fts + vec + row + lineage in ONE lock/commit — rowid reuse race
        # prevention (see delete_user_profile comment, #196).
        ph = ",".join("?" for _ in profile_ids)
        batch_request_id = uuid.uuid4().hex
        with self._lock:
            pre_rows = self.conn.execute(
                f"SELECT rowid, profile_id FROM profiles WHERE profile_id IN ({ph})",
                profile_ids,
            ).fetchall()
            if not pre_rows:
                return 0
            existing = [r["profile_id"] for r in pre_rows]
            rowids = [r["rowid"] for r in pre_rows]
            self._delete_in_chunks("profiles_fts", "profile_id", existing)
            if self._has_sqlite_vec and rowids:
                self._delete_in_chunks("profiles_vec", "rowid", rowids)
            cur = self.conn.execute(
                f"DELETE FROM profiles WHERE profile_id IN ({ph})",
                profile_ids,  # noqa: S608
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
