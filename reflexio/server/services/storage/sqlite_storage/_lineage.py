import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Literal

from reflexio.models.api_schema.domain.entities import LineageContext, LineageEvent
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.storage_base._lineage import LegalHold
from reflexio.server.tracing import capture_anomaly

from ._base import _epoch_now

EntityType = Literal["user_playbook", "agent_playbook", "profile"]

# GC-eligible statuses — rows with these statuses may be hard-deleted by TTL GC.
# Also used as the merge guard: a source that already carries any of these
# statuses is skipped (no re-tombstone, no clock reset).
_GC_ELIGIBLE_STATUSES: frozenset[str] = frozenset(
    {Status.MERGED.value, Status.SUPERSEDED.value, Status.ARCHIVED.value}
)

# Mapping from entity_type string to (table_name, primary_key_column).
_TABLE: dict[str, tuple[str, str]] = {
    "user_playbook": ("user_playbooks", "user_playbook_id"),
    "agent_playbook": ("agent_playbooks", "agent_playbook_id"),
    "profile": ("profiles", "profile_id"),
}

# Error message used by merge_records and supersede_record guards.
# Shared here so tests can reference this exact string without hardcoding.
_EMPTY_REQUEST_ID_MSG = "request_id must be non-empty"


def _resolve_table(entity_type: str) -> tuple[str, str]:
    """Map an entity_type to its (table, primary_key), raising on bad input."""
    table = _TABLE.get(entity_type)
    if table is None:
        raise ValueError(f"unknown entity_type: {entity_type!r}")
    return table


def _append_event_stmt(
    conn: sqlite3.Connection,
    *,
    org_id: str,
    entity_type: str,
    entity_id: str,
    op: str,
    prov: str,
    source_ids: list[str],
    actor: str,
    request_id: str,
    reason: str,
    created_at: int | None = None,
    from_status: str | None = None,
    to_status: str | None = None,
    status_namespace: str | None = None,
) -> sqlite3.Cursor:
    """Insert a lineage event row; no-ops on (org_id, entity_type, entity_id, op, request_id) duplicate.

    Returns the cursor so callers can inspect ``rowcount``/``lastrowid``.
    """
    return conn.execute(
        "INSERT OR IGNORE INTO lineage_event "
        "(org_id, entity_type, entity_id, op, prov_relation, source_ids, "
        "actor, request_id, reason, created_at, "
        "from_status, to_status, status_namespace) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            org_id,
            entity_type,
            entity_id,
            op,
            prov,
            json.dumps(source_ids),
            actor,
            request_id,
            reason,
            created_at if created_at is not None else int(time.time()),
            from_status,
            to_status,
            status_namespace,
        ),
    )


class SQLiteLineageMixin:
    """SQLite implementation of the append-only, content-free lineage event log."""

    # Type hints for instance attributes provided by SQLiteStorageBase via MRO.
    conn: sqlite3.Connection
    _lock: threading.RLock
    org_id: str

    def append_lineage_event(self, event: LineageEvent) -> int:
        """Append an event; idempotent on (org_id, entity_type, entity_id, op, request_id).

        Args:
            event (LineageEvent): The fully-formed event to persist. ``event_id``
                may be 0; the storage layer assigns a real id on insert. On a
                duplicate ``(org_id, entity_type, entity_id, op, request_id)`` the
                existing row is returned unchanged.

        Returns:
            int: The assigned or existing ``event_id``.
        """
        created = event.created_at or int(time.time())
        with self._lock:
            cur = _append_event_stmt(
                self.conn,
                org_id=event.org_id,
                entity_type=event.entity_type,
                entity_id=event.entity_id,
                op=event.op,
                prov=event.prov_relation,
                source_ids=event.source_ids,
                actor=event.actor,
                request_id=event.request_id,
                reason=event.reason,
                created_at=created,
                from_status=event.from_status,
                to_status=event.to_status,
                status_namespace=event.status_namespace,
            )
            if (
                cur.rowcount == 0
            ):  # duplicate (org_id, entity_type, entity_id, op, request_id)
                row = self.conn.execute(
                    "SELECT event_id FROM lineage_event WHERE org_id=? AND entity_type=? "
                    "AND entity_id=? AND op=? AND request_id=?",
                    (
                        event.org_id,
                        event.entity_type,
                        event.entity_id,
                        event.op,
                        event.request_id,
                    ),
                ).fetchone()
                eid = row[0] if row else None
                self.conn.commit()
                return int(eid) if eid is not None else 0
            last = cur.lastrowid
            self.conn.commit()
            return int(last) if last is not None else 0

    def get_lineage_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        org_id: str | None = None,
        request_id: str | None = None,
    ) -> list[LineageEvent]:
        """Retrieve lineage events, optionally filtered.

        Args:
            entity_type (str | None): Filter to events for this entity type.
            entity_id (str | None): Filter to events for this entity id.
            org_id (str | None): Filter to events for this org.
            request_id (str | None): Filter to events for this request id.

        Returns:
            list[LineageEvent]: Matching events ordered by ``event_id`` ascending.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for col, val in (
            ("entity_type", entity_type),
            ("entity_id", entity_id),
            ("org_id", org_id),
            ("request_id", request_id),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM lineage_event{where} ORDER BY event_id",  # noqa: S608
                params,
            ).fetchall()
        return [
            LineageEvent(
                event_id=r["event_id"],
                org_id=r["org_id"],
                entity_type=r["entity_type"],
                entity_id=r["entity_id"],
                op=r["op"],
                prov_relation=r["prov_relation"],
                source_ids=json.loads(r["source_ids"]),
                actor=r["actor"],
                request_id=r["request_id"],
                reason=r["reason"],
                created_at=r["created_at"],
                from_status=r["from_status"],
                to_status=r["to_status"],
                status_namespace=r["status_namespace"],
            )
            for r in rows
        ]

    def merge_records(
        self,
        *,
        entity_type: EntityType,
        survivor_id: str,
        source_ids: list[str],
        context: LineageContext,
    ) -> None:
        """Soft-delete each source into the survivor in one atomic transaction.

        Sets ``status=MERGED`` and ``merged_into=survivor_id`` on each source
        whose status is not already a tombstone. Appends a single ``merge``
        lineage event keyed on ``survivor_id``. Idempotent — re-running on
        already-tombstoned sources is a no-op.

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            survivor_id (str): The id of the record that survives the merge.
            source_ids (list[str]): Ids of records to tombstone as merged.
            context (LineageContext): Caller-supplied intent (actor, reason, etc.).

        Raises:
            ValueError: If ``entity_type`` is not a recognized entity type.
            ValueError: If ``context.request_id`` is empty or whitespace-only.
        """
        if not (context.request_id and context.request_id.strip()):
            raise ValueError(f"lineage merge: {_EMPTY_REQUEST_ID_MSG}")
        table, pk = _resolve_table(entity_type)
        now = _epoch_now()
        eligible_ph = ",".join("?" * len(_GC_ELIGIBLE_STATUSES))
        eligible_vals = list(_GC_ELIGIBLE_STATUSES)
        with self._lock:
            for sid in source_ids:
                if sid == survivor_id:
                    # Never tombstone the survivor itself, even if it is
                    # accidentally listed among the source ids.
                    continue
                # Skip sources that already carry any eligible/tombstone status
                # (MERGED, SUPERSEDED, or ARCHIVED) — avoids re-tombstoning an
                # already-archived source and resetting its retired_at clock.
                self.conn.execute(
                    f"UPDATE {table} SET status=?, merged_into=?, retired_at=? "  # noqa: S608
                    f"WHERE {pk}=? AND {pk}!=? "
                    f"AND (status IS NULL OR status NOT IN ({eligible_ph}))",
                    (
                        Status.MERGED.value,
                        survivor_id,
                        now,
                        sid,
                        survivor_id,
                        *eligible_vals,
                    ),
                )
            _append_event_stmt(
                self.conn,
                org_id=self.org_id,
                entity_type=entity_type,
                entity_id=survivor_id,
                op="merge",
                prov="wasDerivedFrom",
                source_ids=source_ids,
                actor=context.actor,
                request_id=context.request_id,
                reason=context.reason,
            )
            self.conn.commit()

    def supersede_record(
        self,
        *,
        entity_type: EntityType,
        incumbent_id: str,
        successor_id: str,
        context: LineageContext,
    ) -> bool:
        """Atomically replace the incumbent with the successor if incumbent is CURRENT.

        Sets ``status=SUPERSEDED`` and ``superseded_by=successor_id`` on the
        incumbent **only** when its ``status IS NULL`` (CURRENT). Appends a
        ``revise`` lineage event when the guard succeeds. Returns ``False``
        without mutating anything when the incumbent is not CURRENT.

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            incumbent_id (str): The id of the record to supersede.
            successor_id (str): The id of the record that replaces the incumbent.
            context (LineageContext): Caller-supplied intent (actor, reason, etc.).

        Returns:
            bool: ``True`` if the incumbent was CURRENT and was superseded;
                ``False`` if the incumbent was not CURRENT and no mutation occurred.

        Raises:
            ValueError: If ``entity_type`` is not a recognized entity type.
            ValueError: If ``context.request_id`` is empty or whitespace-only.
        """
        if not (context.request_id and context.request_id.strip()):
            raise ValueError(f"lineage supersede: {_EMPTY_REQUEST_ID_MSG}")
        table, pk = _resolve_table(entity_type)
        with self._lock:
            cur = self.conn.execute(
                f"UPDATE {table} SET status=?, superseded_by=?, retired_at=? "  # noqa: S608
                f"WHERE {pk}=? AND status IS NULL",
                (Status.SUPERSEDED.value, successor_id, _epoch_now(), incumbent_id),
            )
            if cur.rowcount == 0:
                self.conn.commit()
                return False
            _append_event_stmt(
                self.conn,
                org_id=self.org_id,
                entity_type=entity_type,
                entity_id=successor_id,
                op="revise",
                prov="wasRevisionOf",
                source_ids=[incumbent_id],
                actor=context.actor,
                request_id=context.request_id,
                reason=context.reason,
            )
            self.conn.commit()
            return True

    def place_hold(
        self,
        *,
        org_id: str,
        scope: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        user_id: str | None = None,
        matter_id: str,
        legal_basis: str,
        reason: str = "",
        placed_by: str = "",
        placed_at: int | None = None,
    ) -> int:
        """Place a legal hold; return the new hold id.

        Args:
            org_id (str): The organisation that owns the held entities.
            scope (str): One of ``'org'``, ``'user'``, or ``'entity'``.
            entity_type (str | None): For ``'entity'`` scope, the held entity's type.
            entity_id (str | None): For ``'entity'`` scope, the held entity's id.
            user_id (str | None): For ``'user'`` scope, the held user's id.
            matter_id (str): Caller-supplied identifier grouping related holds.
            legal_basis (str): One of ``'litigation_hold'``, ``'regulatory_order'``,
                or ``'legal_obligation'``.
            reason (str): Free-text justification.
            placed_by (str): Actor placing the hold.
            placed_at (int | None): Unix epoch; defaults to now when None.

        Returns:
            int: The new hold's id.
        """
        now = placed_at if placed_at is not None else _epoch_now()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO lineage_legal_hold "
                "(org_id, scope, entity_type, entity_id, user_id, matter_id, "
                "legal_basis, reason, placed_by, placed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    org_id,
                    scope,
                    entity_type,
                    entity_id,
                    user_id,
                    matter_id,
                    legal_basis,
                    reason,
                    placed_by,
                    now,
                ),
            )
            self.conn.commit()
            return int(cur.lastrowid or 0)

    def release_hold(
        self,
        *,
        org_id: str,
        hold_id: int | None = None,
        matter_id: str | None = None,
        scope: str | None = None,
        user_id: str | None = None,
        released_by: str = "",
        released_at: int | None = None,
    ) -> int:
        """Release holds by id OR by matter+scope+optional user; return count released.

        Args:
            org_id (str): The organisation that owns the holds.
            hold_id (int | None): Release this specific hold id.
            matter_id (str | None): Release active holds with this matter id.
            scope (str | None): Narrow matter-release to this scope.
            user_id (str | None): Narrow matter-release to this user.
            released_by (str): Actor releasing the holds.
            released_at (int | None): Unix epoch; defaults to now when None.

        Returns:
            int: The number of holds released.

        Raises:
            ValueError: If neither ``hold_id`` nor ``matter_id`` is supplied.
        """
        now = released_at if released_at is not None else _epoch_now()
        with self._lock:
            if hold_id is not None:
                cur = self.conn.execute(
                    "UPDATE lineage_legal_hold SET released_at = ?, released_by = ? "
                    "WHERE id = ? AND org_id = ? AND released_at IS NULL",
                    (now, released_by, hold_id, org_id),
                )
            elif matter_id is not None:
                params: list[Any] = [now, released_by, org_id, matter_id]
                sql_extra = ""
                if scope is not None:
                    sql_extra += " AND scope = ?"
                    params.append(scope)
                if user_id is not None:
                    sql_extra += " AND user_id = ?"
                    params.append(user_id)
                cur = self.conn.execute(
                    "UPDATE lineage_legal_hold SET released_at = ?, released_by = ? "
                    "WHERE org_id = ? AND matter_id = ? AND released_at IS NULL"
                    f"{sql_extra}",  # noqa: S608 — sql_extra is built from fixed clauses only
                    params,
                )
            else:
                raise ValueError("release_hold requires either hold_id or matter_id")
            self.conn.commit()
            return cur.rowcount

    def get_holds(self, org_id: str, *, active_only: bool = True) -> list[LegalHold]:
        """Return holds for ``org_id``.

        Args:
            org_id (str): The organisation whose holds to return.
            active_only (bool): When True (default), only return rows where
                ``released_at IS NULL``.

        Returns:
            list[LegalHold]: Matching holds.
        """
        sql_q = "SELECT * FROM lineage_legal_hold WHERE org_id = ?"
        if active_only:
            sql_q += " AND released_at IS NULL"
        rows = self.conn.execute(sql_q, (org_id,)).fetchall()
        return [
            LegalHold(
                id=r["id"],
                org_id=r["org_id"],
                scope=r["scope"],
                entity_type=r["entity_type"],
                entity_id=r["entity_id"],
                user_id=r["user_id"],
                matter_id=r["matter_id"],
                legal_basis=r["legal_basis"],
                reason=r["reason"],
                placed_by=r["placed_by"],
                placed_at=r["placed_at"],
                released_at=r["released_at"],
                released_by=r["released_by"],
            )
            for r in rows
        ]

    def _is_on_legal_hold(
        self,
        org_id: str,
        entity_type: str,
        entity_id: str,
    ) -> bool:
        """Return True if this entity is under an active legal hold.

        Checks three hold scopes in order:

        1. org-scope: any active hold for this org covers all entities.
        2. user-scope: any active hold for this entity's user. Skipped for
           ``agent_playbook``, which has no ``user_id`` column.
        3. entity-scope: any active hold for this specific (entity_type, entity_id).

        Called from ``gc_expired_tombstones`` inside ``with self._lock:`` — the
        queries here run on the same connection, so the check is atomic with the
        DELETE that follows.

        Args:
            org_id (str): The organisation that owns the entity.
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            entity_id (str): The entity's primary key as a string.

        Returns:
            bool: True if the entity is covered by an active hold.
        """
        # org-scope: covers everything in this org.
        row = self.conn.execute(
            "SELECT 1 FROM lineage_legal_hold "
            "WHERE org_id = ? AND scope = 'org' AND released_at IS NULL LIMIT 1",
            (org_id,),
        ).fetchone()
        if row:
            return True

        # user-scope: covers profile and user_playbook (agent_playbook has no user_id).
        if entity_type != "agent_playbook":
            table, pk = _resolve_table(entity_type)
            entity_row = self.conn.execute(
                f"SELECT user_id FROM {table} WHERE {pk} = ?",  # noqa: S608 — table/pk from fixed mapping
                (entity_id,),
            ).fetchone()
            if entity_row and entity_row["user_id"] is not None:
                row = self.conn.execute(
                    "SELECT 1 FROM lineage_legal_hold "
                    "WHERE org_id = ? AND scope = 'user' AND user_id = ? "
                    "AND released_at IS NULL LIMIT 1",
                    (org_id, entity_row["user_id"]),
                ).fetchone()
                if row:
                    return True

        # entity-scope: covers only this specific row.
        row = self.conn.execute(
            "SELECT 1 FROM lineage_legal_hold "
            "WHERE org_id = ? AND scope = 'entity' AND entity_type = ? "
            "AND entity_id = ? AND released_at IS NULL LIMIT 1",
            (org_id, entity_type, str(entity_id)),
        ).fetchone()
        return bool(row)

    def list_org_ids(self) -> list[str]:
        """Return the single org_id for this SQLite storage instance.

        SQLite storage is single-tenant: each instance is scoped to exactly one
        org. Returns ``[self.org_id]``.

        Returns:
            list[str]: A one-element list containing this instance's org_id.
        """
        return [self.org_id]

    def gc_expired_tombstones(
        self, *, entity_type: str, older_than_epoch: int, limit: int = 1000
    ) -> int:
        """Hard-delete tombstone rows whose retirement instant is older than the cutoff.

        Ages on the uniform INTEGER ``retired_at`` column set at every tombstone
        write-path (T1).  Rows with ``retired_at = NULL`` (pre-T1 tombstones) are
        never selected — they have no retirement clock and must be retained.

        Emits one ``hard_delete`` lineage event per deleted row, atomically, before
        the DELETE commits. Rows on legal hold are skipped without emitting an event.

        Args:
            entity_type (str): One of ``"user_playbook"``, ``"agent_playbook"``,
                or ``"profile"``.
            older_than_epoch (int): Unix timestamp cutoff (exclusive). Rows whose
                ``retired_at`` is strictly less than this value are eligible.
            limit (int): Maximum rows to delete per call. Defaults to 1000.

        Returns:
            int: The number of rows physically deleted.

        Raises:
            ValueError: If ``entity_type`` is not a recognised entity type.
        """
        if limit <= 0:
            return 0
        table, pk = _resolve_table(entity_type)

        eligible_ph = ",".join("?" * len(_GC_ELIGIBLE_STATUSES))
        eligible_vals = list(_GC_ELIGIBLE_STATUSES)

        # ORDER BY retired_at ASC for deterministic forward progress.
        # No SQL LIMIT here — the limit is applied after the legal-hold filter
        # below so held rows at the front of the batch don't starve eligible rows.
        select_sql = (
            f"SELECT {pk} FROM {table} "  # noqa: S608
            f"WHERE status IN ({eligible_ph}) "
            f"AND retired_at IS NOT NULL AND retired_at < ? "
            f"ORDER BY retired_at ASC"
        )
        select_params: list[Any] = [*eligible_vals, older_than_epoch]

        with self._lock:
            rows = self.conn.execute(select_sql, select_params).fetchall()
            if not rows:
                return 0

            candidate_ids: list[str] = [str(r[0]) for r in rows]
            ids_to_delete: list[str] = []
            for eid in candidate_ids:
                if self._is_on_legal_hold(self.org_id, entity_type, eid):
                    # NOTE: any real hold-check implementation must run inside
                    # the same transaction as the DELETE to remain atomic.
                    capture_anomaly(
                        "lineage.gc.legal_hold_skip",
                        level="info",
                        org_id=self.org_id,
                        entity_type=entity_type,
                        entity_id=eid,
                    )
                    continue
                ids_to_delete.append(eid)
                if len(ids_to_delete) >= limit:
                    break

            if not ids_to_delete:
                return 0

            batch_request_id = uuid.uuid4().hex
            ph = ",".join("?" * len(ids_to_delete))

            try:
                # Emit hard_delete events BEFORE the DELETE, in the same transaction.
                for eid in ids_to_delete:
                    _append_event_stmt(
                        self.conn,
                        org_id=self.org_id,
                        entity_type=entity_type,
                        entity_id=eid,
                        op="hard_delete",
                        prov="wasInvalidatedBy",
                        source_ids=[],
                        actor="system",
                        request_id=batch_request_id,
                        reason="ttl-gc",
                    )

                # Inline FTS/vec cleanup — raw DELETE to preserve atomicity.
                # Do NOT call self._fts_delete/_vec_delete: they self-commit.
                if entity_type in ("user_playbook", "agent_playbook"):
                    kind = "user" if entity_type == "user_playbook" else "agent"
                    int_ids = [int(eid) for eid in ids_to_delete]
                    int_ph = ",".join("?" * len(int_ids))
                    self.conn.execute(
                        f"DELETE FROM {kind}_playbooks_fts WHERE rowid IN ({int_ph})",
                        int_ids,
                    )
                    if self._has_sqlite_vec:  # type: ignore[attr-defined]
                        self.conn.execute(
                            f"DELETE FROM {kind}_playbooks_vec WHERE rowid IN ({int_ph})",
                            int_ids,
                        )
                else:
                    # profiles: FTS keyed on TEXT profile_id; vec keyed on implicit rowid.
                    self.conn.execute(
                        f"DELETE FROM profiles_fts WHERE profile_id IN ({ph})",
                        ids_to_delete,
                    )
                    if self._has_sqlite_vec:  # type: ignore[attr-defined]
                        rowid_rows = self.conn.execute(
                            f"SELECT rowid FROM profiles WHERE profile_id IN ({ph})",  # noqa: S608
                            ids_to_delete,
                        ).fetchall()
                        if rowid_rows:
                            rowids = [r[0] for r in rowid_rows]
                            rowid_ph = ",".join("?" * len(rowids))
                            self.conn.execute(
                                f"DELETE FROM profiles_vec WHERE rowid IN ({rowid_ph})",
                                rowids,
                            )

                cur = self.conn.execute(
                    f"DELETE FROM {table} WHERE {pk} IN ({ph})",  # noqa: S608
                    ids_to_delete,
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

        return cur.rowcount
