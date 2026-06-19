import json
import sqlite3
import threading
import time
from typing import Any

from reflexio.models.api_schema.domain.entities import LineageEvent


class SQLiteLineageMixin:
    """SQLite implementation of the append-only, content-free lineage event log."""

    # Type hints for instance attributes provided by SQLiteStorageBase via MRO.
    conn: sqlite3.Connection
    _lock: threading.RLock

    def append_lineage_event(self, event: LineageEvent) -> int:
        """Append an event; idempotent on (entity_id, op, request_id). Return the row id.

        Args:
            event (LineageEvent): The fully-formed event to persist. ``event_id``
                may be 0; the storage layer assigns a real id on insert. On a
                duplicate ``(entity_id, op, request_id)`` the existing row is
                returned unchanged.

        Returns:
            int: The assigned or existing ``event_id``.
        """
        created = event.created_at or int(time.time())
        with self._lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO lineage_event "
                "(org_id, entity_type, entity_id, op, prov_relation, source_ids, "
                "actor, request_id, reason, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    event.org_id,
                    event.entity_type,
                    event.entity_id,
                    event.op,
                    event.prov_relation,
                    json.dumps(event.source_ids),
                    event.actor,
                    event.request_id,
                    event.reason,
                    created,
                ),
            )
            if cur.rowcount == 0:  # duplicate (entity_id, op, request_id)
                row = self.conn.execute(
                    "SELECT event_id FROM lineage_event "
                    "WHERE entity_id=? AND op=? AND request_id=?",
                    (event.entity_id, event.op, event.request_id),
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
    ) -> list[LineageEvent]:
        """Retrieve lineage events, optionally filtered.

        Args:
            entity_type (str | None): Filter to events for this entity type.
            entity_id (str | None): Filter to events for this entity id.
            org_id (str | None): Filter to events for this org.

        Returns:
            list[LineageEvent]: Matching events ordered by ``event_id`` ascending.
        """
        clauses: list[str] = []
        params: list[Any] = []
        for col, val in (
            ("entity_type", entity_type),
            ("entity_id", entity_id),
            ("org_id", org_id),
        ):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
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
            )
            for r in rows
        ]
