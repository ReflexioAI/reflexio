# ruff: noqa: S608 -- Table/column identifiers are fixed receipt-schema names.
"""One SQL-backed receipt implementation shared by SQLite and native storage.

These primitives do not publish learning. The publisher owns source validation,
learning writes and status transitions in the same outer writer transaction.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from typing import Any, Literal, cast

from reflexio.server.extensions import get_service
from reflexio.server.services.extraction.publication_receipts import (
    ENTITY_PIPELINES,
    PublicationReceipt,
    require_financial_facts,
    require_learning_ids,
    require_output_id,
    window_handoff_facts,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.work_scope import WORK_SCOPE_PROVIDER, current_project_id

from ._extraction_stream import _OWNER_PROOF_PAGE_SIZE, StreamSQL, decoded
from .agent_run._models import AgentRunRecord

_RECEIPT_COLUMNS = (
    "receipt_id,org_id,project_id,run_id,output_id,row_kind,entity_type,learning_ids,"
    "delivery_owner,delivery_state,delivery_key,delivery_facts"
)


def _observed_learning_ids(
    rows: list[dict[str, Any]], entity_type: str, *, legacy: bool = False
) -> list[str] | None:
    if not rows:
        return None
    if len(rows) != 1:
        raise StorageError("Finalization receipt is corrupt")
    if rows[0]["entity_type"] != entity_type:
        raise StorageError("Finalization receipt entity type changed")
    try:
        return require_learning_ids(decoded(rows[0]["learning_ids"]), legacy=legacy)
    except (TypeError, ValueError) as error:
        raise StorageError("Finalization receipt is corrupt") from error


class PublicationReceiptStore:
    org_id: str
    commit_scope: Any
    _stream_sql: Any
    _require_current_agent_output: Any
    _independent_writer_metadata_sql: Any
    _durable_work_route: Any
    _publication_receipt_sql: Any
    _require_independent_writer: Any

    def _publication_scope_matches(self, org_id: str, project_id: str) -> bool:
        provider = get_service(WORK_SCOPE_PROVIDER)
        scope = provider.current() if provider is not None else None
        return (
            org_id == self.org_id
            and (provider is None or scope is not None)
            and project_id == ((scope.project_id or "") if scope else "")
            and (scope is None or not scope.org_id or scope.org_id == self.org_id)
        )

    def _get_publication_receipt(
        self, receipt_id: str, *, project_id: str
    ) -> PublicationReceipt | None:
        """Reload an exact original-scope key on the ordinary APP writer route."""
        if not self._publication_scope_matches(self.org_id, project_id):
            raise ValueError("Publication financial scope changed")
        with self._publication_receipt_sql() as db:
            rows = db.query(
                f"SELECT {_RECEIPT_COLUMNS} FROM {db.table('_agent_output_receipts')} "
                "WHERE receipt_id=? AND org_id=? AND project_id=? LIMIT 1",
                (receipt_id, self.org_id, project_id),
            )
        if not self._publication_scope_matches(self.org_id, project_id):
            raise ValueError("Publication financial scope changed")
        return PublicationReceipt.from_row(rows[0]) if rows else None

    def _publication_delivery_page(
        self, *, after: int | None = None, through: int | None = None
    ) -> tuple[int | None, dict[str, Any] | None]:
        """Inspect one pending position in a finite sweep, outside caller scopes.

        Only metadata crosses this boundary. Advance the caller's frontier before
        parsing financial facts or binding the captured project for delivery.
        No receipt is claimed here and unknown/failed reads never imply idle.
        """
        with self._independent_writer_metadata_sql() as db:
            table, order = db.table("_agent_output_receipts"), db.publication_order()
            if through is None:
                ceiling = db.query(
                    f"SELECT {order} AS delivery_order FROM {table} "
                    f"WHERE org_id=? AND delivery_state='pending' ORDER BY {order} DESC LIMIT 1",
                    (self.org_id,),
                )
                if not ceiling:
                    return None, None
                through = int(ceiling[0]["delivery_order"])
            # None is unstarted/empty, not a numeric position. SQLite's valid
            # ROWID domain includes zero and negative positions.
            lower = "" if after is None else f" AND {order}>?"
            params = (
                (self.org_id, through)
                if after is None
                else (self.org_id, after, through)
            )
            rows = db.query(
                f"SELECT {order} AS delivery_order,receipt_id,org_id,project_id FROM {table} "
                f"WHERE org_id=? AND delivery_state='pending'{lower} AND {order}<=? "
                f"ORDER BY {order} LIMIT 1",
                params,
            )
            return through, rows[0] if rows else None

    def _lock_erasure_callback_closure(self, db: StreamSQL, user_id: str) -> None:
        """Close callbacks before R/receipt handoff; never wait in the A/D/R cycle.

        W and erasure authorization belong to the caller. Lock originated
        answers first so new FK attachments cannot enlarge their affected-run
        set, then the affected runs, their dependencies and remaining answers.
        The other user's surviving run is part of this closure even though it
        will not be deleted. SQLite's enclosing writer already serializes it.
        """
        runs = db.table("_agent_runs")
        dependencies = db.table("_run_tool_dependencies")
        answers = db.table("_pending_tool_calls")
        affected = (
            "r.org_id=? AND (r.user_id=? OR EXISTS ("
            f"SELECT 1 FROM {dependencies} link JOIN {answers} origin "
            "ON origin.id=link.pending_tool_call_id WHERE link.run_id=r.id "
            "AND origin.org_id=? AND origin.user_id=?))"
        )
        owners = (self.org_id, user_id, self.org_id, user_id)
        closure = (
            (
                f"SELECT p.id FROM {answers} p WHERE p.org_id=? AND p.user_id=?",
                ("p.id",),
                (self.org_id, user_id),
            ),
            (f"SELECT r.id FROM {runs} r WHERE {affected}", ("r.id",), owners),
            (
                f"SELECT d.run_id,d.pending_tool_call_id FROM {dependencies} d WHERE EXISTS ("
                f"SELECT 1 FROM {runs} r WHERE r.id=d.run_id AND {affected})",
                ("d.run_id", "d.pending_tool_call_id"),
                owners,
            ),
            (
                f"SELECT p.id FROM {answers} p WHERE p.org_id=? AND (p.user_id=? OR EXISTS ("
                f"SELECT 1 FROM {dependencies} link JOIN {runs} r ON r.id=link.run_id "
                f"WHERE link.pending_tool_call_id=p.id AND {affected}))",
                ("p.id",),
                (self.org_id, user_id, *owners),
            ),
        )
        for statement, keys, parameters in closure:
            after = None
            ordering = ",".join(keys)
            while True:
                frontier = (
                    ""
                    if after is None
                    else f" AND ({ordering})>({','.join('?' for _ in keys)})"
                )
                rows = db.query(
                    statement
                    + frontier
                    + f" ORDER BY {ordering} LIMIT {_OWNER_PROOF_PAGE_SIZE}"
                    + db.validation_lock(),
                    parameters if after is None else (*parameters, *after),
                )
                if not rows:
                    break
                after = tuple(rows[-1][key.rsplit(".", 1)[-1]] for key in keys)

    def _handoff_erased_output_finances(self, db: StreamSQL, user_id: str) -> None:
        """Retain pending output debt on authorized erasure's existing writer.

        Lock parents before receipts, after the caller's user-work fence. Only
        financial descriptors survive; accepted and historical rows cascade.
        """
        self._lock_erasure_callback_closure(db, user_id)
        after_run = ""
        while True:
            runs = db.query(
                f"SELECT {db.erasure_run_columns()} FROM {db.table('_agent_runs')} r "
                "WHERE user_id=? AND id>? ORDER BY id LIMIT 200" + db.validation_lock(),
                (user_id, after_run),
            )
            if not runs:
                return
            for run in runs:
                after_run = run["id"]
                if run["org_id"] != self.org_id or not isinstance(
                    run["project_id"], str
                ):
                    raise ValueError("Original output financial scope is unavailable")
                after_receipt = None
                while True:
                    # Native receipt IDs are UUIDs: the first page must not
                    # compare them against an invalid empty-string frontier.
                    frontier = " AND receipt_id>?" if after_receipt is not None else ""
                    receipts = db.query(
                        f"SELECT {_RECEIPT_COLUMNS} FROM {db.table('_agent_output_receipts')} "
                        "WHERE run_id=? AND row_kind='output' AND delivery_state='pending'"
                        + frontier
                        + " ORDER BY receipt_id LIMIT 200"
                        + db.lock(),
                        (run["id"], after_receipt)
                        if after_receipt is not None
                        else (run["id"],),
                    )
                    if not receipts:
                        break
                    for row in receipts:
                        receipt = PublicationReceipt.from_row(row)
                        if (
                            receipt.org_id != self.org_id
                            or receipt.project_id != run["project_id"]
                            or receipt.run_id != run["id"]
                            or receipt.entity_type
                            != {"profile": "profile", "playbook": "user_playbook"}.get(
                                run["extractor_kind"]
                            )
                        ):
                            raise ValueError(
                                "Original output financial binding changed"
                            )
                        saved = db.query(
                            f"UPDATE {db.table('_agent_output_receipts')} SET run_id=NULL,output_id=NULL "
                            "WHERE receipt_id=? AND run_id=? "
                            f"RETURNING {_RECEIPT_COLUMNS}",
                            (receipt.receipt_id, run["id"]),
                        )
                        expected = replace(receipt, run_id=None, output_id=None)
                        if (
                            len(saved) != 1
                            or PublicationReceipt.from_row(saved[0]) != expected
                        ):
                            raise ValueError("Output financial handoff changed")
                        after_receipt = receipt.receipt_id

    def _handoff_erased_window_finances(self, db: StreamSQL, user_id: str) -> None:
        """Join authorized erasure's writer; never open or commit a transaction.

        The outer owner holds the subject's work fence before calling this.
        Window locks and unique receipt keys keep transfer and deletion atomic.
        Pages bound reads, not the number of obligations preserved by erasure.
        """
        after = ""
        while True:
            rows = db.query(
                f"SELECT window_id,org_id,kind,project_id,effects->'billing' AS billing "
                f"FROM {db.table('extraction_windows')} WHERE user_id=? "
                "AND completed=1 AND effects_done=0 AND window_id>? "
                "ORDER BY window_id LIMIT 200" + db.lock(),
                (user_id, after),
            )
            if not rows:
                return
            for row in rows:
                after = row["window_id"]
                billing = decoded(row["billing"])
                if billing is None:
                    continue
                if row["org_id"] != self.org_id:
                    raise ValueError("Original window financial scope is unavailable")
                if not isinstance(row["project_id"], str):
                    raise ValueError("Window financial project is unavailable")
                ids, facts = window_handoff_facts(row["kind"], billing)
                existing = db.query(
                    f"SELECT {_RECEIPT_COLUMNS} FROM {db.table('_agent_output_receipts')} "
                    "WHERE org_id=? AND project_id=? AND row_kind='window_handoff' "
                    "AND delivery_key=?" + db.lock(),
                    (self.org_id, row["project_id"], row["window_id"]),
                )
                if existing:
                    observed = PublicationReceipt.from_row(existing[0])
                    if (
                        len(existing) != 1
                        or observed.learning_ids != tuple(ids)
                        or observed.financial_facts != facts
                        or observed.delivery_state != "pending"
                    ):
                        raise ValueError("Window financial handoff changed")
                    continue
                db.query(
                    f"INSERT INTO {db.table('_agent_output_receipts')} "
                    "(receipt_id,org_id,project_id,row_kind,entity_type,learning_ids,"
                    "delivery_state,delivery_key,delivery_facts) VALUES(?,?,?,?,?,?,?,?,?)",
                    (
                        uuid.uuid4().hex,
                        self.org_id,
                        row["project_id"],
                        "window_handoff",
                        "profile" if row["kind"] == "profile" else "user_playbook",
                        json.dumps(ids, separators=(",", ":")),
                        "pending",
                        row["window_id"],
                        json.dumps(
                            facts,
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    ),
                )

    def get_agent_run_finalization_receipt(
        self, *, run_id: str, entity_type: str
    ) -> list[str] | None:
        """Observe current output or explicit old history, never an arbitrary row."""
        with self._stream_sql(read_only=True) as db:
            rows = db.query(
                f"SELECT {db.publication_binding_columns()} FROM {db.table('_agent_runs')} r "
                f"WHERE r.org_id=? AND r.id=? AND {db.project_filter()}",
                (self.org_id, run_id, current_project_id() or ""),
            )
            if not rows:
                return None
            if rows[0]["has_publication"]:
                try:
                    output_id = require_output_id(rows[0]["output_id"])
                except ValueError as error:
                    raise StorageError("Finalization receipt is corrupt") from error
                receipts = db.query(
                    f"SELECT entity_type,learning_ids FROM {db.table('_agent_output_receipts')} "
                    "WHERE org_id=? AND project_id=? AND run_id=? AND output_id=? AND row_kind='output'",
                    (self.org_id, current_project_id() or "", run_id, output_id),
                )
                legacy = False
            else:
                receipts = db.query(
                    f"SELECT q.entity_type,q.learning_ids FROM {db.table('_agent_run_finalization_receipts')} q "
                    f"JOIN {db.table('_agent_runs')} r ON r.id=q.run_id "
                    f"WHERE r.org_id=? AND {db.project_filter()} AND {db.project_filter('q')} AND r.id=?",
                    (
                        self.org_id,
                        current_project_id() or "",
                        current_project_id() or "",
                        run_id,
                    ),
                )
                legacy = True
        return _observed_learning_ids(receipts, entity_type, legacy=legacy)

    def get_agent_output_finalization_receipt(
        self, *, run_id: str, output_id: str, entity_type: str
    ) -> list[str] | None:
        """Observe exactly one output; neither claim authority nor infer history."""
        require_output_id(output_id)
        with self._stream_sql(read_only=True) as db:
            rows = db.query(
                f"SELECT entity_type,learning_ids FROM {db.table('_agent_output_receipts')} "
                "WHERE org_id=? AND project_id=? AND run_id=? AND output_id=? AND row_kind='output'",
                (self.org_id, current_project_id() or "", run_id, output_id),
            )
        return _observed_learning_ids(rows, entity_type)

    def _get_output_publication_receipt(
        self, db: StreamSQL, run_id: str, output_id: str
    ) -> PublicationReceipt | None:
        rows = db.query(
            f"SELECT {_RECEIPT_COLUMNS} FROM {db.table('_agent_output_receipts')} "
            "WHERE org_id=? AND project_id=? AND run_id=? AND output_id=? AND row_kind='output'",
            (self.org_id, current_project_id() or "", run_id, output_id),
        )
        if len(rows) > 1:
            raise ValueError("Publication receipt identity is not unique")
        return PublicationReceipt.from_row(rows[0]) if rows else None

    def _create_agent_output_receipt(
        self,
        run: AgentRunRecord,
        *,
        learning_ids: list[str],
        origin: Literal["initial_inline", "resume"],
    ) -> tuple[PublicationReceipt, bool]:
        """Join the actual retained writer; the enclosing publisher owns effects."""
        from reflexio.server.services.durable_learning.user_lease import (
            owned_agent_run_write,
        )
        from reflexio.server.services.storage.storage_base import BaseStorage

        ids = require_learning_ids(learning_ids)
        output = run.committed_output
        publication = output.get("publication") if isinstance(output, dict) else None
        if not isinstance(publication, dict):
            raise ValueError("Publication requires retained output identity")
        output_id = require_output_id(publication.get("id"))
        entity_type = {"profile": "profile", "playbook": "user_playbook"}.get(
            run.binding.extractor_kind
        )
        if entity_type is None or origin not in {"initial_inline", "resume"}:
            raise ValueError("Unknown publication origin")
        with (
            owned_agent_run_write(cast(BaseStorage, self), run),
            self._stream_sql() as db,
        ):
            self._require_current_agent_output(db, run, output)
            existing = self._get_output_publication_receipt(db, run.id, output_id)
            if existing is not None:
                if (
                    existing.entity_type != entity_type
                    or existing.learning_ids != tuple(ids)
                ):
                    raise ValueError("Publication learning binding is immutable")
                return existing, False
            source = (
                "resumable_extraction" if origin == "resume" else run.binding.source
            )
            if origin == "resume":
                clock = run.created_at or run.agent_completed_at
                if clock is None:
                    raise ValueError("Publication billing timestamp is unavailable")
                created_at = clock.timestamp()
            else:
                created_at = db.now()
            facts = (
                require_financial_facts(
                    "output",
                    entity_type,
                    {
                        "pipeline": ENTITY_PIPELINES[entity_type],
                        "source": source,
                        "created_at": created_at,
                    },
                )
                if ids
                else None
            )
            receipt_id = uuid.uuid4().hex
            db.query(
                f"INSERT INTO {db.table('_agent_output_receipts')} "
                "(receipt_id,org_id,project_id,run_id,output_id,entity_type,learning_ids,delivery_state,delivery_facts) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    receipt_id,
                    self.org_id,
                    current_project_id() or "",
                    run.id,
                    output_id,
                    entity_type,
                    json.dumps(ids, separators=(",", ":")),
                    "pending" if ids else "accepted",
                    json.dumps(
                        facts, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    if facts is not None
                    else None,
                ),
            )
            saved = self._get_output_publication_receipt(db, run.id, output_id)
            if saved is None or saved.receipt_id != receipt_id:
                raise ValueError("Publication receipt was not retained")
            return saved, True

    def _ack_publication_receipt(self, expected: PublicationReceipt) -> bool:
        """Acknowledge exact observed strict acceptance; take no source/owner locks."""
        if not self._publication_scope_matches(expected.org_id, expected.project_id):
            return False
        if expected.row_kind == "legacy" or expected.delivery_state == "historical":
            raise ValueError("Historical publication is not delivery acknowledgement")
        with self._publication_receipt_sql(read_only=False) as db:
            rows = db.query(
                f"SELECT {_RECEIPT_COLUMNS} FROM {db.table('_agent_output_receipts')} "
                "WHERE receipt_id=? AND org_id=? AND project_id=?" + db.lock(),
                (expected.receipt_id, self.org_id, expected.project_id),
            )
            if not rows:
                return False  # absence is not proof of acceptance
            current = PublicationReceipt.from_row(rows[0])
            for name in (
                "receipt_id",
                "org_id",
                "project_id",
                "row_kind",
                "entity_type",
                "learning_ids",
                "delivery_owner",
                "delivery_key",
            ):
                if getattr(current, name) != getattr(expected, name):
                    return False
            if (current.run_id, current.output_id) != (
                expected.run_id,
                expected.output_id,
            ) and (current.run_id, current.output_id) != (None, None):
                return False
            if current.delivery_state == "accepted":
                return True
            if (
                expected.delivery_state != "pending"
                or current.financial_facts != expected.financial_facts
            ):
                return False
            db.query(
                f"UPDATE {db.table('_agent_output_receipts')} SET delivery_state='accepted',delivery_facts=NULL "
                "WHERE receipt_id=? AND org_id=? AND project_id=?",
                (current.receipt_id, self.org_id, current.project_id),
            )
            if current.run_id is None:
                db.query(
                    f"DELETE FROM {db.table('_agent_output_receipts')} "
                    "WHERE receipt_id=? AND org_id=? AND project_id=? AND delivery_state='accepted' AND run_id IS NULL",
                    (current.receipt_id, self.org_id, current.project_id),
                )
            return True
