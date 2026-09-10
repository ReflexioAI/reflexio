# ruff: noqa: S608 -- SQL identifiers are resolved from a fixed backend table registry.
"""Durable, arrival-ordered extraction state shared by SQL storage backends.

The backend supplies only a transaction-bound SQL adapter. Scheduling, window
selection and fencing live here so SQLite and PostgreSQL obey the same rules.
"""

from __future__ import annotations

import json
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from reflexio.server.work_scope import current_project_id

Kind = Literal["profile", "playbook"]
KINDS: tuple[Kind, ...] = ("profile", "playbook")


class StreamSQL(Protocol):
    """Parameterised queries; identifiers are resolved by the backend."""

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]: ...
    def table(self, name: str) -> str: ...
    def now(self) -> float: ...
    def lock(self, *, skip_locked: bool = False) -> str: ...
    def eligible(self, kind: Kind) -> str: ...
    def project_filter(self) -> str: ...
    def input_lock(self) -> str: ...
    def admission_value(self, field: str) -> str: ...


def decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


@dataclass(frozen=True)
class Window:
    window_id: str
    user_id: str
    kind: Kind
    project_id: str
    predecessor: int
    end_seq: int
    manifest: list[dict[str, Any]]
    policy: dict[str, Any]
    force: bool
    skip_aggregation: bool
    outcome: dict[str, Any] | None = None
    invalidated: bool = False

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Window:
        return cls(
            window_id=row["window_id"],
            user_id=row["user_id"],
            kind=row["kind"],
            project_id=row["project_id"],
            predecessor=int(row["predecessor"]),
            end_seq=int(row["end_seq"]),
            manifest=decoded(row["manifest"]),
            policy=decoded(row["policy"]),
            force=bool(row["force"]),
            skip_aggregation=bool(row["skip_aggregation"]),
            outcome=decoded(row.get("outcome")),
            invalidated=bool(row.get("invalidated", False)),
        )


class WindowInputsDeletedError(RuntimeError):
    """A saved window has been invalidated by explicit deletion."""


class LeaseLostError(RuntimeError):
    """The complete transaction must roll back when a claim is superseded."""


class ExtractionStreamStore:
    """Methods join the storage commit_scope; never perform model/network work."""

    org_id: str

    def _stream_sql(
        self, *, discovery: bool = False, read_only: bool = False
    ) -> AbstractContextManager[StreamSQL]:
        raise NotImplementedError

    def admit_extraction(
        self,
        user_id: str,
        request_id: str,
        interaction_ids: list[int],
        admission: dict[str, Any],
    ) -> None:
        """Called in the SAME commit_scope as request/interaction inserts.

        lock_extraction_stream MUST be called before those inserts. Holding the
        row lock until commit makes sequence order equal accepted commit order.
        """
        with self._stream_sql() as db:
            work = db.table("learning_work")
            rows = db.query(
                f"UPDATE {work} SET highwater=highwater+?, revision=revision+1, "
                "due_at=0,pending_since=CASE WHEN pending_since=0 THEN ? ELSE pending_since END WHERE org_id=? AND user_id=? RETURNING highwater",
                (len(interaction_ids), db.now(), self.org_id, user_id),
            )
            end = int(rows[0]["highwater"])
            start = end - len(interaction_ids) + 1
            admission = {**admission, "min_seq": start, "max_seq": end}
            db.query(
                f"UPDATE {db.table('requests')} SET learning_admission=? "
                "WHERE request_id=? AND user_id=?",
                (json.dumps(admission), request_id, user_id),
            )
            # Bounded set updates avoid one database round trip per interaction.
            for offset in range(0, len(interaction_ids), 200):
                ids = interaction_ids[offset : offset + 200]
                cases = " ".join("WHEN ? THEN ?" for _ in ids)
                placeholders = ",".join("?" for _ in ids)
                values = tuple(
                    value
                    for seq, i in enumerate(ids, start + offset)
                    for value in (i, seq)
                )
                db.query(
                    f"UPDATE {db.table('interactions')} SET ingestion_seq=CASE interaction_id {cases} END "
                    f"WHERE user_id=? AND interaction_id IN ({placeholders})",
                    (*values, user_id, *ids),
                )

    def lock_extraction_stream(self, user_id: str) -> None:
        with self._stream_sql() as db:
            work = db.table("learning_work")
            db.query(
                f"INSERT INTO {work} (org_id,user_id) VALUES (?,?) "
                "ON CONFLICT (org_id,user_id) DO NOTHING",
                (self.org_id, user_id),
            )
            db.query(
                f"SELECT highwater FROM {work} WHERE org_id=? AND user_id=?{db.lock()}",
                (self.org_id, user_id),
            )
            for kind in KINDS:
                db.query(
                    f"INSERT INTO {db.table('extraction_cursors')} (user_id,kind,project_id) "
                    "VALUES (?,?,?) ON CONFLICT (user_id,kind,project_id) DO NOTHING",
                    (user_id, kind, current_project_id() or ""),
                )

    def claim_extraction(
        self, owner: str, lease_seconds: int
    ) -> tuple[str, str] | None:
        """Claim ONE user only after the caller has reserved execution capacity."""
        with self._stream_sql() as db:
            now = db.now()
            work = db.table("learning_work")
            rows = db.query(
                f"SELECT user_id FROM {work} WHERE org_id=? AND due_at<=? "
                "AND lease_until<=? ORDER BY last_turn,user_id LIMIT 1"
                + db.lock(skip_locked=True),
                (self.org_id, now, now),
            )
            if not rows:
                return None
            user_id, token = rows[0]["user_id"], uuid.uuid4().hex
            db.query(
                f"UPDATE {work} SET lease_token=?,lease_owner=?,lease_until=?,last_turn=? "
                "WHERE org_id=? AND user_id=?",
                (token, owner, now + lease_seconds, now, self.org_id, user_id),
            )
            return user_id, token

    def renew_extraction(self, user_id: str, token: str, seconds: int) -> bool:
        with self._stream_sql() as db:
            now = db.now()
            return bool(
                db.query(
                    f"UPDATE {db.table('learning_work')} SET lease_until=? "
                    "WHERE org_id=? AND user_id=? AND lease_token=? AND lease_until>? RETURNING user_id",
                    (now + seconds, self.org_id, user_id, token, now),
                )
            )

    def _fence(self, db: StreamSQL, user_id: str, token: str) -> None:
        if not db.query(
            f"SELECT user_id FROM {db.table('learning_work')} "
            "WHERE org_id=? AND user_id=? AND lease_token=? AND lease_until>?"
            + db.lock(),
            (self.org_id, user_id, token, db.now()),
        ):
            raise LeaseLostError("Extraction lease was superseded")

    def _inputs(
        self,
        db: StreamSQL,
        user_id: str,
        kind: Kind,
        after: int,
        limit: int,
        *,
        project_id: str = "",
        context: bool = False,
    ) -> list[dict[str, Any]]:
        comparator, order = ("<=", "DESC") if context else (">", "ASC")
        rows = db.query(
            f"SELECT i.interaction_id,i.ingestion_seq,i.request_id,r.learning_admission "
            f"FROM {db.table('interactions')} i JOIN {db.table('requests')} r "
            "ON r.request_id=i.request_id AND r.user_id=i.user_id "
            f"WHERE i.user_id=? AND {db.project_filter()} AND i.ingestion_seq {comparator} ? AND {db.eligible(kind)} "
            f"ORDER BY i.ingestion_seq {order} LIMIT ?",
            (user_id, project_id, after, limit),
        )
        return list(reversed(rows)) if context else rows

    def _select(
        self, db: StreamSQL, user_id: str, kind: Kind, cursor: dict[str, Any]
    ) -> tuple[Window | None, int]:
        after = int(cursor["completed_seq"])
        project_id = cursor["project_id"]
        first = self._inputs(db, user_id, kind, after, 1, project_id=project_id)
        if not first:
            return None, 0
        admission = decoded(first[0]["learning_admission"])
        policy = admission[kind]
        width, stride = int(policy["window_size"]), int(policy["stride_size"])
        if not 1 <= stride <= width:
            raise ValueError("Extraction requires 1 <= stride_size <= window_size")
        context = self._inputs(
            db, user_id, kind, after, width, project_id=project_id, context=True
        )
        needed = max(stride, width - len(context)) if cursor["started"] else width
        new = self._inputs(db, user_id, kind, after, width, project_id=project_id)
        # A force barrier covers all preceding work, including full windows.
        end_expr = db.admission_value("max_seq")
        force_expr = db.admission_value("force")
        barrier = db.query(
            f"SELECT MIN({end_expr}) AS barrier FROM {db.table('requests')} r "
            f"WHERE r.user_id=? AND {db.project_filter()} AND {force_expr}=1 "
            f"AND {db.eligible(kind)} AND {end_expr}>? "
            f"AND EXISTS(SELECT 1 FROM {db.table('interactions')} live "
            "WHERE live.user_id=r.user_id AND live.request_id=r.request_id AND live.ingestion_seq>?)",
            (user_id, project_id, after, after),
        )[0]["barrier"]
        forced = barrier is not None
        if barrier is not None:
            new = [row for row in new if int(row["ingestion_seq"]) <= barrier]
        if len(new) < needed and not forced:
            return None, needed - len(new)
        new = new[:needed]
        if not new:
            return None, 0
        manifest = [
            {
                "interaction_id": int(row["interaction_id"]),
                "seq": int(row["ingestion_seq"]),
                "request_id": row["request_id"],
            }
            for row in (context + new)[-width:]
        ]
        window = Window(
            window_id=uuid.uuid4().hex,
            user_id=user_id,
            kind=kind,
            project_id=project_id,
            predecessor=after,
            end_seq=int(new[-1]["ingestion_seq"]),
            manifest=manifest,
            policy=policy,
            force=forced,
            skip_aggregation=any(
                decoded(row["learning_admission"]).get("skip_aggregation", False)
                for row in new
            ),
        )
        return window, 0

    def prepare_extraction(self, user_id: str, token: str) -> Window | None:
        with self._stream_sql(discovery=True) as db:
            self._fence(db, user_id, token)
            now = db.now()
            cursors = db.query(
                f"SELECT * FROM {db.table('extraction_cursors')} WHERE user_id=? "
                "ORDER BY last_turn,kind",
                (user_id,),
            )
            effects_due = db.query(
                f"SELECT MIN(effects_retry_at) AS due FROM {db.table('extraction_windows')} "
                "WHERE user_id=? AND completed=1 AND effects_done=0",
                (user_id,),
            )[0]["due"]
            next_due = float(effects_due) if effects_due is not None else float("inf")
            for cursor in cursors:
                if cursor["retry_at"] > now:
                    next_due = min(next_due, cursor["retry_at"])
                    continue
                if cursor["window_id"]:
                    rows = db.query(
                        f"SELECT * FROM {db.table('extraction_windows')} WHERE window_id=?",
                        (cursor["window_id"],),
                    )
                    return Window.from_row(rows[0])
                window, _ = self._select(db, user_id, cursor["kind"], cursor)
                if window is None:
                    continue
                db.query(
                    f"INSERT INTO {db.table('extraction_windows')} "
                    "(window_id,user_id,kind,predecessor,end_seq,manifest,policy,force,skip_aggregation,project_id) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        window.window_id,
                        user_id,
                        window.kind,
                        window.predecessor,
                        window.end_seq,
                        json.dumps(window.manifest),
                        json.dumps(window.policy),
                        int(window.force),
                        int(window.skip_aggregation),
                        window.project_id,
                    ),
                )
                db.query(
                    f"UPDATE {db.table('extraction_cursors')} SET window_id=? WHERE user_id=? AND kind=? AND project_id=?",
                    (window.window_id, user_id, window.kind, window.project_id),
                )
                return window
            # Same row lock as ingestion: publication either precedes this
            # inspection or follows this update and sets due_at=0 again.
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=?,pending_since=CASE WHEN ?=1 THEN 0 ELSE pending_since END,lease_until=0,lease_token=NULL "
                "WHERE org_id=? AND user_id=?",
                (next_due, int(next_due == float("inf")), self.org_id, user_id),
            )
            return None

    def save_extraction_outcome(
        self, window: Window, token: str, outcome: dict[str, Any]
    ) -> None:
        with self._stream_sql() as db:
            self._fence(db, window.user_id, token)
            rows = db.query(
                f"UPDATE {db.table('extraction_windows')} SET outcome=? WHERE window_id=? AND outcome IS NULL AND invalidated=0 RETURNING window_id",
                (json.dumps(outcome), window.window_id),
            )
            if not rows:
                raise WindowInputsDeletedError("Computed window is no longer valid")

    def complete_extraction(
        self, window: Window, token: str, effects: dict[str, Any]
    ) -> None:
        """Call after output writes INSIDE their commit_scope; failures roll back all."""
        effects = {
            **effects,
            "request_ids": list(
                dict.fromkeys(
                    m["request_id"]
                    for m in window.manifest
                    if m["seq"] > window.predecessor
                )
            ),
        }
        with self._stream_sql() as db:
            self._fence(db, window.user_id, token)
            rows = db.query(
                f"UPDATE {db.table('extraction_cursors')} SET completed_seq=?,started=1,window_id=NULL,"
                "attempts=0,retry_at=0,last_error=NULL,last_turn=? "
                "WHERE user_id=? AND kind=? AND completed_seq=? AND window_id=? AND project_id=? RETURNING user_id",
                (
                    window.end_seq,
                    db.now(),
                    window.user_id,
                    window.kind,
                    window.predecessor,
                    window.window_id,
                    window.project_id,
                ),
            )
            if not rows:
                raise LeaseLostError("Extraction cursor was superseded")
            db.query(
                f"UPDATE {db.table('extraction_windows')} SET completed=1,effects=? WHERE window_id=?",
                (json.dumps(effects), window.window_id),
            )
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=0,lease_until=0,lease_token=NULL "
                "WHERE org_id=? AND user_id=?",
                (self.org_id, window.user_id),
            )

    def retry_extraction(self, window: Window, token: str, error: str) -> None:
        with self._stream_sql() as db:
            self._fence(db, window.user_id, token)
            row = db.query(
                f"SELECT attempts FROM {db.table('extraction_cursors')} WHERE user_id=? AND kind=? AND project_id=?",
                (window.user_id, window.kind, window.project_id),
            )[0]
            attempts = int(row["attempts"]) + 1
            due = db.now() + min(300, 2 ** min(attempts - 1, 9))
            db.query(
                f"UPDATE {db.table('extraction_cursors')} SET attempts=?,retry_at=?,last_error=?,last_turn=? "
                "WHERE user_id=? AND kind=? AND project_id=?",
                (
                    attempts,
                    due,
                    error,
                    db.now(),
                    window.user_id,
                    window.kind,
                    window.project_id,
                ),
            )
            # Let the sibling cursor run before the failed cursor's next retry.
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=0,lease_until=0,lease_token=NULL "
                "WHERE org_id=? AND user_id=?",
                (self.org_id, window.user_id),
            )

    def extraction_status(self, user_id: str, request_id: str) -> dict[str, Any]:
        with self._stream_sql(read_only=True) as db:
            rows = db.query(
                f"SELECT learning_admission FROM {db.table('requests')} WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            )
            if not rows or rows[0]["learning_admission"] is None:
                return {"status": "not_tracked", "reason": "before_cutover"}
            admission = decoded(rows[0]["learning_admission"])
            surviving = db.query(
                f"SELECT MAX(ingestion_seq) AS end_seq FROM {db.table('interactions')} WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            )[0]["end_seq"]
            if surviving is None:
                return {
                    "status": "done",
                    "reason": "inputs_erased"
                    if admission["max_seq"] >= admission["min_seq"]
                    else "not_applicable",
                }
            cursors = db.query(
                f"SELECT * FROM {db.table('extraction_cursors')} WHERE user_id=? AND project_id=?",
                (user_id, current_project_id() or ""),
            )
            by_kind = {c["kind"]: c for c in cursors}
            required = [
                kind for kind in KINDS if admission.get(kind, {}).get("eligible")
            ]
            missing = [kind for kind in required if kind not in by_kind]
            if missing:
                return {"status": "pending", "reason": "cursor_unavailable"}
            pending = [
                by_kind[kind]
                for kind in required
                if int(by_kind[kind]["completed_seq"]) < int(surviving)
            ]
            if not pending:
                return {
                    "status": "done",
                    "reason": "covered" if required else "not_applicable",
                }
            if any(c["retry_at"] > db.now() for c in pending):
                return {"status": "pending", "reason": "retrying"}
            work = db.query(
                f"SELECT lease_until FROM {db.table('learning_work')} WHERE org_id=? AND user_id=?",
                (self.org_id, user_id),
            )
            if (
                any(c["window_id"] for c in pending)
                and work
                and work[0]["lease_until"] > db.now()
            ):
                return {"status": "processing", "reason": "extracting"}
            if any(
                c["window_id"] or self._select(db, user_id, c["kind"], c)[0] is not None
                for c in pending
            ):
                return {"status": "pending", "reason": "queued"}
            return {"status": "pending", "reason": "waiting_for_window"}

    def list_extraction_orgs(self) -> list[str]:
        with self._stream_sql(read_only=True) as db:
            now = db.now()
            return [
                row["org_id"]
                for row in db.query(
                    f"SELECT DISTINCT org_id FROM {db.table('learning_work')} WHERE due_at<=? AND lease_until<=? ORDER BY org_id",
                    (now, now),
                )
            ]

    def oldest_extraction_backlog_age(self) -> float | None:
        with self._stream_sql(read_only=True) as db:
            first = db.query(
                f"SELECT MIN(pending_since) AS oldest FROM {db.table('learning_work')} WHERE pending_since>0"
            )[0]["oldest"]
            return max(0, db.now() - float(first)) if first is not None else None

    def pending_extraction_effects(
        self, limit: int = 1, *, user_id: str | None = None, token: str | None = None
    ) -> list[tuple[Window, dict[str, Any]]]:
        with self._stream_sql(discovery=True) as db:
            if token is not None and user_id is not None:
                self._fence(db, user_id, token)
            return [
                (Window.from_row(row), decoded(row["effects"]))
                for row in db.query(
                    f"SELECT * FROM {db.table('extraction_windows')} WHERE completed=1 AND effects_done=0 AND effects_retry_at<=? AND (? IS NULL OR user_id=?) LIMIT ?",
                    (db.now(), user_id, user_id, limit),
                )
            ]

    def claim_extraction_derived(self, window: Window, token: str) -> bool:
        """At-most-once handoff to the existing best-effort derived schedulers.

        Billing and run finalization must succeed first. A subsequent ack/retry
        cannot reschedule a winner's optimization, aggregation or tagging.
        """
        with self._stream_sql() as db:
            self._fence(db, window.user_id, token)
            return bool(
                db.query(
                    f"UPDATE {db.table('extraction_windows')} SET derived_claimed=1 WHERE window_id=? AND completed=1 AND derived_claimed=0 RETURNING window_id",
                    (window.window_id,),
                )
            )

    def ack_extraction_effects(
        self, window_id: str, *, user_id: str | None = None, token: str | None = None
    ) -> None:
        with self._stream_sql() as db:
            if token is not None and user_id is not None:
                self._fence(db, user_id, token)
            rows = db.query(
                f"SELECT effects FROM {db.table('extraction_windows')} WHERE window_id=?",
                (window_id,),
            )
            if rows:
                effects = decoded(rows[0]["effects"]) or {}
                summary = {
                    "version": 1,
                    "skipped": True,
                    "billing": effects.get("billing"),
                    "request_ids": effects.get("request_ids", []),
                }
                db.query(
                    f"UPDATE {db.table('extraction_windows')} SET effects_done=1,outcome=NULL,effects=?,manifest='[]' WHERE window_id=?",
                    (json.dumps(summary), window_id),
                )

    def invalidate_extraction(self, window: Window, token: str) -> None:
        """Explicit erasure invalidates compute; surviving inputs are selected anew."""
        with self._stream_sql() as db:
            self._fence(db, window.user_id, token)
            db.query(
                f"UPDATE {db.table('extraction_cursors')} SET window_id=NULL WHERE window_id=?",
                (window.window_id,),
            )
            db.query(
                f"UPDATE {db.table('_agent_runs')} SET status='cancelled',"
                "claimed_by=NULL,claimed_at=NULL,updated_at=CURRENT_TIMESTAMP "
                "WHERE id=? AND status NOT IN ('finalized','cancelled','expired')",
                (f"window:{window.window_id}",),
            )
            db.query(
                f"DELETE FROM {db.table('extraction_windows')} WHERE window_id=? AND completed=0",
                (window.window_id,),
            )
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=0,lease_until=0,lease_token=NULL WHERE org_id=? AND user_id=?",
                (self.org_id, window.user_id),
            )

    def retry_extraction_effects(self, window: Window) -> None:
        with self._stream_sql() as db:
            due = db.now() + 5
            db.query(
                f"UPDATE {db.table('extraction_windows')} SET effects_retry_at=? WHERE window_id=?",
                (due, window.window_id),
            )
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=MIN(due_at,?) WHERE org_id=? AND user_id=?",
                (due, self.org_id, window.user_id),
            )

    def extraction_counts(self, user_id: str, request_id: str) -> dict[str, int]:
        result = {"profile": 0, "playbook": 0}
        with self._stream_sql(read_only=True) as db:
            rows = db.query(
                f"SELECT learning_admission FROM {db.table('requests')} WHERE user_id=? AND request_id=?",
                (user_id, request_id),
            )
            if not rows or rows[0]["learning_admission"] is None:
                return result
            admission = decoded(rows[0]["learning_admission"])
            for row in db.query(
                f"SELECT kind,effects FROM {db.table('extraction_windows')} "
                "WHERE user_id=? AND project_id=? AND completed=1 AND predecessor<? AND end_seq>=?",
                (
                    user_id,
                    current_project_id() or "",
                    admission["max_seq"],
                    admission["min_seq"],
                ),
            ):
                effect = decoded(row["effects"])
                if (
                    effect
                    and effect.get("billing")
                    and request_id in effect.get("request_ids", [])
                    and admission.get(row["kind"], {}).get("eligible")
                ):
                    result[row["kind"]] += len(effect["billing"]["ids"])
        return result

    def filter_extraction_retention(
        self, target: str, keys: list[tuple[Any, ...]]
    ) -> list[tuple[Any, ...]]:
        """Keep unfinished inputs and the last complete overlap context.

        Only ordinary row-count retention calls this. Explicit deletion never
        does; the window executor revalidates its manifest before committing.
        """
        if target not in {"interactions", "requests"}:
            return keys
        with self._stream_sql(read_only=True) as db:
            floors: dict[str, int] = {}
            for cursor in db.query(f"SELECT * FROM {db.table('extraction_cursors')}"):
                user, kind, project = (
                    cursor["user_id"],
                    cursor["kind"],
                    cursor["project_id"],
                )
                first = self._inputs(db, user, kind, 0, 1, project_id=project)
                if not first:
                    continue
                after = int(cursor["completed_seq"])
                floor = int(first[0]["ingestion_seq"])
                if after:
                    receipt = db.query(
                        f"SELECT policy FROM {db.table('extraction_windows')} WHERE user_id=? AND kind=? AND project_id=? AND end_seq=? AND completed=1",
                        (user, kind, project, after),
                    )
                    if receipt:
                        width = int(decoded(receipt[0]["policy"])["window_size"])
                        context = self._inputs(
                            db,
                            user,
                            kind,
                            after,
                            width,
                            project_id=project,
                            context=True,
                        )
                        if context:
                            floor = int(context[0]["ingestion_seq"])
                floors[user] = min(floors.get(user, floor), floor)
            protected = set()
            for row in db.query(
                f"SELECT manifest FROM {db.table('extraction_windows')} WHERE completed=0 OR effects_done=0"
            ):
                for item in decoded(row["manifest"]):
                    protected.add(
                        item["interaction_id"]
                        if target == "interactions"
                        else item["request_id"]
                    )
            id_column = "interaction_id" if target == "interactions" else "request_id"
            # Bounded chunks keep parameter counts and SQL text bounded.
            candidates = [key[0] for key in keys]
            for offset in range(0, len(candidates), 400):
                ids = candidates[offset : offset + 400]
                placeholders = ",".join("?" for _ in ids)
                for row in db.query(
                    f"SELECT interaction_id,request_id,user_id,ingestion_seq FROM {db.table('interactions')} WHERE {id_column} IN ({placeholders})",
                    tuple(ids),
                ):
                    if (
                        row["ingestion_seq"] is not None
                        and row["user_id"] in floors
                        and row["ingestion_seq"] >= floors[row["user_id"]]
                    ):
                        protected.add(row[id_column])
            return [key for key in keys if key[0] not in protected]

    def begin_extraction_resume(
        self, run_id: str, claimed_by: str, claimed_at: str
    ) -> int | None:
        """Count an actual resume attempt after its user lease is acquired.

        The caller fences the user lease in the enclosing commit scope. A claim
        lost while waiting cannot spend an attempt belonging to a new owner.
        """
        with self._stream_sql() as db:
            rows = db.query(
                f"UPDATE {db.table('_agent_runs')} SET resume_attempts=resume_attempts+1 "
                "WHERE id=? AND status='resuming' AND claimed_by=? AND claimed_at=? "
                "RETURNING resume_attempts",
                (run_id, claimed_by, claimed_at),
            )
            return int(rows[0]["resume_attempts"]) if rows else None

    def claim_user_extraction(
        self, user_id: str, owner: str, seconds: int
    ) -> str | None:
        """Manual/resume operations share automatic extraction's user lease."""
        with self._stream_sql() as db:
            self.lock_extraction_stream(user_id)
            token = uuid.uuid4().hex
            now = db.now()
            rows = db.query(
                f"UPDATE {db.table('learning_work')} SET lease_token=?,lease_owner=?,lease_until=? "
                "WHERE org_id=? AND user_id=? AND lease_until<=? RETURNING user_id",
                (token, owner, now + seconds, self.org_id, user_id, now),
            )
            return token if rows else None

    def release_user_extraction(self, user_id: str, token: str) -> None:
        with self._stream_sql() as db:
            db.query(
                f"UPDATE {db.table('learning_work')} SET lease_token=NULL,lease_until=0 "
                "WHERE org_id=? AND user_id=? AND lease_token=?",
                (self.org_id, user_id, token),
            )

    def defer_extraction_setup(self, user_id: str, token: str) -> None:
        with self._stream_sql() as db:
            db.query(
                f"UPDATE {db.table('learning_work')} SET due_at=? "
                "WHERE org_id=? AND user_id=? AND lease_token=?",
                (db.now() + 5, self.org_id, user_id, token),
            )

    def fence_user_extraction(self, user_id: str, token: str) -> None:
        with self._stream_sql() as db:
            self._fence(db, user_id, token)

    def validate_extraction_inputs(self, window: Window) -> None:
        with self._stream_sql() as db:
            expected = {m["interaction_id"]: m["seq"] for m in window.manifest}
            placeholders = ",".join("?" for _ in expected)
            if not expected:
                raise WindowInputsDeletedError("Window inputs were erased")
            rows = db.query(
                f"SELECT i.interaction_id,i.ingestion_seq FROM {db.table('interactions')} i "
                f"JOIN {db.table('requests')} r ON i.request_id=r.request_id AND i.user_id=r.user_id "
                f"WHERE i.user_id=? AND {db.project_filter()} AND i.interaction_id IN ({placeholders})"
                + db.input_lock(),
                (window.user_id, window.project_id, *expected),
            )
            if {
                row["interaction_id"]: row["ingestion_seq"] for row in rows
            } != expected:
                raise WindowInputsDeletedError("Window inputs were erased")

    def cleanup_extraction_work(self, user_id: str) -> None:
        """After erasure commits, remove the counter only if ALL scopes are gone.

        Cross-project discovery is read internally; the org/user row lock also
        serializes a concurrent new publish with this final cleanup.
        """
        with self._stream_sql(discovery=True) as db:
            db.query(
                f"SELECT user_id FROM {db.table('learning_work')} WHERE org_id=? AND user_id=?"
                + db.lock(),
                (self.org_id, user_id),
            )
            if not db.query(
                f"SELECT user_id FROM {db.table('extraction_cursors')} WHERE user_id=? LIMIT 1",
                (user_id,),
            ):
                db.query(
                    f"DELETE FROM {db.table('learning_work')} WHERE org_id=? AND user_id=?",
                    (self.org_id, user_id),
                )
