# ruff: noqa: S608 -- Identifiers and dialect predicates come from the SQL adapter.
"""Bounded writer inspection for the two existing resumable-work classes.

Descriptors and sweep positions are metadata, never ownership or source proof.
Selection must advance private progress and adopt the exact run under its real
user lease in the subsequent APP writer transaction before any execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from reflexio.server.work_scope import current_project_id

from ._extraction_stream import StreamSQL
from ._user_work_position import ResumeClass, ResumeCursor, UserWorkPosition, WorkClass
from .agent_run._models import AgentRunRecord, AgentRunStatus

if TYPE_CHECKING:
    from . import BaseStorage


def resume_work_condition(
    db: StreamSQL,
    org_id: str,
    work_class: ResumeClass,
    *,
    user_id: str | None,
    token: str | None,
    now: float | None = None,
) -> tuple[str, tuple]:
    """Same eligibility on metadata inspection and fenced APP revalidation."""
    if now is None:
        now = db.now()
    clock = datetime.fromtimestamp(now, UTC).isoformat()
    ordinary, actionable = db.resume_json_predicates()
    work = db.table("learning_work")
    common = (
        f"r.org_id=? AND {ordinary} "
        "AND (r.next_resume_at IS NULL OR r.next_resume_at<=?) "
        "AND (r.user_work_token IS NULL OR r.user_work_token=? OR NOT EXISTS ("
        f"SELECT 1 FROM {work} owner WHERE owner.org_id=r.org_id "
        "AND owner.lease_token=r.user_work_token AND owner.lease_until>?)) "
        "AND (r.id NOT LIKE ? OR EXISTS ("
        f"SELECT 1 FROM {db.table('extraction_windows')} w "
        "WHERE w.window_id=substr(r.id,8) AND (w.org_id IS NULL OR w.org_id=r.org_id) "
        "AND w.completed=1 AND w.invalidated=0)) "
    )
    params: tuple = (org_id, clock, token, now, "window:%")
    if work_class == "finalization":
        common += (
            "AND r.status IN ('finalization_failed','finalizing','agent_completed') "
            "AND r.committed_output IS NOT NULL "
        )
    else:
        common += (
            "AND r.status IN ('resume_ready','resuming') AND EXISTS ("
            f"SELECT 1 FROM {db.table('_run_tool_dependencies')} d "
            f"JOIN {db.table('_pending_tool_calls')} p ON p.id=d.pending_tool_call_id "
            "WHERE d.run_id=r.id AND d.resolved_at IS NOT NULL "
            "AND d.consumed_at IS NULL AND p.status='resolved' "
            f"AND {actionable}) "
        )
    if user_id is not None:
        common += (
            "AND (r.user_id=? OR r.user_work_token=? OR ("
            "COALESCE(trim(r.user_id),'')='' AND r.extractor_kind='playbook')) "
        )
        params += (user_id, token)
    return common, params


@dataclass(frozen=True)
class DeferredWorkPresence:
    """One org's source facts, independent of admission and current config.

    Expiry belongs to housekeeping, not execution. None of these facts grants
    source or execution authority, and unavailable reads must raise, not say idle.
    """

    resume: bool
    financial: bool
    expiry: bool

    @property
    def needs_turn(self) -> bool:
        return self.resume or self.financial


def deferred_work_statement(
    db: StreamSQL, org_id: str, now: float
) -> tuple[str, tuple]:
    """One canonical fixed source projection for local and native fleet reads."""
    final, final_params = resume_work_condition(
        db, org_id, "finalization", user_id=None, token=None, now=now
    )
    continuation, continuation_params = resume_work_condition(
        db, org_id, "continuation", user_id=None, token=None, now=now
    )
    runs = db.table("_agent_runs")
    return (
        f"SELECT (EXISTS(SELECT 1 FROM {runs} r WHERE {final}) OR "
        f"EXISTS(SELECT 1 FROM {runs} r WHERE {continuation})) AS resume,"
        f"EXISTS(SELECT 1 FROM {db.table('_agent_output_receipts')} "
        "WHERE org_id=? AND delivery_state='pending') AS financial,"
        f"EXISTS(SELECT 1 FROM {db.table('_pending_tool_calls')} "
        "WHERE org_id=? AND status='pending' AND expires_at<=?) AS expiry LIMIT 1",
        (
            *final_params,
            *continuation_params,
            org_id,
            org_id,
            datetime.fromtimestamp(now, UTC).isoformat(),
        ),
    )


@dataclass(frozen=True)
class RunOwnerPresence:
    """Writer-backed diagnostic facts, never execution authority."""

    matches_revision: bool
    live: bool
    lease_until: float | None


class ResumeSelectionStore:
    org_id: str
    _independent_writer_metadata_sql: Any
    _require_independent_writer: Any
    commit_scope: Any
    _stream_sql: Any
    _owned_user_work_position: Any
    _store_owned_user_work_position: Any
    _adopt_owned_user_work_run: Any
    _fence: Any
    get_agent_run: Any

    def _deferred_work_presence(self) -> DeferredWorkPresence:
        """Read fixed, bounded source facts in one independent metadata query.

        One clock sample serves both recovery classes, finance-only debt and
        expiry-only housekeeping. No payload, attempt or configuration is read.
        This is org-scoped: fleet callers must use their verified source routes.
        """
        with self._independent_writer_metadata_sql() as db:
            now = db.now()
            statement, params = deferred_work_statement(db, self.org_id, now)
            rows = db.query(statement, params)
            if len(rows) != 1:
                raise ValueError("Deferred work discovery was incomplete")
            row = rows[0]
            return DeferredWorkPresence(
                resume=bool(row["resume"]),
                financial=bool(row["financial"]),
                expiry=bool(row["expiry"]),
            )

    def _run_owner_presence(
        self, expected: AgentRunRecord, *, executing_token: str | None = None
    ) -> RunOwnerPresence:
        from ._extraction_stream import _run_revision_time

        if expected.binding.org_id != self.org_id:
            raise ValueError("Run observation organization does not match storage")
        # Join a caller's APP transaction when present. A selected apply already
        # holds W: borrowing a second independent connection here could deadlock.
        with self._stream_sql(read_only=True) as db:
            now = db.now()
            work = db.table("learning_work")
            rows = db.query(
                "SELECT r.updated_at,r.status,r.claimed_by,r.claimed_at,EXISTS(SELECT 1 FROM "
                f"{work} u WHERE u.org_id=r.org_id AND u.user_id=r.user_id "
                "AND u.lease_until>? AND (? IS NULL OR u.lease_token<>?)) AS user_live,"
                f"EXISTS(SELECT 1 FROM {work} p WHERE p.org_id=r.org_id "
                "AND p.lease_token=r.user_work_token AND p.lease_until>? "
                "AND (? IS NULL OR p.lease_token<>?)) AS parent_live,"
                f"(SELECT u.lease_until FROM {work} u WHERE u.org_id=r.org_id "
                "AND u.user_id=r.user_id AND u.lease_until>? LIMIT 1) AS lease_until "
                f"FROM {db.table('_agent_runs')} r WHERE r.id=? AND r.org_id=? "
                f"AND {db.project_filter('r')} LIMIT 1",
                (
                    now,
                    executing_token,
                    executing_token,
                    now,
                    executing_token,
                    executing_token,
                    now,
                    expected.id,
                    self.org_id,
                    current_project_id() or "",
                ),
            )
            if not rows:
                return RunOwnerPresence(False, False, None)
            row = rows[0]
            return RunOwnerPresence(
                (
                    _run_revision_time(row["updated_at"]) == expected.updated_at
                    and row["status"] == expected.status.value
                    and row["claimed_by"] == expected.claimed_by
                    and _run_revision_time(row["claimed_at"]) == expected.claimed_at
                ),
                bool(row["user_live"] or row["parent_live"]),
                float(row["lease_until"]) if row["lease_until"] is not None else None,
            )

    def claim_agent_run_for_recovery(
        self,
        *,
        expected: AgentRunRecord,
        worker_id: str,
        operation_id: str,
    ) -> AgentRunRecord | None:
        """Adopt exactly one selected finalization under its real user owner."""
        from reflexio.server.services.durable_learning.user_lease import (
            _require_user_work_owner,
        )

        from . import LeaseLostError
        from ._extraction_stream import _run_revision_time, decoded

        run_id, org_id = expected.id, expected.binding.org_id
        if org_id != self.org_id:
            raise ValueError("Recovery organization does not match storage")
        if not worker_id or not operation_id:
            raise ValueError("Recovery requires operation and worker")
        if expected.committed_output is None or expected.pending_tool_call_ids:
            return None
        storage = cast("BaseStorage", self)
        owner = _require_user_work_owner(storage)
        with self.commit_scope(), self._stream_sql() as db:
            owner.fence(storage)
            if (expected.project_id or "") != (current_project_id() or ""):
                return None
            condition, params = resume_work_condition(
                db, org_id, "finalization", user_id=owner.user_id, token=owner.token
            )
            rows = db.query(
                "SELECT r.updated_at,r.status,r.claimed_by,r.claimed_at,r.pending_tool_call_ids "
                f"FROM {db.table('_agent_runs')} r WHERE r.id=? "
                f"AND {db.project_filter('r')} AND {condition} "
                f"AND NOT EXISTS(SELECT 1 FROM {db.table('_run_tool_dependencies')} d "
                "WHERE d.run_id=r.id AND d.consumed_at IS NULL)" + db.lock(),
                (run_id, current_project_id() or "", *params),
            )
            owner.fence(storage)
            if not rows or (
                _run_revision_time(rows[0]["updated_at"]) != expected.updated_at
                or rows[0]["status"] != expected.status.value
                or rows[0]["claimed_by"] != expected.claimed_by
                or _run_revision_time(rows[0]["claimed_at"]) != expected.claimed_at
            ):
                return None
            pending = decoded(rows[0]["pending_tool_call_ids"])
            if pending is not None and (not isinstance(pending, list) or pending):
                return None
            claim = self._adopt_owned_user_work_run(
                db,
                expected=expected,
                user_id=owner.user_id,
                token=owner.token,
                worker_id=worker_id,
            )
            if claim is None:
                raise LeaseLostError("Agent execution adoption was refused")
            changed = db.query(
                f"UPDATE {db.table('_agent_runs')} SET status='finalizing',"
                "recovery_operation_id=? WHERE id=? AND org_id=? "
                "AND claimed_by=? AND claimed_at=? RETURNING id",
                (operation_id, run_id, org_id, claim[0], claim[1].isoformat()),
            )
            if not changed:
                raise LeaseLostError("Selected recovery lost its adopted revision")
            owner.fence(storage)
            return self.get_agent_run(run_id)

    def _resume_candidate_page(
        self,
        work_class: ResumeClass,
        *,
        cursor: ResumeCursor | None = None,
        user_id: str | None = None,
        token: str | None = None,
    ) -> tuple[ResumeCursor, dict[str, Any] | None]:
        """Inspect at most one descriptor; never mutate claims or run revisions.

        A user-local sweep sees direct bindings, its current parent token, and
        nullable legacy playbooks whose parent is no longer live. Complete source
        proof still decides adoption. Invalid/busy candidates must advance the
        caller's cursor too. A fixed upper boundary defers newer IDs until wrap.
        A failed read raises; only a successful empty page resets the sweep.
        """
        if cursor is None:
            cursor = ResumeCursor()
        if work_class not in ("finalization", "continuation"):
            raise ValueError("Unknown resumable work class")
        if (user_id is None) != (token is None) or (
            user_id is not None and (not user_id.strip() or not token)
        ):
            raise ValueError("Owner-local inspection requires a user and token")
        with self._independent_writer_metadata_sql() as db:
            common, params = resume_work_condition(
                db, self.org_id, work_class, user_id=user_id, token=token
            )
            runs = db.table("_agent_runs")
            through = cursor.through_id
            if through is None:
                ceiling = db.query(
                    f"SELECT r.id FROM {runs} r WHERE {common} ORDER BY r.id DESC LIMIT 1",
                    params,
                )
                if not ceiling:
                    return ResumeCursor(), None
                through = ceiling[0]["id"]
            lower = "" if cursor.after_id is None else "AND r.id>? "
            page_params = params + (
                (through,) if cursor.after_id is None else (cursor.after_id, through)
            )
            rows = db.query(
                f"SELECT {db.erasure_run_columns()},r.user_id FROM {runs} r "
                f"WHERE {common}{lower}AND r.id<=? ORDER BY r.id LIMIT 1",
                page_params,
            )
            if not rows:
                return ResumeCursor(), None
            row = rows[0]
            return ResumeCursor(row["id"], through), row

    def _observe_live_user_work_parent(
        self, *, expected: AgentRunRecord, user_id: str
    ) -> bool:
        """Attach only a private dependency to an inspected, busy run revision.

        This is NOT adoption, a claim or execution/source authority. The caller
        resolved the original project and user before a canonical claim refused.
        A source/run mutation refuses this observation; a release or replacement
        makes the recorded token eligible immediately through actual liveness.
        """
        from ._extraction_stream import _run_revision_time

        self._require_independent_writer()
        if (
            expected.binding.org_id != self.org_id
            or expected.updated_at is None
            or not user_id.strip()
            or (
                expected.binding.user_id
                and expected.binding.user_id.strip()
                and expected.binding.user_id != user_id
            )
        ):
            return False
        with self._stream_sql(discovery=True) as db:
            # Same W-before-run lock order as real adoption, with NOWAIT rather
            # than waiting behind a provider or another user's source mutation.
            parents = db.query(
                f"SELECT lease_token FROM {db.table('learning_work')} "
                "WHERE org_id=? AND user_id=? AND lease_token IS NOT NULL "
                "AND lease_until>?" + db.validation_lock(),
                (self.org_id, user_id, db.now()),
            )
            if not parents:
                return False
            token = parents[0]["lease_token"]
            runs = db.table("_agent_runs")
            rows = db.query(
                "SELECT r.user_id,r.extractor_kind,r.status,r.updated_at,"
                f"r.claimed_by,r.claimed_at,r.user_work_token FROM {runs} r "
                f"WHERE r.org_id=? AND r.id=? AND {db.project_filter()}"
                + db.validation_lock(),
                (self.org_id, expected.id, current_project_id() or ""),
            )
            if not rows:
                return False
            row = rows[0]
            if (
                row["user_id"] != expected.binding.user_id
                or row["extractor_kind"] != expected.binding.extractor_kind
                or row["status"] != expected.status.value
                or _run_revision_time(row["updated_at"]) != expected.updated_at
                or row["claimed_by"] != expected.claimed_by
                or _run_revision_time(row["claimed_at"]) != expected.claimed_at
            ):
                return False
            parent = row["user_work_token"]
            if (
                parent is not None
                and parent != token
                and db.query(
                    f"SELECT user_id FROM {db.table('learning_work')} "
                    "WHERE org_id=? AND lease_token=? AND lease_until>? LIMIT 1",
                    (self.org_id, parent, db.now()),
                )
            ):
                return False
            db.query(
                f"UPDATE {runs} AS r SET user_work_token=? "
                f"WHERE r.org_id=? AND r.id=? AND {db.project_filter()}",
                (token, self.org_id, expected.id, current_project_id() or ""),
            )
            return True

    def _read_owned_work_position(self, user_id: str, token: str) -> UserWorkPosition:
        """Read under the live user fence, without retaining a transaction."""
        with self._stream_sql(coordination=True) as db:
            return self._owned_user_work_position(db, user_id, token)

    def _commit_owned_work_selection(
        self,
        *,
        user_id: str,
        token: str,
        expected_position: UserWorkPosition,
        work_class: WorkClass,
        cursor: ResumeCursor | None = None,
        expected_run: AgentRunRecord | None = None,
        worker_id: str,
    ) -> tuple[bool, AgentRunRecord | None]:
        """Consume an opportunity and, when valid, adopt in the SAME APP writer.

        False means another inspector already changed this private position.
        True/None includes empty or refused inspections; no run revision or
        attempt is charged. Statement failures roll back ALL changes. A lost
        commit reply grants no execution authority; the next position CAS must
        recognize any already-committed progress. A caller may consume a busy
        or unreadable descriptor in a fresh fenced transaction with
        expected_run=None after the failed writer has completely rolled back.
        """
        from reflexio.server.services.durable_learning.user_lease import (
            _require_user_work_owner,
        )

        from ._extraction_stream import LeaseLostError

        storage = cast("BaseStorage", self)
        owner = _require_user_work_owner(storage)
        if owner.user_id != user_id or owner.token != token:
            raise LeaseLostError("Selection belongs to a different user-work owner")
        next_position = expected_position.selected(work_class, cursor=cursor)
        if work_class == "learning" and expected_run is not None:
            raise ValueError("Learning selection cannot adopt a resumable run")
        with (
            self.commit_scope(),
            self._stream_sql(coordination=expected_run is None) as db,
        ):
            if self._owned_user_work_position(db, user_id, token) != expected_position:
                return False, None
            owner.fence(storage)
            self._store_owned_user_work_position(db, user_id, token, next_position)
            if expected_run is None:
                owner.fence(storage)
                return True, None
            # The original project is already bound by the caller. Re-evaluate
            # due/dependency/window/parent eligibility before adopting any row.
            common, params = resume_work_condition(
                db,
                self.org_id,
                cast(ResumeClass, work_class),
                user_id=user_id,
                token=token,
            )
            rows = db.query(
                f"SELECT r.id FROM {db.table('_agent_runs')} r WHERE {common} "
                f"AND r.id=? AND {db.project_filter()}" + db.validation_lock(),
                (*params, expected_run.id, current_project_id() or ""),
            )
            if not rows:
                owner.fence(storage)
                return True, None
            claim = self._adopt_owned_user_work_run(
                db,
                expected=expected_run,
                user_id=user_id,
                token=token,
                worker_id=worker_id,
            )
            if claim is None:
                owner.fence(storage)
                return True, None
            status = (
                AgentRunStatus.FINALIZING
                if work_class == "finalization"
                else AgentRunStatus.RESUMING
            )
            db.query(
                f"UPDATE {db.table('_agent_runs')} SET status=? WHERE org_id=? AND id=? "
                "AND claimed_by=? AND claimed_at=?",
                (
                    status.value,
                    self.org_id,
                    expected_run.id,
                    claim[0],
                    claim[1].isoformat(),
                ),
            )
            current = self.get_agent_run(expected_run.id)
            if (
                current is None
                or current.status != status
                or (current.claimed_by, current.claimed_at) != claim
            ):
                raise LeaseLostError("Adopted user work revision is unavailable")
            owner.fence(storage)
            return True, current
