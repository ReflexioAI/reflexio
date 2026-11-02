# ruff: noqa: S608 -- Fixed identifiers and dialect projections come from StreamSQL.
"""Narrow question expiry: metadata selection, then original-project APP mutation."""

from datetime import UTC, datetime
from typing import Any

from reflexio.server.work_scope import current_project_id

from ._user_work_position import ResumeCursor


class PendingToolExpiryStore:
    org_id: str
    _independent_writer_metadata_sql: Any
    commit_scope: Any
    _stream_sql: Any

    def _expired_question_page(
        self, *, cursor: ResumeCursor | None = None
    ) -> tuple[ResumeCursor, dict[str, Any] | None]:
        """Inspect one immutable-ID descriptor; a refusal also advances the sweep.

        This reads no question/answer payload or config, grants no mutation
        authority and takes no user lease. IDs above the fixed ceiling wait for wrap.
        """
        cursor = cursor or ResumeCursor()
        with self._independent_writer_metadata_sql() as db:
            now = datetime.fromtimestamp(db.now(), UTC).isoformat()
            table = db.table("_pending_tool_calls")
            condition = "q.org_id=? AND q.status='pending' AND q.expires_at<=? "
            params = (self.org_id, now)
            through = cursor.through_id
            if through is None:
                ceiling = db.query(
                    f"SELECT q.id FROM {table} q WHERE {condition}ORDER BY q.id DESC LIMIT 1",
                    params,
                )
                if not ceiling:
                    return ResumeCursor(), None
                through = ceiling[0]["id"]
            lower = "" if cursor.after_id is None else "AND q.id>? "
            bounds = (
                (through,) if cursor.after_id is None else (cursor.after_id, through)
            )
            rows = db.query(
                f"SELECT {db.expiry_question_columns()} FROM {table} q WHERE {condition}"
                f"{lower}AND q.id<=? ORDER BY q.id LIMIT 1",
                (*params, *bounds),
            )
            if not rows:
                return ResumeCursor(), None
            return ResumeCursor(rows[0]["id"], through), rows[0]

    def _expire_pending_question(self, question_id: str) -> bool:
        """Revalidate one pending question on the original-project APP writer.

        This is the narrow trusted dependency callback, not execution adoption:
        it cannot change executing run phases, attempts, ownership or output.
        Only this question's waiting dependents are projected to ready/terminal;
        unrelated runs and other orgs/projects are never swept as a side effect.
        """
        with self.commit_scope(), self._stream_sql() as db:
            now = datetime.fromtimestamp(db.now(), UTC).isoformat()
            project = current_project_id() or ""
            calls = db.table("_pending_tool_calls")
            changed = db.query(
                f"UPDATE {calls} AS q SET status='expired' "
                f"WHERE q.id=? AND q.org_id=? AND {db.project_filter('q')} "
                "AND q.status='pending' AND q.expires_at<=? RETURNING id",
                (question_id, self.org_id, project, now),
            )
            if not changed:
                return False
            dependencies, runs = (
                db.table("_run_tool_dependencies"),
                db.table("_agent_runs"),
            )
            db.query(
                f"UPDATE {dependencies} SET resolved_at=? "
                "WHERE pending_tool_call_id=? AND resolved_at IS NULL "
                f"AND run_id IN (SELECT r.id FROM {runs} r "
                f"WHERE r.org_id=? AND {db.project_filter()})",
                (now, question_id, self.org_id, project),
            )
            ordinary, actionable = db.resume_json_predicates()
            affected = (
                f"r.org_id=? AND {db.project_filter()} "
                f"AND r.status='finalized_pending_tool' AND COALESCE(({ordinary}),true) "
                f"AND EXISTS (SELECT 1 FROM {dependencies} changed "
                "WHERE changed.run_id=r.id AND changed.pending_tool_call_id=?) "
            )
            action = (
                f"EXISTS (SELECT 1 FROM {dependencies} d JOIN {calls} p "
                "ON p.id=d.pending_tool_call_id WHERE d.run_id=r.id "
                "AND d.resolved_at IS NOT NULL AND d.consumed_at IS NULL "
                f"AND p.status='resolved' AND {actionable})"
            )
            params = (now, self.org_id, project, question_id)
            db.query(
                f"UPDATE {runs} AS r SET status='resume_ready',updated_at=? "
                f"WHERE {affected}AND {action}",
                params,
            )
            db.query(
                f"UPDATE {runs} AS r SET status='finalized',updated_at=?,"
                "finalized_at=COALESCE(finalized_at,?) "
                f"WHERE {affected}AND NOT {action} "
                f"AND NOT EXISTS (SELECT 1 FROM {dependencies} d JOIN {calls} p "
                "ON p.id=d.pending_tool_call_id WHERE d.run_id=r.id "
                "AND d.resolved_at IS NULL AND d.consumed_at IS NULL AND p.status='pending')",
                (now, *params),
            )
            return True
