"""Fixed native facts for detached closed windows, never publication proof.

The caller authenticates its physical namespace and project scope. All SQL
identifiers originate in its native adapter; these fragments grant no routing,
financial, or provider authority and introduce no caller-selectable policy.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .transition_sources import detached_window_sql, live_work_sql

if TYPE_CHECKING:
    from ._extraction_stream import StreamSQL


@dataclass(frozen=True)
class ClosedWindowHistorySQL:
    inert: str
    queue_floor: str
    erased_floor: str


def closed_window_history_sql(
    db: StreamSQL,
    *,
    org_sql: str = "scope.org_id",
    project_sql: str = "scope.project_id",
    now: float,
) -> ClosedWindowHistorySQL:
    """Return fixed booleans for window alias w and authenticated SQL scope.

    Scope expressions are source-owned column references, not arbitrary caller
    predicates. A roster owner may use w.project_id only after independently
    proving that project belongs to its actual organization and physical route.
    """
    if (
        not re.fullmatch(r"[a-z][a-z_]*\.org_id", org_sql)
        or not (
            project_sql == "''"
            or re.fullmatch(r"[a-z][a-z_]*\.project_id", project_sql)
        )
        or type(now) not in (int, float)
        or not math.isfinite(now)
        or now < 0
    ):
        raise ValueError("Closed window history scope is invalid")
    ack, retired = db.transition_window_history_format()
    no_billing = db.transition_window_history_no_billing()
    valid_billing = db.transition_window_billing_descriptor_valid()
    idle_parent = db.transition_window_history_parent_idle().replace("a.", "h.")
    runs = db.table("_agent_runs")
    cursors = db.table("extraction_cursors")
    dependencies = db.table("_run_tool_dependencies")
    preparations = db.table("_agent_output_preparations")
    receipts = db.table("_agent_output_receipts")
    parent_id = "'window:'||w.window_id"
    parent_project = (
        db.transition_project_scope("runs")[0]
        .removesuffix(" AS project_id")
        .replace("r.", "h.")
    )
    parent_bound = (
        f"h.id={parent_id} AND h.org_id={org_sql} AND h.user_id=w.user_id "
        f"AND h.extractor_kind=w.kind AND {parent_project}=w.project_id"
    )
    past_cursor = (
        f"EXISTS(SELECT 1 FROM {cursors} c WHERE c.user_id=w.user_id AND c.kind=w.kind "  # noqa: S608 -- Fixed registered table and source-owned aliases.
        "AND c.project_id=w.project_id AND c.started=1 AND c.completed_seq>=w.end_seq)"
    )
    common = (
        f"w.project_id={project_sql} AND (w.org_id={org_sql} OR w.org_id IS NULL) "  # noqa: S608 -- Fixed adapter tables and validated source column references.
        "AND length(trim(w.user_id))>0 AND w.kind IN ('profile','playbook') "
        "AND w.completed=1 AND w.effects_done=1 AND w.outcome IS NULL AND w.manifest='[]' "
        f"AND (w.completed_at IS NULL OR (w.completed_at>=0 AND w.completed_at<={now!r})) "
        "AND w.predecessor>=0 AND w.end_seq>w.predecessor "
        f"AND ({detached_window_sql(db)}) "
        f"AND NOT ({live_work_sql(db, alias='w', org_sql=org_sql, now=now)}) "
        f"AND NOT EXISTS(SELECT 1 FROM {dependencies} d WHERE d.run_id={parent_id} AND d.consumed_at IS NULL) "
        f"AND NOT EXISTS(SELECT 1 FROM {preparations} p WHERE p.run_id={parent_id}) "
        f"AND {parent_id} NOT IN(SELECT p.run_id FROM {receipts} p WHERE p.run_id IS NOT NULL "
        "AND (p.row_kind='output' OR p.delivery_state='pending')) "
        f"AND w.window_id NOT IN(SELECT p.delivery_key FROM {receipts} p "
        "WHERE p.delivery_key IS NOT NULL AND p.row_kind='window_handoff')"
    )
    acknowledged = (
        f"({ack}) AND w.invalidated=0 AND w.derived_claimed IN (0,1) "  # noqa: S608 -- Fixed adapter tables and validated source column references.
        "AND (((w.org_id IS NULL OR w.completed_at IS NULL) "
        f"AND EXISTS(SELECT 1 FROM {runs} h WHERE {parent_bound} AND ({idle_parent}) "
        f"AND ((h.committed_output IS NULL AND ({no_billing})) OR ({valid_billing})) "
        "AND (h.status IN ('finalized','failed','cancelled') OR (h.status='expired' AND h.committed_output IS NULL))) "
        ") "
        f"OR (w.derived_claimed=1 AND ({no_billing}) AND ({past_cursor}) "
        f"AND NOT EXISTS(SELECT 1 FROM {runs} h WHERE h.id={parent_id})))"
    )
    retired_history = (
        f"({retired}) AND w.invalidated=0 AND w.derived_claimed=1 "  # noqa: S608 -- Fixed adapter tables and validated source column references.
        f"AND ({past_cursor}) "
        f"AND (NOT EXISTS(SELECT 1 FROM {runs} h WHERE h.id={parent_id}) OR "
        f"EXISTS(SELECT 1 FROM {runs} h WHERE {parent_bound} AND ({idle_parent}) "
        "AND h.status='cancelled' AND h.committed_output IS NULL))"
    )
    erased_history = (
        f"({ack}) AND w.invalidated=1 AND w.derived_claimed IN (0,1) "  # noqa: S608 -- Fixed adapter tables and validated source column references.
        f"AND EXISTS(SELECT 1 FROM {runs} h WHERE {parent_bound} AND ({idle_parent}) "
        "AND h.status='cancelled' AND h.claimed_by IS NULL AND h.claimed_at IS NULL)"
    )
    return ClosedWindowHistorySQL(
        f"({common} AND (({acknowledged}) OR ({retired_history}) OR ({erased_history})))",
        f"({common} AND (({acknowledged}) OR ({retired_history})))",
        f"({common} AND ({erased_history}))",
    )
