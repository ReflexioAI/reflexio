# ruff: noqa: S608 -- Identifiers are fixed registered tables and projection names.
"""Content-free, exhaustive metadata pages for a quiesced policy transition.

These observations are neither execution authority nor a readiness receipt. A
caller must exhaust every source on every verified route, under writer
quiescence. Missing tables, failed pages and unknown witnesses must block that
proof. Runnable selectors are deliberately not used here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from reflexio.server.work_scope import current_project_id

from ._extraction_stream import StreamSQL

InventorySource = Literal[
    "runs", "windows", "cursors", "dependencies", "preparations", "receipts", "work"
]
InventoryReason = Literal[
    "unfinished_run",
    "unknown_state",
    "dependency",
    "preparation",
    "debt",
    "window",
    "overflow_requires_inspection",
    "unreceipted_output",
    "invalidated_window",
    "unfinished_window",
    "effects_debt",
    "unknown_route",
    "parked_cursor",
    "held_cursor",
    "unfinished_cursor",
    "ready_dependency",
    "waiting_dependency",
    "orphan_dependency",
    "active_preparation",
    "saved_preparation",
    "orphan_preparation",
    "financial_debt",
    "pending_admission",
    "unproven_admission",
    "live_owner",
    "unknown_project",
    "cursor_completion_unproven",
    "cursor_source_unproven",
    "uncovered_cursor_source",
    "cursor_execution_debt",
]
_RUN_FACTS: tuple[InventoryReason, ...] = (
    "dependency",
    "preparation",
    "debt",
    "window",
)
INVENTORY_SOURCES: tuple[InventorySource, ...] = (
    "runs",
    "windows",
    "cursors",
    "dependencies",
    "preparations",
    "receipts",
    "work",
)
_TERMINAL = {"finalized", "failed", "expired", "cancelled"}
_ACTIVE = {
    "running",
    "agent_completed",
    "finalizing",
    "finalized_pending_tool",
    "resume_ready",
    "resuming",
    "finalization_failed",
}
_UNKNOWN = {
    "unknown_state",
    "unknown_route",
    "unknown_project",
    "orphan_dependency",
    "orphan_preparation",
    "unreceipted_output",
    "overflow_requires_inspection",
    "unproven_admission",
    "cursor_completion_unproven",
    "cursor_source_unproven",
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SCHEMA = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,127}\Z")


@dataclass(frozen=True)
class TransitionInventoryItem:
    source: InventorySource
    key: tuple[str, ...]
    run_id: str | None
    user_id: str | None
    project_id: str | None
    revision: str | None
    # None is unavailable/insufficient evidence, never a zero-work observation.
    execution_bearing: bool | None
    reasons: tuple[InventoryReason, ...]
    state: str | None = None
    live_owner: bool = False
    saved_output: bool = False
    overflow: bool = False
    schema_name: str | None = None
    instruction_digest: str | None = None
    policy_digest: str | None = None
    configuration_digest: str | None = None
    window_id: str | None = None
    position: tuple[int, ...] = ()

    @property
    def witness_known(self) -> bool:
        return self.schema_name is not None and self.instruction_digest is not None


@dataclass(frozen=True)
class TransitionInventoryPage:
    items: tuple[TransitionInventoryItem, ...]
    next_after: tuple[str, ...] | None
    complete: bool


class TransitionInventoryScopeError(ValueError):
    """The storage route cannot attribute this source without exposing identifiers."""


@dataclass(frozen=True)
class TransitionInventoryAttribution:
    """Internal owner-produced binding, never a user-supplied verification claim.

    The enterprise discovery owner validates the complete roster, project and
    actual routes before and after a page. This value binds that operation to
    its exact storage writer; it grants neither execution nor fleet readiness.
    """

    org_id: str
    project_id: str
    route: object


def _attributed_cursors(
    db: StreamSQL,
    attribution: TransitionInventoryAttribution,
    after: tuple[str, ...] | None,
    limit: int,
) -> TransitionInventoryPage:
    """Observe retained cursor lineage in one statement, including empty pages."""
    if after is not None and (
        not isinstance(after, tuple)
        or len(after) != 3
        or not all(isinstance(value, str) for value in after)
    ):
        raise ValueError("Invalid transition inventory cursor")
    windows, cursors = db.table("extraction_windows"), db.table("extraction_cursors")
    requests, interactions = db.table("requests"), db.table("interactions")
    runs = db.table("_agent_runs")
    work = db.table("learning_work")
    now = db.now()
    # Reuse native eligibility/project/admission projections. SQLite has no
    # project-bearing source tables, and its adapter supplies the neutral scope.
    project = db.project_filter("q").replace("?", "scope.project_id")
    eligible = (
        "CASE r.kind "
        + " ".join(
            f"WHEN '{kind}' THEN ({db.eligible(kind).replace('r.', 'q.')})"
            for kind in ("profile", "playbook")
        )
        + " END"
    )
    maximum = db.admission_value("max_seq").replace("r.", "q.")
    minimum = db.admission_value("min_seq").replace("r.", "q.")
    admitted = db.admission_value("admitted_at").replace("r.", "q.")
    source = f"{requests} q WHERE q.user_id=r.user_id AND ({project})"
    input_project = db.project_filter("i").replace("?", "scope.project_id")
    input_scope = (
        f"i.user_id=q.user_id AND i.request_id=q.request_id AND ({input_project})"
    )
    matching = (
        "w.org_id=scope.org_id AND w.user_id=r.user_id AND w.kind=r.kind "
        "AND w.project_id=r.project_id AND w.completed=1 AND w.invalidated=0 "
        "AND w.effects_done=1"
        f" AND w.completed_at>=0 AND w.completed_at<={now!r} "
        "AND w.predecessor>=0 AND w.end_seq>w.predecessor"
    )
    # This is the acceptance-to-coverage predicate in learning_completion._page:
    # a cursor counter alone is insufficient without its completed-window receipt.
    coverage = (
        f"EXISTS(SELECT 1 FROM {windows} w WHERE {matching} "
        f"AND w.predecessor<{maximum} AND w.end_seq>={maximum} "
        f"AND w.completed_at>={admitted})"
    )
    _, _, _, run_columns = _projection(db, "runs", now)
    run_project = (
        db.project_filter().replace("r.", "a.").replace("?", "scope.project_id")
    )
    run_debt = (
        f"EXISTS(SELECT 1 FROM (SELECT {run_columns.replace('r.', 'a.')} "
        f"FROM {runs} a WHERE a.org_id=scope.org_id AND a.user_id=r.user_id "
        "AND a.extractor_kind=r.kind "
        f"AND ({run_project})) facts WHERE facts.status NOT IN ('finalized','failed','expired','cancelled') "
        "OR facts.dependency OR facts.preparation OR facts.debt OR facts.window "
        "OR (facts.saved_output AND NOT facts.published AND NOT facts.legacy_published AND NOT facts.window_published) "
        "OR (facts.overflow AND NOT facts.overflow_retired AND NOT facts.published AND NOT facts.window_published))"
    )
    unsafe = (
        # Full-source guard is not narrowed by the caller's after cursor.
        f"EXISTS(SELECT 1 FROM {cursors} c WHERE "
        "(c.project_id=scope.project_id OR c.project_id IS NULL) AND "
        "(c.user_id IS NULL OR c.kind IS NULL OR c.project_id IS NULL)) OR "
        f"EXISTS(SELECT 1 FROM {cursors} c WHERE COALESCE(c.project_id,'')='' "
        "AND scope.project_id<>'') OR "
        f"EXISTS(SELECT 1 FROM {runs} a WHERE a.org_id<>scope.org_id "
        f"AND ({run_project})) OR "
        f"EXISTS(SELECT 1 FROM {windows} w WHERE w.project_id=scope.project_id "
        "AND (w.org_id IS NULL OR w.org_id<>scope.org_id))"
    )
    boundary = (
        " AND (r.user_id,r.kind,r.project_id)>(?,?,?)" if after is not None else ""
    )
    statement = (
        "WITH scope AS (SELECT ? AS org_id,? AS project_id) "
        f"SELECT page.*,({unsafe}) AS inventory_unsafe FROM scope LEFT JOIN ("
        "SELECT r.user_id AS key0,r.kind AS key1,r.project_id AS key2,"
        "r.user_id,r.project_id,r.window_id,r.started,r.attempts,r.retry_at,"
        "r.last_error IS NOT NULL AS held,r.completed_seq,"
        f"EXISTS(SELECT 1 FROM {windows} w WHERE {matching} "
        "AND w.end_seq=r.completed_seq AND r.started=1) AS completion_proven,"
        f"EXISTS(SELECT 1 FROM {source} AND ({eligible}) AND {maximum}>r.completed_seq) AS uncovered_source,"
        f"EXISTS(SELECT 1 FROM {source} AND (q.learning_admission IS NULL OR ({eligible}) IS NULL "
        f"OR (({eligible}) AND ({maximum} IS NULL OR {minimum} IS NULL OR {admitted} IS NULL "
        f"OR {admitted}<0 OR {admitted}>{now!r} "
        f"OR {maximum}<{minimum} OR NOT ({coverage}) "
        f"OR (SELECT COUNT(*) FROM {interactions} i WHERE {input_scope})<>{maximum}-{minimum}+1 "
        f"OR (SELECT MAX(i.ingestion_seq) FROM {interactions} i WHERE {input_scope})<>{maximum})))) AS source_unproven,"
        f"({run_debt} OR EXISTS(SELECT 1 FROM {windows} w WHERE w.org_id=scope.org_id "
        "AND w.user_id=r.user_id AND w.kind=r.kind AND w.project_id=r.project_id "
        "AND (w.completed=0 OR w.effects_done=0))) AS execution_debt,"
        f"EXISTS(SELECT 1 FROM {work} u WHERE u.org_id=scope.org_id "
        f"AND u.user_id=r.user_id AND u.lease_until>{now!r}) AS live_owner "
        f"FROM {cursors} r CROSS JOIN scope WHERE r.project_id=scope.project_id{boundary} "
        "ORDER BY r.user_id,r.kind,r.project_id LIMIT ?) page ON 1=1 "
        "ORDER BY page.key0,page.key1,page.key2"
    )
    rows = db.query(
        statement, (attribution.org_id, attribution.project_id, *(after or ()), limit)
    )
    if not rows or any(row["inventory_unsafe"] for row in rows):
        raise TransitionInventoryScopeError(
            "Transition inventory source attribution is unavailable"
        )
    items = []
    previous = after
    # The matching SQL keyset predicate and ORDER BY own collation/type order.
    # Reject repeats without comparing database keys using Python's ordering.
    seen: set[tuple[str, ...]] = {after} if after is not None else set()
    for row in rows:
        if row["key0"] is None:
            continue
        key = tuple(str(row[f"key{i}"]) for i in range(3))
        if key in seen:
            raise ValueError("Transition inventory page did not advance")
        seen.add(key)
        items.append(_item("cursors", key, row))
        previous = key
    return TransitionInventoryPage(tuple(items), previous if items else None, not items)


def _ambiguity(
    db: StreamSQL, source: InventorySource, org_id: str, project: str
) -> tuple[str, tuple]:
    """Content-free full-source guard, evaluated in the same snapshot as the page."""
    runs = db.table("_agent_runs")
    unbound = db.transition_unbound_project()
    _, project_scope = db.transition_project_scope(source)
    if source == "cursors":
        return (
            f"EXISTS(SELECT 1 FROM {db.table('extraction_cursors')} r WHERE ({project_scope}) OR ({unbound}))",
            (project,),
        )
    if source == "dependencies":
        return (
            f"(EXISTS(SELECT 1 FROM {db.table('_run_tool_dependencies')} r WHERE NOT EXISTS(SELECT 1 FROM {runs} a WHERE a.id=r.run_id)) "
            f"OR EXISTS(SELECT 1 FROM {db.table('_run_tool_dependencies')} d JOIN {runs} r ON r.id=d.run_id WHERE r.org_id=? AND ({unbound})))",
            (org_id,),
        )
    if source == "windows":
        return (
            f"EXISTS(SELECT 1 FROM {db.table('extraction_windows')} r WHERE "
            f"(r.org_id IS NULL AND (({project_scope}) OR ({unbound}))) OR (r.org_id=? AND ({unbound})))",
            (project, org_id),
        )
    if source in {"runs", "preparations", "receipts"}:
        table = db.table(
            {
                "runs": "_agent_runs",
                "preparations": "_agent_output_preparations",
                "receipts": "_agent_output_receipts",
            }[source]
        )
        return f"EXISTS(SELECT 1 FROM {table} r WHERE r.org_id=? AND ({unbound}))", (
            org_id,
        )
    return "0=1", ()


def _witness(value: Any, pattern: re.Pattern[str]) -> str | None:
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _projection(db: StreamSQL, source: InventorySource, now: float) -> tuple:
    runs = db.table("_agent_runs")
    windows = db.table("extraction_windows")
    work = db.table("learning_work")
    deps = db.table("_run_tool_dependencies")
    prep = db.table("_agent_output_preparations")
    receipts = db.table("_agent_output_receipts")
    if source == "runs":
        return (
            runs,
            ("r.id",),
            "r.org_id=?",
            "r.user_id,r.id AS run_id,r.status,r.updated_at AS revision,r.extractor_config_hash AS configuration_digest,"
            "r.committed_output IS NOT NULL AS saved_output,"
            f"{db.transition_run_witness_columns()},"
            f"{db.transition_json_scalar('overflow_stage')} IN ('retired','invalidated') AS overflow_retired,"
            f"EXISTS(SELECT 1 FROM {receipts} p WHERE p.run_id=r.id AND p.org_id=r.org_id "
            f"AND p.output_id={db.transition_json_scalar('output_id')}) AS published,"
            f"(EXISTS(SELECT 1 FROM {db.table('_agent_run_finalization_receipts')} p WHERE p.run_id=r.id) "
            f"AND {db.transition_json_scalar('output_id')} IS NULL) AS legacy_published,"
            f"EXISTS(SELECT 1 FROM {windows} w WHERE w.window_id=substr(r.id,8) "
            "AND substr(r.id,1,7)='window:' AND r.status='finalized' AND w.org_id=r.org_id AND w.user_id=r.user_id "
            "AND w.kind=r.extractor_kind AND w.completed=1 AND w.invalidated=0 "
            "AND w.effects_done=1 AND w.derived_claimed=1 AND w.outcome IS NULL "
            f"AND w.completed_at>=0 AND w.completed_at<={now!r} "
            "AND w.predecessor>=0 AND w.end_seq>w.predecessor "
            f"AND ({db.transition_json_scalar('v1_window_publication')}) "
            f"AND NOT EXISTS(SELECT 1 FROM {receipts} p WHERE p.run_id=r.id AND p.org_id=r.org_id) "
            f"AND w.window_id NOT IN (SELECT c.window_id FROM {db.table('extraction_cursors')} c "
            "WHERE c.window_id IS NOT NULL)) AS window_published,"
            f"EXISTS(SELECT 1 FROM {work} u WHERE u.org_id=r.org_id AND "
            "(u.user_id=r.user_id OR u.lease_token=r.user_work_token) "
            f"AND u.lease_until>{now!r}) AS live_owner,"
            f"EXISTS(SELECT 1 FROM {deps} d WHERE d.run_id=r.id AND d.consumed_at IS NULL) AS dependency,"
            f"EXISTS(SELECT 1 FROM {prep} p WHERE p.run_id=r.id AND p.org_id=r.org_id "
            f"AND (p.active_attempt IS NOT NULL OR NOT EXISTS(SELECT 1 FROM {receipts} p2 "
            "WHERE p2.run_id=p.run_id AND p2.org_id=p.org_id "
            f"AND p2.output_id={db.transition_json_scalar('preparation_id').replace('r.', 'p.')}))) AS preparation,"
            f"EXISTS(SELECT 1 FROM {receipts} p WHERE p.run_id=r.id AND p.org_id=r.org_id AND p.delivery_state='pending') AS debt,"
            f"EXISTS(SELECT 1 FROM {windows} w WHERE r.id='window:' || w.window_id "
            "AND (w.org_id=r.org_id OR w.org_id IS NULL) AND (w.completed=0 OR w.effects_done=0)) AS window",
        )
    if source == "windows":
        return (
            windows,
            ("r.window_id",),
            "r.org_id=?",
            "r.user_id,r.window_id,r.predecessor,r.end_seq,'window:' || r.window_id AS run_id,r.completed,r.invalidated,"
            "r.effects_done,r.org_id IS NULL AS unknown_route,"
            f"EXISTS(SELECT 1 FROM {work} u WHERE u.org_id=? AND u.user_id=r.user_id AND u.lease_until>{now!r}) AS live_owner",
        )
    if source == "cursors":
        return (
            db.table("extraction_cursors"),
            ("r.user_id", "r.kind", "r.project_id"),
            # Without authoritative routing evidence no cursor identifier may
            # leave the metadata owner. The guard refuses a nonempty source.
            "0=1 AND ? IS NOT NULL",
            "r.user_id,r.window_id,r.started,r.attempts,"
            "r.retry_at,r.last_error IS NOT NULL AS held,r.completed_seq",
        )
    if source == "dependencies":
        return (
            deps,
            ("r.run_id", "r.pending_tool_call_id"),
            f"EXISTS(SELECT 1 FROM {runs} a WHERE a.id=r.run_id AND a.org_id=?)",
            "r.run_id,r.resolved_at IS NOT NULL AS resolved,"
            "r.consumed_at IS NOT NULL AS consumed,"
            f"NOT EXISTS(SELECT 1 FROM {runs} a WHERE a.id=r.run_id) AS orphan",
        )
    if source == "preparations":
        return (
            prep,
            ("r.run_id",),
            "r.org_id=?",
            "r.run_id,"
            "r.active_attempt IS NOT NULL AS active_attempt,"
            f"EXISTS(SELECT 1 FROM {receipts} p WHERE p.run_id=r.run_id AND p.org_id=r.org_id "
            f"AND p.output_id={db.transition_json_scalar('preparation_id')}) AS published,"
            f"NOT EXISTS(SELECT 1 FROM {runs} a WHERE a.id=r.run_id AND a.org_id=r.org_id) AS orphan",
        )
    if source == "receipts":
        return (
            receipts,
            ("r.receipt_id",),
            "r.org_id=?",
            "r.run_id,r.delivery_state='pending' AS debt",
        )
    if source == "work":
        return (
            work,
            ("r.user_id",),
            "r.org_id=?",
            "r.user_id,"
            "r.revision,r.pending_since,r.highwater,"
            "r.highwater>0 AS unproven_admission,"
            f"r.lease_until>{now!r} AS live_owner",
        )
    raise ValueError("Unknown transition inventory source")


def _run_reasons(row: dict) -> list[InventoryReason]:
    reasons: list[InventoryReason] = []
    if row["status"] in _ACTIVE:
        reasons.append("unfinished_run")
    elif row["status"] not in _TERMINAL:
        reasons.append("unknown_state")
    reasons.extend(fact for fact in _RUN_FACTS if row[fact])
    if (
        row["overflow"]
        and not row["overflow_retired"]
        and not row["published"]
        and not row["window_published"]
    ):
        reasons.append("overflow_requires_inspection")
    if (
        row["saved_output"]
        and not row["published"]
        and not row["legacy_published"]
        and not row["window_published"]
    ):
        reasons.append("unreceipted_output")
    return reasons


def _cursor_reasons(row: dict) -> list[InventoryReason]:
    reasons: list[InventoryReason] = []
    if row["window_id"] or row["started"] > row["completed_seq"] or row["held"]:
        reasons.append(
            "parked_cursor"
            if row["retry_at"] == float("inf")
            else "held_cursor"
            if row["held"]
            else "unfinished_cursor"
        )
    if "completion_proven" in row:
        facts: tuple[tuple[bool, InventoryReason], ...] = (
            (not row["completion_proven"], "cursor_completion_unproven"),
            (row["source_unproven"], "cursor_source_unproven"),
            (row["uncovered_source"], "uncovered_cursor_source"),
            (row["execution_debt"], "cursor_execution_debt"),
        )
        for condition, reason in facts:
            if condition:
                reasons.append(reason)
        if (row["attempts"] > 0 or row["retry_at"] > 0) and not reasons:
            reasons.append("unfinished_cursor")
    return reasons


def _item(
    source: InventorySource, key: tuple[str, ...], row: dict
) -> TransitionInventoryItem:
    reasons: list[InventoryReason] = []
    if source == "runs":
        reasons.extend(_run_reasons(row))
    elif source == "windows":
        if not row["completed"]:
            reasons.append(
                "invalidated_window" if row["invalidated"] else "unfinished_window"
            )
        if row["completed"] and not row["effects_done"]:
            reasons.append("effects_debt")
        if row["unknown_route"]:
            reasons.append("unknown_route")
    elif source == "cursors":
        reasons.extend(_cursor_reasons(row))
    elif source == "dependencies":
        if not row["consumed"]:
            reasons.append(
                "ready_dependency" if row["resolved"] else "waiting_dependency"
            )
        if row["orphan"]:
            reasons.append("orphan_dependency")
    elif source == "preparations":
        if row["active_attempt"] or not row["published"]:
            reasons.append(
                "active_preparation" if row["active_attempt"] else "saved_preparation"
            )
        if row["orphan"]:
            reasons.append("orphan_preparation")
    elif source == "receipts" and row["debt"]:
        reasons.append("financial_debt")
    elif source == "work":
        if row["pending_since"] > 0:
            reasons.append("pending_admission")
        if row["unproven_admission"]:
            reasons.append("unproven_admission")
    if row.get("live_owner") and (reasons or source in {"work", "cursors"}):
        reasons.append("live_owner")
    if (
        reasons
        and source != "work"
        and current_project_id()
        and not row.get("project_id")
    ):
        reasons.append("unknown_project")
    return TransitionInventoryItem(
        source=source,
        key=key,
        run_id=row.get("run_id"),
        user_id=row.get("user_id"),
        project_id=row.get("project_id"),
        revision=str(row["revision"]) if row.get("revision") is not None else None,
        execution_bearing=None if _UNKNOWN.intersection(reasons) else bool(reasons),
        reasons=tuple(reasons),
        state=row.get("status")
        if row.get("status")
        in {
            "running",
            "agent_completed",
            "finalizing",
            "finalized",
            "finalized_pending_tool",
            "resume_ready",
            "resuming",
            "finalization_failed",
            "failed",
            "expired",
            "cancelled",
        }
        else None,
        live_owner=bool(row.get("live_owner")),
        saved_output=bool(row.get("saved_output")),
        overflow=bool(row.get("overflow")),
        schema_name=_witness(row.get("schema_name"), _SCHEMA),
        instruction_digest=_witness(row.get("instruction_digest"), _DIGEST),
        policy_digest=_witness(row.get("policy_digest"), _DIGEST),
        configuration_digest=_witness(row.get("configuration_digest"), _DIGEST),
        window_id=row.get("window_id"),
        position=tuple(
            int(row[field])
            for field in {
                "windows": ("predecessor", "end_seq"),
                "cursors": ("completed_seq", "started", "attempts"),
                "work": ("highwater",),
            }.get(source, ())
        ),
    )


class TransitionInventoryStore:
    org_id: str
    _independent_writer_metadata_sql: Any
    _durable_work_route: Any

    def transition_inventory_page(
        self,
        source: InventorySource,
        *,
        after: tuple[str, ...] | None = None,
        limit: int = 100,
        attribution: TransitionInventoryAttribution | None = None,
    ) -> TransitionInventoryPage:
        """One bounded metadata page; only an empty page proves exhaustion.

        This uses the existing independent writer metadata owner. It does not acquire leases,
        inspect payloads or hide backend failures. No cross-page snapshot or
        fleet completeness is implied. `work` is org-wide ownership metadata;
        Org-less and legacy-unbound sources without proven attribution raise
        TransitionInventoryScopeError before returning any identifiers or an
        exhausted page. Nonempty cursors are unsupported until the routing
        owner can supply authoritative ownership. Nested APP transactions are refused.
        """
        if (
            source not in INVENTORY_SOURCES
            or type(limit) is not int
            or not 1 <= limit <= 500
        ):
            raise ValueError("Invalid transition inventory page")
        with self._independent_writer_metadata_sql() as db:
            if attribution is not None:
                if (
                    attribution.org_id != self.org_id
                    or attribution.project_id != (current_project_id() or "")
                    or attribution.route != self._durable_work_route()
                ):
                    raise TransitionInventoryScopeError(
                        "Transition inventory scope binding changed"
                    )
                if source == "cursors":
                    return _attributed_cursors(db, attribution, after, limit)
            return read_transition_inventory_page_db(
                db,
                source,
                org_id=self.org_id,
                project_id=current_project_id() or "",
                after=after,
                limit=limit,
            )


def read_transition_inventory_page_db(
    db: StreamSQL,
    source: InventorySource,
    *,
    org_id: str,
    project_id: str,
    after: tuple[str, ...] | None = None,
    limit: int = 100,
) -> TransitionInventoryPage:
    """Read through the caller's existing writer, without opening or committing.

    Used by the fixed closure owner while the project control is locked. It
    preserves the raw observer's conservative cursor/route interpretation; it
    grants neither routing authority nor a whole-fleet readiness receipt.
    """
    if (
        source not in INVENTORY_SOURCES
        or type(limit) is not int
        or not 1 <= limit <= 500
    ):
        raise ValueError("Invalid transition inventory page")
    if project_id != (current_project_id() or ""):
        raise TransitionInventoryScopeError("Transition inventory project changed")
    table, keys, scope, columns = _projection(db, source, db.now())
    if after is not None and (
        not isinstance(after, tuple)
        or len(after) != len(keys)
        or not all(isinstance(k, str) for k in after)
    ):
        raise ValueError("Invalid transition inventory cursor")
    params: tuple = (org_id,)
    if source == "windows":
        # Projection's live-owner placeholder precedes WHERE bindings.
        params = (org_id, org_id)
    project_column, project_scope = db.transition_project_scope(source)
    ambiguity, ambiguity_params = _ambiguity(db, source, org_id, project_id)
    project = f" AND {project_scope}"
    params += (project_id,)
    cursor_sql = ""
    if after is not None:
        cursor_sql = (
            f" AND ({','.join(keys)})>({','.join('?' for _ in keys)})"
            if len(keys) > 1
            else f" AND {keys[0]}>?"
        )
        params += after
    key_columns = ",".join(f"{key} AS key{i}" for i, key in enumerate(keys))
    # SQLite's runs/dependencies have no project_id; the adapter owns
    # that dialect distinction along with witness scalar projection.
    page_statement = (
        f"SELECT {key_columns},{project_column},{columns} FROM {table} r "
        f"WHERE {scope}{project}{cursor_sql} ORDER BY {','.join(keys)} LIMIT ?"
    )
    # A NULL key is retained corruption, not the outer join's empty-page row.
    # Check the full scoped source before keyset filtering: NULL comparisons can
    # otherwise hide it even when the caller's cursor is beyond every valid key.
    invalid_keys = (
        f"EXISTS(SELECT 1 FROM {table} r WHERE {scope}{project} AND ("
        + " OR ".join(f"{key} IS NULL" for key in keys)
        + "))"
    )
    guarded = db.query(
        f"SELECT page.*,(({ambiguity}) OR ({invalid_keys})) AS inventory_unsafe FROM (SELECT 1) guard "
        f"LEFT JOIN ({page_statement}) page ON 1=1 ORDER BY "
        + ",".join(f"page.key{i}" for i in range(len(keys))),
        (*ambiguity_params, org_id, project_id, *params, limit),
    )
    if not guarded or any(row["inventory_unsafe"] for row in guarded):
        raise TransitionInventoryScopeError(
            "Transition inventory source attribution is unavailable"
        )
    rows = [row for row in guarded if row["key0"] is not None]
    items = []
    previous = after
    # The matching SQL keyset predicate and ORDER BY own collation/type order.
    # Reject repeats without comparing database keys using Python's ordering.
    seen: set[tuple[str, ...]] = {after} if after is not None else set()
    for row in rows:
        key = tuple(str(row[f"key{i}"]) for i in range(len(keys)))
        if key in seen:
            raise ValueError("Transition inventory page did not advance")
        seen.add(key)
        items.append(_item(source, key, row))
        previous = key
    return TransitionInventoryPage(tuple(items), previous if rows else None, not rows)
