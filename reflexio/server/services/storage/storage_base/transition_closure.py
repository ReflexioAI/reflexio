# ruff: noqa: S608 -- Fixed adapter-owned tables and scope predicates only.
"""Project closure under the caller's existing locked control transaction.

Authentic queued intake is allowed; it is not unfinished generation. This does
not change the conservative raw inventory and does not prove whole-fleet or
org-wide queue closure. In particular learning_work.highwater spans projects;
the fleet owner must reconcile every project before releasing a fleet barrier.
No connection, commit, execution, callback or user-supplied proof is introduced.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from typing import Any

from reflexio.server.work_scope import current_project_id

from ._extraction_stream import StreamSQL, decoded
from .closed_window_history import closed_window_history_sql
from .transition_inventory import (
    INVENTORY_SOURCES,
    TransitionInventoryScopeError,
    read_transition_inventory_page_db,
)
from .transition_sources import finalized_run_candidate_sql, nonintake_request_sql


@dataclass(frozen=True)
class ProjectTransitionClosure:
    ready: bool
    blockers: tuple[str, ...]
    counts: dict[str, int] = field(default_factory=dict)


_MAX_PAGES = 1000
_INVENTORY_PAGE_SIZE = 500
# Four bind parameters per check plus the shared scope and clock.
_COVERAGE_BATCH_SIZE = 100
_CoverageCheck = tuple[str, str, int, float | None]


class _ObservationRefusedError(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _pages(
    db: StreamSQL,
    table: str,
    columns: str,
    predicate: str,
    params: tuple,
    keys: tuple[str, ...],
    *,
    prefix_params: tuple = (),
) -> Iterator[dict[str, Any]]:
    # NULL tuple keys are not ordered by SQL comparisons and can vanish below
    # the upper key or after cursor. They are unproven retained source, never an
    # empty source, even if the current schema normally forbids such keys.
    if db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table(table)} r WHERE {predicate} AND ("
        + " OR ".join(f"{key} IS NULL" for key in keys)
        + ")) AS present",
        params,
    )[0]["present"]:
        raise _ObservationRefusedError("source_key_unproven")
    # A fixed DB-ordered upper key prevents a steady stream of new higher IDs
    # extending this observation. Compare keys in SQL, never Python collation.
    top = db.query(
        f"SELECT {','.join(keys)} FROM {db.table(table)} r WHERE {predicate} "
        f"ORDER BY {','.join(key + ' DESC' for key in keys)} LIMIT 1",
        params,
    )
    if not top:
        return
    fields = tuple(key.split(".")[-1] for key in keys)
    ceiling = tuple(top[0][field] for field in fields)
    upper = f" AND ({','.join(keys)})<=({','.join('?' for _ in keys)})"
    after = None
    seen: set[tuple] = set()
    observed = 0
    for _ in range(_MAX_PAGES):
        boundary = (
            ""
            if after is None
            else f" AND ({','.join(keys)})>({','.join('?' for _ in keys)})"
        )
        rows = db.query(
            f"SELECT {columns} FROM {db.table(table)} r WHERE {predicate}{upper}{boundary} "
            f"ORDER BY {','.join(keys)} LIMIT {_INVENTORY_PAGE_SIZE}",
            prefix_params + params + ceiling + (after or ()),
        )
        if not rows:
            return
        observed += len(rows)
        if observed > 100 * _MAX_PAGES:
            raise _ObservationRefusedError("observation_limit_reached")
        for row in rows:
            key = tuple(row[field] for field in fields)
            if key in seen or key == after:
                raise _ObservationRefusedError("observation_did_not_advance")
            seen.add(key)
            yield row
        after = tuple(rows[-1][field] for field in fields)
    raise _ObservationRefusedError("observation_limit_reached")


def _covered(
    db: StreamSQL,
    org: str,
    project: str,
    checks: Sequence[_CoverageCheck],
    now: float,
    *,
    exact: bool = False,
) -> bool:
    if not checks:
        return True
    # Clockless batches can contain only SQL NULL clocks; PostgreSQL would
    # otherwise resolve the VALUES column as text before numeric comparison.
    values = ",".join("(?,?,?,CAST(? AS DOUBLE PRECISION))" for _ in checks)
    end_match = "=" if exact else ">="
    history = closed_window_history_sql(db, now=now)
    routed = f"(w.org_id=scope.org_id AND w.invalidated=0 AND w.completed_at IS NOT NULL) OR ({history.queue_floor})"
    if exact:
        routed += f" OR ({history.erased_floor})"
    # The original native admission format has no clock. Its numeric input
    # range can prove queue consumption, never latency or publication timing.
    timing = "((p.admitted_at IS NULL AND (w.completed_at IS NULL OR w.completed_at>=0)) OR w.completed_at>=p.admitted_at) AND (w.completed_at IS NULL OR w.completed_at<=?)"
    if exact:
        # An exact stored queue position has no publication-time claim.
        # NULL dates enter only through the shared positive history routes.
        timing = f"(w.completed_at IS NULL OR ({timing}))"
    rows = db.query(
        f"WITH pending_coverage(user_id,kind,seq,admitted_at) AS (VALUES {values}), "
        "scope AS (SELECT ? AS org_id,? AS project_id) "
        "SELECT NOT EXISTS(SELECT 1 FROM pending_coverage p WHERE NOT EXISTS("
        f"SELECT 1 FROM {db.table('extraction_windows')} w CROSS JOIN scope "
        f"WHERE ({routed}) AND w.project_id=scope.project_id AND w.user_id=p.user_id AND w.kind=p.kind "
        "AND w.completed=1 AND w.effects_done=1 "
        f"AND w.predecessor>=0 AND w.predecessor<p.seq AND w.end_seq{end_match}p.seq "
        f"AND {timing})) AS proven",
        (*chain.from_iterable(checks), org, project, now),
    )
    return bool(rows[0]["proven"])


def derive_project_transition_closure_db(
    db: StreamSQL,
    *,
    org_id: str,
    project_id: str,
) -> ProjectTransitionClosure:
    """Derive fixed source facts on this transaction, never treat errors as zero.

    The storage control owner calls this while holding its actual project gate.
    A ready result covers this project only. Org work counters are observed,
    never relabelled globally inert from a single project's successful proof.
    """
    if not org_id or project_id != (current_project_id() or ""):
        raise TransitionInventoryScopeError("Closure scope changed")
    db.learning_transition_project(project_id)
    blockers: set[str] = set()
    counts: dict[str, int] = {}
    now = db.now()
    try:
        _observe_sources(db, org_id, project_id, blockers, counts)
        cursors = _cursor_frontiers(db, org_id, project_id, now, blockers, counts)
        _admissions(db, org_id, project_id, now, cursors, blockers, counts)
        _source_sanity(db, org_id, project_id, blockers)
    except _ObservationRefusedError as error:
        blockers.add(error.reason)
    return ProjectTransitionClosure(not blockers, tuple(sorted(blockers)), counts)


def _observe_sources(
    db: StreamSQL,
    org_id: str,
    project_id: str,
    blockers: set[str],
    counts: dict[str, int],
) -> None:
    for source in INVENTORY_SOURCES:
        if source == "cursors":
            continue
        after = None
        counts[source] = 0
        seen: set[tuple[str, ...]] = set()
        for _ in range(_MAX_PAGES):
            try:
                page = read_transition_inventory_page_db(
                    db,
                    source,
                    org_id=org_id,
                    project_id=project_id,
                    after=after,
                    limit=_INVENTORY_PAGE_SIZE,
                )
            except TransitionInventoryScopeError:
                blockers.add("source_attribution_unavailable")
                break
            counts[source] += len(page.items)
            # Preserve the former 100-row x page-limit inventory ceiling while
            # using the native observer's existing larger bounded page.
            if counts[source] > 100 * _MAX_PAGES:
                raise _ObservationRefusedError("observation_limit_reached")
            for item in page.items:
                if item.key in seen:
                    raise _ObservationRefusedError("observation_did_not_advance")
                seen.add(item.key)
                if source == "work":
                    # Work is org/user-global, not project execution lineage.
                    # Record its backlog conservatively for final fleet reconciliation.
                    if item.live_owner:
                        blockers.add("live_work_owner")
                    if item.reasons:
                        counts["org_work_requiring_fleet_reconciliation"] = (
                            counts.get("org_work_requiring_fleet_reconciliation", 0) + 1
                        )
                elif item.execution_bearing is not False:
                    blockers.add(f"{source}_not_closed")
            if page.complete:
                break
            if page.next_after is None or page.next_after == after:
                raise _ObservationRefusedError("observation_did_not_advance")
            after = page.next_after
        else:
            raise _ObservationRefusedError("observation_limit_reached")


def _virgin_frontiers(
    db: StreamSQL, org: str, project: str, checks: Sequence[_CoverageCheck]
) -> bool:
    if not checks:
        return True
    values = ",".join("(?,?)" for _ in checks)
    project_scope = db.project_filter().replace("?", "scope.project_id")
    # _observe_sources already checks the full native execution/debt facts.
    # Historical generic finalization does not consume a window cursor; every
    # actual window or other run class still prevents a virgin proof.
    rows = db.query(
        f"WITH virgin_cursors(user_id,kind) AS (VALUES {values}), "
        "scope AS (SELECT ? AS org_id,? AS project_id) "
        "SELECT NOT EXISTS(SELECT 1 FROM virgin_cursors p CROSS JOIN scope WHERE "
        f"EXISTS(SELECT 1 FROM {db.table('extraction_windows')} w WHERE w.user_id=p.user_id "
        "AND w.kind=p.kind AND w.project_id=scope.project_id) OR "
        f"EXISTS(SELECT 1 FROM {db.table('_agent_runs')} r WHERE r.org_id=scope.org_id "
        "AND r.user_id=p.user_id AND r.extractor_kind=p.kind "
        f"AND ({project_scope}) AND ({finalized_run_candidate_sql(db)}) IS NOT TRUE)) AS proven",
        (
            *chain.from_iterable((user, kind) for user, kind, _, _ in checks),
            org,
            project,
        ),
    )
    return bool(rows[0]["proven"])


def _certify_frontiers(
    db: StreamSQL,
    org: str,
    project: str,
    now: float,
    checks: list[_CoverageCheck],
    frontiers: dict[tuple[str, str], int],
    blockers: set[str],
    *,
    virgin: bool,
) -> None:
    proven = (
        _virgin_frontiers(db, org, project, checks)
        if virgin
        else _covered(db, org, project, checks, now, exact=True)
    )
    if proven:
        frontiers.update(((user, kind), seq) for user, kind, seq, _ in checks)
    else:
        # One unproved member cannot certify any sibling in this bounded batch.
        blockers.add("cursor_completion_unproven")
    checks.clear()


def _cursor_frontiers(
    db: StreamSQL,
    org_id: str,
    project_id: str,
    now: float,
    blockers: set[str],
    counts: dict[str, int],
) -> dict[tuple[str, str], int]:
    cursor_scope = "r.project_id=?"
    if project_id:
        unknown = db.query(
            f"SELECT EXISTS(SELECT 1 FROM {db.table('extraction_cursors')} WHERE COALESCE(project_id,'')='') AS unknown"
        )
        if unknown[0]["unknown"]:
            blockers.add("cursor_attribution_unavailable")
    foreign = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('_agent_runs')} r WHERE {db.project_filter()} AND r.org_id<>?) AS unknown",
        (project_id, org_id),
    )
    if foreign[0]["unknown"]:
        blockers.add("source_attribution_unavailable")
    cursors: dict[tuple[str, str], int] = {}
    counts["cursors"] = 0
    virgin: list[_CoverageCheck] = []
    completed: list[_CoverageCheck] = []
    for row in _pages(
        db,
        "extraction_cursors",
        "r.user_id,r.kind,r.project_id,r.completed_seq,r.started,r.window_id,r.attempts,r.retry_at,r.last_error",
        cursor_scope,
        (project_id,),
        ("r.user_id", "r.kind", "r.project_id"),
    ):
        counts["cursors"] += 1
        user, kind, seq = row["user_id"], row["kind"], row["completed_seq"]
        clean = (
            bool(user)
            and kind in {"profile", "playbook"}
            and type(seq) is int
            and seq >= 0
            and row["window_id"] is None
            and row["attempts"] == 0
            and row["retry_at"] == 0
            and row["last_error"] is None
        )
        if not clean:
            blockers.add("cursor_execution_or_unknown_lineage")
            continue
        is_virgin = seq == 0 and row["started"] == 0
        if not is_virgin and not (row["started"] == 1 and seq > 0):
            blockers.add("cursor_completion_unproven")
            continue
        batch = virgin if is_virgin else completed
        batch.append((user, kind, seq, 0))
        if len(batch) == _COVERAGE_BATCH_SIZE:
            _certify_frontiers(
                db, org_id, project_id, now, batch, cursors, blockers, virgin=is_virgin
            )
    for batch, is_virgin in ((virgin, True), (completed, False)):
        if batch:
            _certify_frontiers(
                db, org_id, project_id, now, batch, cursors, blockers, virgin=is_virgin
            )

    return cursors


def _admissions(
    db: StreamSQL,
    org_id: str,
    project_id: str,
    now: float,
    cursors: dict[tuple[str, str], int],
    blockers: set[str],
    counts: dict[str, int],
) -> None:
    # Scope both sides, so a request cannot borrow another user's/project's inputs.
    input_scope = f"i.user_id=r.user_id AND i.request_id=r.request_id AND {db.project_filter('i')}"
    stats = ",".join(
        f"(SELECT {expression} FROM {db.table('interactions')} i WHERE {input_scope}) AS {name}"
        for expression, name in [
            ("COUNT(*)", "n"),
            ("COUNT(DISTINCT i.ingestion_seq)", "distinct_n"),
            ("MIN(i.ingestion_seq)", "lo"),
            ("MAX(i.ingestion_seq)", "hi"),
        ]
    )
    counts["admissions"] = counts["queued_admissions"] = 0
    coverage: list[_CoverageCheck] = []

    def prove_coverage() -> None:
        if not _covered(db, org_id, project_id, coverage, now):
            blockers.add("processed_admission_coverage_unproven")
        coverage.clear()

    # Request shells never entered durable intake, even when created today.
    # Exclude them before keyset paging so non-intake cannot exhaust its cap.
    # Invalid keys stay visible to _pages; malformed JSON is not SQL NULL.
    intake_scope = f"{db.project_filter()} AND NOT ({nonintake_request_sql(db, request_alias='r', project_sql='?')})"
    for row in _pages(
        db,
        "requests",
        f"r.user_id,r.request_id,r.learning_admission,{stats},"
        f"(SELECT MAX(u.highwater) FROM {db.table('learning_work')} u WHERE u.org_id=? AND u.user_id=r.user_id) AS highwater",
        intake_scope,
        (project_id, project_id),
        ("r.user_id", "r.request_id"),
        prefix_params=(project_id,) * 4 + (org_id,),
    ):
        counts["admissions"] += 1
        try:
            admission = decoded(row["learning_admission"])
        except (ValueError, TypeError):
            admission = None
        if not isinstance(admission, dict):
            blockers.add("admission_authentication_unproven")
            continue
        valid = all(
            isinstance(admission.get(kind), dict)
            and type(admission[kind].get("eligible")) is bool
            for kind in ("profile", "playbook")
        )
        if not valid:
            blockers.add("admission_authentication_unproven")
            continue
        lo: Any = admission.get("min_seq")
        hi: Any = admission.get("max_seq")
        stamp: Any = admission.get("admitted_at")
        highwater = row["highwater"]
        valid = (
            type(lo) is int
            and type(hi) is int
            and lo > 0
            and hi >= 0
            and hi >= lo - 1
            and (
                "admitted_at" not in admission
                or (
                    type(stamp) in {int, float}
                    and math.isfinite(stamp)
                    and 0 <= stamp <= now
                )
            )
            and type(highwater) is int
            and hi <= highwater
        )
        if valid:
            # Native deletion/retention leaves the immutable allocation range
            # intact. Authenticate every retained input, not erased content.
            valid = row["n"] == row["distinct_n"] and (
                (row["n"] == 0 and row["lo"] is None and row["hi"] is None)
                or (
                    row["n"] > 0
                    and row["lo"] is not None
                    and row["hi"] is not None
                    and lo <= row["lo"] <= row["hi"] <= hi
                )
            )
        if not valid:
            blockers.add("admission_authentication_unproven")
            continue
        queued = False
        for kind in ("profile", "playbook"):
            frontier = cursors.get((row["user_id"], kind))
            if frontier is None:
                blockers.add("admission_cursor_unproven")
                continue
            if not admission[kind]["eligible"] or hi < lo:
                continue
            processed = min(hi, frontier)
            if processed >= lo:
                coverage.append((row["user_id"], kind, processed, stamp))
                if len(coverage) == _COVERAGE_BATCH_SIZE:
                    prove_coverage()
            if hi > frontier:
                queued = True
        if queued:
            counts["queued_admissions"] += 1
    prove_coverage()


def _source_sanity(
    db: StreamSQL, org_id: str, project_id: str, blockers: set[str]
) -> None:
    # An interaction with no matching scoped request cannot disappear from proof.
    orphan = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('interactions')} i WHERE {db.project_filter('i')} AND NOT EXISTS(SELECT 1 FROM {db.table('requests')} r "
        f"WHERE r.user_id=i.user_id AND r.request_id=i.request_id AND {db.project_filter()})) AS present",
        (project_id, project_id),
    )[0]["present"]
    if orphan:
        blockers.add("orphan_source_input")

    # Allocation is monotonic across projects; retention/deletion can remove
    # its former tail without reusing those sequence numbers. Prove bounds on
    # retained inputs, never require a deleted tail or rewrite the counter.
    frontier = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('extraction_cursors')} c WHERE c.project_id=? AND NOT EXISTS(SELECT 1 FROM {db.table('learning_work')} u WHERE u.org_id=? AND u.user_id=c.user_id AND u.highwater>=c.completed_seq)) AS present",
        (project_id, org_id),
    )[0]["present"]
    if frontier:
        blockers.add("cursor_highwater_unproven")
    unknown = db.query(
        f"SELECT (EXISTS(SELECT 1 FROM {db.table('learning_work')} u WHERE u.org_id=? AND u.highwater<0) OR "
        f"EXISTS(SELECT 1 FROM {db.table('interactions')} i WHERE i.ingestion_seq IS NOT NULL AND "
        f"(i.ingestion_seq<=0 OR NOT EXISTS(SELECT 1 FROM {db.table('learning_work')} u "
        "WHERE u.org_id=? AND u.user_id=i.user_id AND u.highwater>=i.ingestion_seq)))) AS present",
        (org_id, org_id),
    )[0]["present"]
    if unknown:
        blockers.add("org_work_source_tail_unproven")
    for table in ("requests", "interactions"):
        if db.query(
            f"SELECT EXISTS(SELECT 1 FROM {db.table(table)} r WHERE {db.transition_unbound_project()}) AS present"
        )[0]["present"]:
            blockers.add("source_attribution_unavailable")
