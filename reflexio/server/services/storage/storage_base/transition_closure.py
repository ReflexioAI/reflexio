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
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from reflexio.server.work_scope import current_project_id

from ._extraction_stream import StreamSQL, decoded
from .transition_inventory import (
    INVENTORY_SOURCES,
    TransitionInventoryScopeError,
    read_transition_inventory_page_db,
)


@dataclass(frozen=True)
class ProjectTransitionClosure:
    ready: bool
    blockers: tuple[str, ...]
    counts: dict[str, int] = field(default_factory=dict)


_MAX_PAGES = 1000


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
    for _ in range(_MAX_PAGES):
        boundary = (
            ""
            if after is None
            else f" AND ({','.join(keys)})>({','.join('?' for _ in keys)})"
        )
        rows = db.query(
            f"SELECT {columns} FROM {db.table(table)} r WHERE {predicate}{upper}{boundary} "
            f"ORDER BY {','.join(keys)} LIMIT 100",
            prefix_params + params + ceiling + (after or ()),
        )
        if not rows:
            return
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
    user: str,
    kind: str,
    seq: int,
    admitted: float,
    now: float,
    *,
    exact: bool = False,
) -> bool:
    rows = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('extraction_windows')} w "
        "WHERE w.org_id=? AND w.project_id=? AND w.user_id=? AND w.kind=? "
        "AND w.completed=1 AND w.invalidated=0 AND w.effects_done=1 "
        "AND w.predecessor>=0 AND w.predecessor<? "
        + ("AND w.end_seq=? " if exact else "AND w.end_seq>=? ")
        + "AND w.completed_at>=? AND w.completed_at<=?) AS proven",
        (org, project, user, kind, seq, seq, admitted, now),
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
                )
            except TransitionInventoryScopeError:
                blockers.add("source_attribution_unavailable")
                break
            counts[source] += len(page.items)
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
        if seq == 0 and row["started"] == 0:
            # A virgin counter cannot hide a previously created execution root.
            lineage = db.query(
                f"SELECT (EXISTS(SELECT 1 FROM {db.table('extraction_windows')} WHERE user_id=? AND kind=? AND project_id=?) OR "
                f"EXISTS(SELECT 1 FROM {db.table('_agent_runs')} r WHERE r.org_id=? AND r.user_id=? AND r.extractor_kind=? AND {db.project_filter()})) AS present",
                (user, kind, project_id, org_id, user, kind, project_id),
            )[0]["present"]
            valid = not lineage
        else:
            valid = (
                row["started"] == 1
                and seq > 0
                and _covered(
                    db, org_id, project_id, user, kind, seq, 0, now, exact=True
                )
            )
        if valid:
            cursors[(user, kind)] = seq
        else:
            blockers.add("cursor_completion_unproven")

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
    for row in _pages(
        db,
        "requests",
        f"r.user_id,r.request_id,r.learning_admission,{stats},"
        f"(SELECT MAX(u.highwater) FROM {db.table('learning_work')} u WHERE u.org_id=? AND u.user_id=r.user_id) AS highwater",
        db.project_filter(),
        (project_id,),
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
            and type(stamp) in {int, float}
            and math.isfinite(stamp)
            and 0 <= stamp <= now
            and type(highwater) is int
            and hi <= highwater
        )
        if valid:
            width = hi - lo + 1
            valid = row["n"] == row["distinct_n"] == width and (
                (width == 0 and row["lo"] is None and row["hi"] is None)
                or (width > 0 and row["lo"] == lo and row["hi"] == hi)
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
            if processed >= lo and not _covered(
                db, org_id, project_id, row["user_id"], kind, processed, stamp, now
            ):
                blockers.add("processed_admission_coverage_unproven")
            if hi > frontier:
                queued = True
        if queued:
            counts["queued_admissions"] += 1


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

    # Org/user allocation spans projects. Do not certify global queue closure,
    # but reject a counter whose actual retained source tail is absent anywhere.
    frontier = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('extraction_cursors')} c WHERE c.project_id=? AND NOT EXISTS(SELECT 1 FROM {db.table('learning_work')} u WHERE u.org_id=? AND u.user_id=c.user_id AND u.highwater>=c.completed_seq)) AS present",
        (project_id, org_id),
    )[0]["present"]
    if frontier:
        blockers.add("cursor_highwater_unproven")
    unknown = db.query(
        f"SELECT EXISTS(SELECT 1 FROM {db.table('learning_work')} u WHERE u.org_id=? AND (u.highwater<0 OR (u.highwater>0 AND "
        f"NOT EXISTS(SELECT 1 FROM {db.table('interactions')} i WHERE i.user_id=u.user_id AND i.ingestion_seq=u.highwater)))) AS present",
        (org_id,),
    )[0]["present"]
    if unknown:
        blockers.add("org_work_source_tail_unproven")
    for table in ("requests", "interactions"):
        if db.query(
            f"SELECT EXISTS(SELECT 1 FROM {db.table(table)} r WHERE {db.transition_unbound_project()}) AS present"
        )[0]["present"]:
            blockers.add("source_attribution_unavailable")
