# ruff: noqa: S608 -- Identifiers come from the fixed stream adapter registry.
"""Bounded, content-free acceptance-to-coverage reporting.

The durable admission and completion receipts own the clock. Worker attempts,
billing and mutable cursor timestamps cannot stand in for those receipts.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING, Any

from reflexio.server.work_scope import current_project_id

if TYPE_CHECKING:
    from ._extraction_stream import StreamSQL

TARGET_SECONDS = 60
MAX_INTERVAL_SECONDS = 86_400


def _page(
    db: StreamSQL,
    start: float,
    end: float,
    after: tuple[float, str, str],
    limit: int,
) -> list[dict[str, Any]]:
    """One bounded statement per page, including receipt and cursor lookups."""
    admitted = db.admission_value("admitted_at")
    project = current_project_id() or ""
    projections = []
    parameters: list[Any] = [project, start, end, *after, limit, project, project]
    for kind in ("profile", "playbook"):
        projections.extend(
            (
                f"(SELECT w.completed_at FROM {db.table('extraction_windows')} w "
                "WHERE w.user_id=p.user_id AND w.project_id=? "
                f"AND w.kind='{kind}' AND w.completed=1 "
                "AND w.predecessor<p.surviving AND w.end_seq>=p.surviving "
                f"ORDER BY w.end_seq LIMIT 1) AS {kind}_completed_at",
                f"c_{kind}.completed_seq AS {kind}_completed_seq",
                f"c_{kind}.last_error AS {kind}_last_error",
                f"c_{kind}.retry_at AS {kind}_retry_at",
                f"c_{kind}.window_id AS {kind}_window_id",
            )
        )
        parameters.append(project)
    parameters.extend((project, project))
    return db.query(
        "WITH admissions AS ("
        f"SELECT r.request_id,r.user_id,{admitted} AS admitted_at,"
        f"{db.admission_value('min_seq')} AS min_seq,"
        f"{db.admission_value('max_seq')} AS max_seq,"
        f"CASE WHEN {db.eligible('profile')} THEN 1 ELSE 0 END AS profile,"
        f"CASE WHEN {db.eligible('playbook')} THEN 1 ELSE 0 END AS playbook "
        f"FROM {db.table('requests')} r WHERE {db.project_filter()} "
        f"AND {admitted}>=? AND {admitted}<? "
        f"AND ({admitted},r.request_id,r.user_id)>(?,?,?) "
        f"ORDER BY {admitted},r.request_id,r.user_id LIMIT ?), "
        "boundaries AS (SELECT a.*,"
        f"(SELECT MAX(i.ingestion_seq) FROM {db.table('interactions')} i "
        f"WHERE i.user_id=a.user_id AND i.request_id=a.request_id AND {db.project_filter('i')}) AS surviving,"
        f"(SELECT COUNT(*) FROM {db.table('interactions')} i "
        f"WHERE i.user_id=a.user_id AND i.request_id=a.request_id AND {db.project_filter('i')}) AS input_count "
        "FROM admissions a) "
        f"SELECT p.*,{','.join(projections)} FROM boundaries p "
        f"LEFT JOIN {db.table('extraction_cursors')} c_profile "
        "ON c_profile.user_id=p.user_id AND c_profile.project_id=? AND c_profile.kind='profile' "
        f"LEFT JOIN {db.table('extraction_cursors')} c_playbook "
        "ON c_playbook.user_id=p.user_id AND c_playbook.project_id=? AND c_playbook.kind='playbook' "
        "ORDER BY p.admitted_at,p.request_id,p.user_id",
        tuple(parameters),
    )


def _classify(row: dict[str, Any], observed_at: float) -> tuple[str, str]:
    required = [kind for kind in ("profile", "playbook") if row[kind]]
    if not required or row["max_seq"] < row["min_seq"]:
        return "excluded", "not_applicable"
    # Do not improve a historical result by deleting its unfinished tail.
    expected = row["max_seq"] - row["min_seq"] + 1
    if row["input_count"] != expected:
        return "excluded", "inputs_erased"
    unknown = False
    unfinished: list[str] = []
    completed = []
    for kind in required:
        stamp = row[f"{kind}_completed_at"]
        if stamp is not None:
            if not math.isfinite(stamp) or stamp < row["admitted_at"]:
                unknown = True
            elif stamp > observed_at:
                unfinished.append(kind)
            else:
                completed.append(stamp)
        elif (row[f"{kind}_completed_seq"] or 0) >= row["surviving"]:
            unknown = True
        else:
            unfinished.append(kind)
    if unfinished:
        # Reasons describe current diagnosis, not reconstructed historical state.
        errors = [row[f"{kind}_last_error"] or "" for kind in unfinished]
        if any(error.startswith("retry_exhausted:") for error in errors):
            reason = "retry_exhausted"
        elif any(error.startswith("input_budget") for error in errors):
            reason = "input_budget_failure"
        elif any(
            float(row[f"{kind}_retry_at"] or 0) > observed_at for kind in unfinished
        ):
            reason = "retrying"
        elif any(row[f"{kind}_window_id"] for kind in unfinished):
            reason = "active_window"
        elif any(row[f"{kind}_completed_seq"] is None for kind in unfinished):
            reason = "cursor_unavailable"
        else:
            reason = "awaiting_coverage"
        return "overdue_unfinished", reason
    if unknown:
        return "unknown", "completion_timestamp_unavailable"
    if max(completed) - row["admitted_at"] <= TARGET_SECONDS:
        return "on_time", "covered"
    return "completed_late", "covered"


def completion_report(
    db: StreamSQL,
    start: float,
    end: float,
    *,
    page_size: int,
    max_requests: int,
) -> dict[str, Any]:
    """Return a conservative rate for a mature cohort of retained admissions.

    Unknown completion times stay in the denominator. Erased/inapplicable rows
    are explicitly excluded. Missing admission timestamps cannot be assigned to
    a time cohort. Concurrent erasure/retention can remove entire records, so
    this is never an audit of deleted history or a repeatable-read snapshot.
    """
    observed_at = db.now()
    if (
        not all(math.isfinite(value) for value in (start, end))
        or start < 0
        or not 0 < end - start <= MAX_INTERVAL_SECONDS
        or end > observed_at - TARGET_SECONDS
    ):
        raise ValueError("Require a mature admission interval of at most 24 hours")
    if not 1 <= page_size <= 500 or not 1 <= max_requests <= 100_000:
        raise ValueError("Require page_size 1..500 and max_requests 1..100000")
    counts: Counter[str] = Counter(
        dict.fromkeys(
            ("on_time", "completed_late", "overdue_unfinished", "unknown", "excluded"),
            0,
        )
    )
    reasons: Counter[str] = Counter()
    after = (start - 1, "", "")
    seen: set[tuple[float, str, str]] = set()
    scanned = 0
    while rows := _page(
        db, start, end, after, min(page_size, max_requests - scanned + 1)
    ):
        if scanned + len(rows) > max_requests:
            raise RuntimeError(
                "Admission report exceeds max_requests; narrow the interval"
            )
        for row in rows:
            key = (float(row["admitted_at"]), row["request_id"], row["user_id"])
            # Text collation is owned by the DB. Equal timestamps still need
            # an explicit replay check; don't compare text using Python ordering.
            if key[0] < after[0] or key in seen:
                raise RuntimeError("Admission report pagination did not advance")
            seen.add(key)
            outcome, reason = _classify(row, observed_at)
            counts[outcome] += 1
            reasons[reason] += 1
            after = key
        scanned += len(rows)
    eligible = scanned - counts["excluded"]
    return {
        "scope": "retained_admissions_in_current_project",
        "start": start,
        "end": end,
        "observed_at": observed_at,
        "target_seconds": TARGET_SECONDS,
        "retained_admissions": scanned,
        "eligible_admissions": eligible,
        **counts,
        "one_minute_fraction_lower_bound": counts["on_time"] / eligible
        if eligible
        else None,
        "current_reasons": dict(sorted(reasons.items())),
    }
