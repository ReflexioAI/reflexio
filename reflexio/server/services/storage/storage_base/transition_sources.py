"""Intrinsic source predicates shared by native transition observations."""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ._extraction_stream import StreamSQL


def nonintake_request_sql(
    db: StreamSQL, *, request_alias: str, project_sql: str
) -> str:
    """A SQL-NULL shell has no sequenced input in its actual owned identity."""
    if request_alias not in {"r", "q"} or not (
        project_sql == "?" or re.fullmatch(r"[a-z][a-z_]*\.project_id", project_sql)
    ):
        raise ValueError("Non-intake source scope is invalid")
    inputs = db.table("interactions")
    project = db.project_filter("i").replace("?", project_sql)
    return (
        f"({request_alias}.user_id IS NOT NULL AND {request_alias}.request_id IS NOT NULL "  # noqa: S608 -- Fixed registered table and validated source aliases.
        f"AND {request_alias}.learning_admission IS NULL AND NOT EXISTS("
        f"SELECT 1 FROM {inputs} i WHERE i.user_id={request_alias}.user_id "
        f"AND i.request_id={request_alias}.request_id AND ({project}) AND i.ingestion_seq IS NOT NULL))"
    )


def generic_closed_run_candidate_sql(db: StreamSQL) -> str:
    """Structural class only; native execution/debt facts must also be clear."""
    idle = db.transition_window_history_parent_idle().replace("a.", "r.")
    no_overflow, _ = db.resume_json_predicates()
    return (
        f"((r.status='finalized' OR (r.status='cancelled' AND ({no_overflow}))) "
        f"AND substr(r.id,1,7)<>'window:' AND ({idle}))"
    )


RUN_EXECUTION_FACTS = ("live_owner", "dependency", "preparation", "debt", "window")


def generic_closed_run_inert_sql(*, alias: str = "facts") -> str:
    """Derive history from already observed native facts, without more queries."""
    if alias != "facts":
        raise ValueError("Finalized history fact alias is invalid")
    return (
        "("
        + f"{alias}.terminal_history IS TRUE AND "
        + " AND ".join(f"{alias}.{fact} IS FALSE" for fact in RUN_EXECUTION_FACTS)
        + ")"
    )


def generic_closed_run_inert(facts: dict) -> bool:
    terminal = facts.get("terminal_history")
    return (terminal is True or (type(terminal) is int and terminal == 1)) and all(
        facts[fact] is False or (type(facts[fact]) is int and facts[fact] == 0)
        for fact in RUN_EXECUTION_FACTS
    )


def detached_window_sql(db: StreamSQL) -> str:
    """Every cursor pointer counts, including foreign project/user pointers."""
    return (
        f"w.window_id NOT IN(SELECT c.window_id FROM {db.table('extraction_cursors')} c "  # noqa: S608 -- Fixed registered table and native aliases.
        "WHERE c.window_id IS NOT NULL)"
    )


def live_work_sql(
    db: StreamSQL, *, alias: Literal["r", "w"], org_sql: str, now: float
) -> str:
    """Uncorrelated native live-owner sets retain both subject and token fences."""
    if (
        alias not in {"r", "w"}
        or not re.fullmatch(r"[a-z][a-z_]*\.org_id", org_sql)
        or type(now) not in {int, float}
        or not math.isfinite(now)
        or now < 0
    ):
        raise ValueError("Live work fact scope is invalid")
    work = db.table("learning_work")
    token = (
        "r.user_work_token"
        if alias == "r"
        else f"(SELECT h.user_work_token FROM {db.table('_agent_runs')} h WHERE h.id='window:'||w.window_id)"  # noqa: S608 -- Fixed registered parent identity.
    )
    # IS TRUE preserves EXISTS' false result for unknown/NULL join identities.
    return (
        f"((({org_sql},{alias}.user_id) IN(SELECT u.org_id,u.user_id FROM {work} u WHERE u.lease_until>{now!r}) "  # noqa: S608 -- Validated aliases and fixed registered table.
        f"OR ({org_sql},{token}) IN(SELECT u.org_id,u.lease_token FROM {work} u "
        f"WHERE u.lease_until>{now!r} AND u.lease_token IS NOT NULL)) IS TRUE)"
    )


def historical_ack_names_request_sql(
    db: StreamSQL, request_sql: Literal["p.request_id", "q.request_id"]
) -> str:
    """Exact native ACK membership, not a publication or timing certificate."""
    if request_sql not in {"p.request_id", "q.request_id"}:
        raise ValueError("Historical ACK request identity is invalid")
    ack, _ = db.transition_window_history_format()
    return f"({ack}) AND ({db.transition_window_history_request_match(request_sql)})"


def foreign_admission_inputs_sql(
    db: StreamSQL,
    *,
    request_alias: Literal["r", "q"],
    project_sql: Literal["?", "scope.project_id"],
) -> str:
    """A retained physical request ID cannot own a foreign subject or project."""
    if request_alias not in {"r", "q"} or project_sql not in {"?", "scope.project_id"}:
        raise ValueError("Foreign input fact scope is invalid")
    project = db.project_filter("i").replace("?", project_sql)
    return (
        f"EXISTS(SELECT 1 FROM {db.table('interactions')} i WHERE i.request_id={request_alias}.request_id "  # noqa: S608 -- Fixed physical request relationship and validated aliases.
        f"AND (i.user_id IS NULL OR i.user_id<>{request_alias}.user_id OR NOT ({project})))"
    )
