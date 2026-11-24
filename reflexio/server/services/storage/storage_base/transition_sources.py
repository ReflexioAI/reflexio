"""Intrinsic source predicates shared by native transition observations."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

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


def finalized_run_candidate_sql(db: StreamSQL) -> str:
    """Structural class only; native execution/debt facts must also be clear."""
    idle = db.transition_window_history_parent_idle().replace("a.", "r.")
    return f"(r.status='finalized' AND substr(r.id,1,7)<>'window:' AND ({idle}))"


RUN_EXECUTION_FACTS = ("live_owner", "dependency", "preparation", "debt", "window")


def finalized_run_inert_sql(*, alias: str = "facts") -> str:
    """Derive history from already observed native facts, without more queries."""
    if alias != "facts":
        raise ValueError("Finalized history fact alias is invalid")
    return (
        "("
        + f"{alias}.terminal_history IS TRUE AND "
        + " AND ".join(f"{alias}.{fact} IS FALSE" for fact in RUN_EXECUTION_FACTS)
        + ")"
    )


def finalized_run_inert(facts: dict) -> bool:
    terminal = facts.get("terminal_history")
    return (terminal is True or (type(terminal) is int and terminal == 1)) and all(
        facts[fact] is False or (type(facts[fact]) is int and facts[fact] == 0)
        for fact in RUN_EXECUTION_FACTS
    )
