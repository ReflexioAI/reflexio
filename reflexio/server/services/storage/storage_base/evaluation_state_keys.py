"""Single source of truth for evaluation ``_operation_state`` key formats.

Three evaluation namespaces store per-session state in ``_operation_state``:

1. ``agent_success_group_eval`` — the runner's "evaluated" marker.
2. ``grade_on_demand`` — the 24h on-demand grading cache.
3. ``retrieved_learning_eval`` — retrieved-learning generation/completion
   state (see ``retrieved_learning_state.build_retrieved_learning_state_key``).

Governance erasure must scrub all three for an erased user's sessions, so the
key builders live here where both the producers (runner / evaluation route)
and the erasers (sqlite + supabase governance) can import them without
layering cycles.
"""

from __future__ import annotations

AGENT_SUCCESS_MARKER_PREFIX = "agent_success_group_eval"
GRADE_ON_DEMAND_CACHE_PREFIX = "grade_on_demand"


def build_agent_success_marker_key(org_id: str, user_id: str, session_id: str) -> str:
    """Build the agent-success "evaluated" marker key for a session.

    Historical plain ``::`` join — kept byte-identical so existing markers
    stay addressable.

    Args:
        org_id (str): Organization ID.
        user_id (str): Session owner.
        session_id (str): Evaluated session.

    Returns:
        str: The marker key.
    """
    return f"{AGENT_SUCCESS_MARKER_PREFIX}::{org_id}::{user_id}::{session_id}"


def build_grade_on_demand_cache_key(
    org_id: str, session_id: str, agent_version: str, evaluation_name: str
) -> str:
    """Build the grade-on-demand cache key (length-prefixed components).

    Length-prefixing each free-form component keeps the join injective even
    when a component contains the ``::`` delimiter.

    Args:
        org_id (str): Organization ID.
        session_id (str): Target session.
        agent_version (str): Agent version filter.
        evaluation_name (str): Evaluator/result namespace.

    Returns:
        str: The cache key.
    """
    parts = "::".join(
        f"{len(s)}:{s}" for s in (org_id, session_id, agent_version, evaluation_name)
    )
    return f"{GRADE_ON_DEMAND_CACHE_PREFIX}::{parts}"


def build_grade_on_demand_session_prefix(org_id: str, session_id: str) -> str:
    """Prefix matching every grade-on-demand cache key for one session.

    Used by governance erasure to find a session's cache rows regardless of
    agent_version / evaluation_name.

    Args:
        org_id (str): Organization ID.
        session_id (str): Target session.

    Returns:
        str: The per-session key prefix (ends with ``::``).
    """
    return (
        f"{GRADE_ON_DEMAND_CACHE_PREFIX}"
        f"::{len(org_id)}:{org_id}::{len(session_id)}:{session_id}::"
    )
