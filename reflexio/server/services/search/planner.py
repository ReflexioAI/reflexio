"""PLAN stage of deep unified search: query → targeted per-arm subqueries."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.search.deep_search_schemas import (
    PlannedSubquery,
    SearchPlan,
)
from reflexio.server.tracing import profile_step

if TYPE_CHECKING:
    from reflexio.models.api_schema.retriever_schema import ConversationTurn
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager

logger = logging.getLogger(__name__)

_PROMPT_ID = "deep_search_plan"


# Plan/reflect outputs are small structured objects; an explicit cap avoids
# the unbounded-max_tokens strict-json tail-latency failure mode some
# providers exhibit (reproduced on MiniMax M3: uncapped + strict json_schema
# deterministically hits the request timeout).
_STRUCTURED_MAX_TOKENS = 4000


def _structured_call(
    *,
    llm_client: LiteLLMClient,
    prompt: str,
    response_format: type,
    model_name: str | None,
    model_role: ModelRole,
    timeout_s: float,
) -> object:
    """One-shot structured LLM call with the model/role precedence handled.

    ``model_role`` overrides an explicit ``model`` inside
    ``generate_chat_response``, so exactly one of the two is passed.
    """
    messages = [{"role": "user", "content": prompt}]
    if model_name:
        return llm_client.generate_chat_response(
            messages=messages,
            response_format=response_format,
            model=model_name,
            timeout=timeout_s,
            max_retries=1,
            max_tokens=_STRUCTURED_MAX_TOKENS,
        )
    return llm_client.generate_chat_response(
        messages=messages,
        response_format=response_format,
        model_role=model_role,
        timeout=timeout_s,
        max_retries=1,
        max_tokens=_STRUCTURED_MAX_TOKENS,
    )


def _conversation_block(history: list[ConversationTurn] | None) -> str:
    if not history:
        return ""
    lines = [f"[{turn.role}]: {turn.content}" for turn in history]
    return "Conversation context:\n" + "\n".join(lines) + "\n"


def fallback_plan(query: str, allowed_arms: list[str]) -> SearchPlan:
    """Deterministic plan used when the planner LLM fails: verbatim fan-out.

    Args:
        query (str): The original search query.
        allowed_arms (list[str]): Arms the request permits.

    Returns:
        SearchPlan: One verbatim hybrid subquery per allowed arm.
    """
    return SearchPlan(
        subqueries=[
            PlannedSubquery(arm=arm, query=query, reason="verbatim fallback")  # type: ignore[arg-type]
            for arm in allowed_arms
        ],
        notes="planner fallback: verbatim fan-out",
    )


def plan_subqueries(
    *,
    query: str,
    conversation_history: list[ConversationTurn] | None,
    allowed_arms: list[str],
    max_subqueries: int,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    model_name: str | None = None,
    timeout_s: float = 15.0,
) -> SearchPlan:
    """Run the PLAN stage: one structured LLM call, hard-validated output.

    Subqueries for arms outside ``allowed_arms`` are dropped and the total
    is clamped to ``max_subqueries`` regardless of what the model returned;
    any failure (call error, parse error, empty plan) degrades to the
    deterministic verbatim fan-out so deep search never dies at planning.

    Args:
        query: The original search query.
        conversation_history: Prior turns for context, when provided.
        allowed_arms: Arms the request permits (already excludes profiles
            when the request has no user_id).
        max_subqueries: Hard cap on planned subqueries.
        llm_client: Shared LLM client.
        prompt_manager: Prompt manager rendering ``deep_search_plan``.
        model_name: Optional explicit model override; defaults to the
            PRE_RETRIEVAL role resolution.
        timeout_s: Per-call timeout.

    Returns:
        SearchPlan: Validated plan (never empty).
    """
    prompt = prompt_manager.render_prompt(
        _PROMPT_ID,
        {
            "query": query,
            "conversation_context_block": _conversation_block(conversation_history),
            "today_iso": datetime.now(UTC).date().isoformat(),
            "allowed_arms": ", ".join(allowed_arms),
            "max_subqueries": max_subqueries,
        },
    )
    with profile_step("search.deep.plan") as span:
        try:
            response = _structured_call(
                llm_client=llm_client,
                prompt=prompt,
                response_format=SearchPlan,
                model_name=model_name,
                model_role=ModelRole.PRE_RETRIEVAL,
                timeout_s=timeout_s,
            )
        except Exception:
            logger.warning("deep search planner call failed", exc_info=True)
            span.set_data("fallback", True)
            return fallback_plan(query, allowed_arms)

        if not isinstance(response, SearchPlan):
            logger.warning(
                "deep search planner returned %s; using fallback plan",
                type(response).__name__,
            )
            span.set_data("fallback", True)
            return fallback_plan(query, allowed_arms)

        subqueries = [
            sq for sq in response.subqueries if sq.arm in allowed_arms and sq.query
        ][:max_subqueries]
        span.set_data("subquery_count", len(subqueries))
        if not subqueries:
            span.set_data("fallback", True)
            return fallback_plan(query, allowed_arms)
        return SearchPlan(subqueries=subqueries, notes=response.notes)
