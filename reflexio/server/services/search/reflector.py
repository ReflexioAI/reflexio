"""REFLECT stage of deep unified search: sufficiency grade + ordering."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.search.deep_search_schemas import ReflectVerdict
from reflexio.server.services.search.planner import _structured_call
from reflexio.server.tracing import profile_step

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.search.executor import Candidate

logger = logging.getLogger(__name__)

_PROMPT_ID = "deep_search_reflect"
_CONTENT_CHAR_CAP = 400


def _candidates_block(candidates: list[Candidate], now: int) -> str:
    lines = []
    for candidate in candidates:
        age = candidate.age_days(now)
        age_repr = f"{age:.1f}" if age is not None else "?"
        content = str(getattr(candidate.entity, "content", ""))[:_CONTENT_CHAR_CAP]
        lines.append(f"{candidate.key} | {candidate.arm} | {age_repr} | {content}")
    return "\n".join(lines) or "(no candidates retrieved)"


def _identity_verdict(candidates: list[Candidate]) -> ReflectVerdict:
    return ReflectVerdict(
        sufficiency="sufficient",
        ranked_candidate_ids=[candidate.key for candidate in candidates],
        rationale="reflect fallback: retrieval order kept",
    )


def reflect_and_rank(
    *,
    query: str,
    plan_notes: str,
    candidates: list[Candidate],
    allow_corrective: bool,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    model_name: str | None = None,
    timeout_s: float = 45.0,
    now: int | None = None,
) -> ReflectVerdict:
    """Run the REFLECT stage: one structured LLM call, validated output.

    Hallucinated candidate keys are dropped; on any failure the verdict
    degrades to "sufficient" with retrieval-order ranking so deep search
    still returns the fused pool.

    Args:
        query: The ORIGINAL search query (not a rewritten subquery).
        plan_notes: The planner's notes (threaded for context).
        candidates: Fused, deduped candidates from the executor.
        allow_corrective: Whether the verdict may request another round.
        llm_client: Shared LLM client.
        prompt_manager: Prompt manager rendering ``deep_search_reflect``.
        model_name: Optional explicit model override; defaults to the
            GENERATION role resolution (strong tier).
        timeout_s: Per-call timeout.
        now: Epoch seconds for age computation (real now when omitted).

    Returns:
        ReflectVerdict: Validated verdict (corrective subqueries emptied
            when correction is not allowed).
    """
    now = now or int(datetime.now(UTC).timestamp())
    prompt = prompt_manager.render_prompt(
        _PROMPT_ID,
        {
            "query": query,
            "today_iso": datetime.now(UTC).date().isoformat(),
            "plan_notes": plan_notes or "(none)",
            "candidates_block": _candidates_block(candidates, now),
            "allow_corrective": "true" if allow_corrective else "false",
        },
    )
    with profile_step("search.deep.reflect", candidates=len(candidates)) as span:
        try:
            response = _structured_call(
                llm_client=llm_client,
                prompt=prompt,
                response_format=ReflectVerdict,
                model_name=model_name,
                model_role=ModelRole.GENERATION,
                timeout_s=timeout_s,
            )
        except Exception:
            logger.warning("deep search reflect call failed", exc_info=True)
            span.set_data("fallback", True)
            return _identity_verdict(candidates)

        if not isinstance(response, ReflectVerdict):
            logger.warning(
                "deep search reflect returned %s; keeping retrieval order",
                type(response).__name__,
            )
            span.set_data("fallback", True)
            return _identity_verdict(candidates)

        known_keys = {candidate.key for candidate in candidates}
        ranked = [key for key in response.ranked_candidate_ids if key in known_keys]
        corrective = response.corrective_subqueries if allow_corrective else []
        span.set_data("sufficiency", response.sufficiency)
        span.set_data("corrective_count", len(corrective))
        return ReflectVerdict(
            sufficiency=response.sufficiency,
            ranked_candidate_ids=ranked,
            corrective_subqueries=corrective,
            rationale=response.rationale,
        )
