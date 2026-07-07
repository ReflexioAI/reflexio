"""Listwise LLM reranking for deep unified search.

Runs after a corrective round has produced the final fused pool: a
sliding-window listwise rerank (RankGPT-style, tail-to-head so strong late
candidates bubble forward) with a two-stage fallback chain:

    listwise window rerank → pointwise ``score_pairs_llm`` → identity order

Candidates from recency-dominant arms are excluded — their final ordering is
timestamp-based in the service and an LLM rerank must not scramble it.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.rerank import score_pairs_llm
from reflexio.server.services.search.deep_search_schemas import RerankOutput
from reflexio.server.services.search.planner import _structured_call
from reflexio.server.tracing import profile_step

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.search.executor import Candidate

logger = logging.getLogger(__name__)

_PROMPT_ID = "deep_search_rerank"
_CONTENT_CHAR_CAP = 400
_WINDOW_SIZE = 20
_WINDOW_STRIDE = 10


def _candidates_block(candidates: list[Candidate], now: int) -> str:
    lines = []
    for candidate in candidates:
        age = candidate.age_days(now)
        age_repr = f"{age:.1f}" if age is not None else "?"
        content = str(getattr(candidate.entity, "content", ""))[:_CONTENT_CHAR_CAP]
        lines.append(f"{candidate.key} | {candidate.arm} | {age_repr} | {content}")
    return "\n".join(lines)


def _rerank_window(
    window: list[Candidate],
    *,
    query: str,
    now: int,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    model_name: str | None,
    timeout_s: float,
) -> list[Candidate] | None:
    """Rerank one window with a single listwise call; None on failure."""
    prompt = prompt_manager.render_prompt(
        _PROMPT_ID,
        {
            "query": query,
            "today_iso": datetime.now(UTC).date().isoformat(),
            "candidates_block": _candidates_block(window, now),
        },
    )
    try:
        response = _structured_call(
            llm_client=llm_client,
            prompt=prompt,
            response_format=RerankOutput,
            model_name=model_name,
            model_role=ModelRole.GENERATION,
            timeout_s=timeout_s,
        )
    except Exception:
        logger.warning("deep search rerank window call failed", exc_info=True)
        return None
    if not isinstance(response, RerankOutput):
        return None

    by_key = {candidate.key: candidate for candidate in window}
    ranked = [by_key[key] for key in response.ranked_candidate_ids if key in by_key]
    ranked_keys = {candidate.key for candidate in ranked}
    ranked.extend(c for c in window if c.key not in ranked_keys)
    return ranked


def _pointwise_fallback(
    candidates: list[Candidate],
    *,
    query: str,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
) -> list[Candidate] | None:
    """Score the whole pool pointwise via the shared LLM relevance judge."""
    docs = [str(getattr(candidate.entity, "content", "")) for candidate in candidates]
    scores = score_pairs_llm(
        query=query,
        docs=docs,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
    )
    if scores is None:
        return None
    paired = sorted(
        zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True
    )
    return [candidate for candidate, _score in paired]


def listwise_rerank(
    *,
    query: str,
    candidates: list[Candidate],
    skip_arms: set[str] | None = None,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    model_name: str | None = None,
    timeout_s: float = 45.0,
    now: int | None = None,
) -> list[Candidate]:
    """Order the fused candidate pool against the original query.

    Sliding windows of ``_WINDOW_SIZE`` with stride ``_WINDOW_STRIDE`` run
    tail-to-head so strong late candidates bubble to the front (one call for
    pools that fit a single window). Any window failure aborts listwise mode
    and falls back to one pointwise ``score_pairs_llm`` call; if that also
    fails the input order is returned.

    Args:
        query: The ORIGINAL search query.
        candidates: Fused, deduped pool in current order.
        skip_arms: Arms excluded from reranking (recency-dominant); their
            candidates keep their positions relative to the reranked rest
            appended at the front of their arm handled by the service.
        llm_client: Shared LLM client.
        prompt_manager: Prompt manager rendering ``deep_search_rerank``.
        model_name: Optional explicit reranker model; defaults to the
            GENERATION role resolution.
        timeout_s: Per-window timeout.
        now: Epoch seconds for age computation (real now when omitted).

    Returns:
        list[Candidate]: Reranked pool (skipped-arm candidates appended in
            their original order — the service re-sorts those arms by
            timestamp anyway).
    """
    now = now or int(datetime.now(UTC).timestamp())
    skip_arms = skip_arms or set()
    skipped = [c for c in candidates if c.arm in skip_arms]
    pool = [c for c in candidates if c.arm not in skip_arms]
    if len(pool) < 2:
        return pool + skipped

    with profile_step("search.deep.rerank", pool=len(pool)) as span:
        ordered = _sliding_window_pass(
            pool,
            query=query,
            now=now,
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            model_name=model_name,
            timeout_s=timeout_s,
        )
        if ordered is None:
            span.set_data("fallback", "pointwise")
            ordered = _pointwise_fallback(
                pool,
                query=query,
                llm_client=llm_client,
                prompt_manager=prompt_manager,
            )
        if ordered is None:
            span.set_data("fallback", "identity")
            ordered = pool
    return ordered + skipped


def _sliding_window_pass(
    pool: list[Candidate],
    *,
    query: str,
    now: int,
    llm_client: LiteLLMClient,
    prompt_manager: PromptManager,
    model_name: str | None,
    timeout_s: float,
) -> list[Candidate] | None:
    """RankGPT-style tail-to-head sliding window; None if any window fails."""
    ordered = list(pool)
    start = max(0, len(ordered) - _WINDOW_SIZE)
    while True:
        window = ordered[start : start + _WINDOW_SIZE]
        reranked = _rerank_window(
            window,
            query=query,
            now=now,
            llm_client=llm_client,
            prompt_manager=prompt_manager,
            model_name=model_name,
            timeout_s=timeout_s,
        )
        if reranked is None:
            return None
        ordered[start : start + _WINDOW_SIZE] = reranked
        if start == 0:
            return ordered
        start = max(0, start - _WINDOW_STRIDE)
