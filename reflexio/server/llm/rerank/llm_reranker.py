"""LLM-based reranker for ``(query, document)`` pairs.

Sends a single LLM call with the query and a numbered list of candidate
facts, expects a JSON array of relevance scores (0-10) in the same order.
Fills the gap that cross-encoders can't bridge: world knowledge / brand
↔ category equivalence (e.g. "Thrive Market" = grocery service).

Usage
-----

>>> from reflexio.server.llm.rerank.llm_reranker import score_pairs_llm
>>> scores = score_pairs_llm(
...     query="grocery store",
...     docs=["Walmart $120 grocery", "Thrive Market organic"],
...     llm_client=client, prompt_manager=pm,
... )
>>> # scores is a list[float] of length 2, or None on failure.

The helper returns ``None`` on any failure (no LLM client, prompt error,
LLM call error, malformed JSON, score-count mismatch). Callers should
fall back to the existing rerank order.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_LOGGER = logging.getLogger(__name__)

_PROMPT_ID = "rerank_relevance"
_LLM_TIMEOUT = 30.0
_LLM_MAX_RETRIES = 1


def _parse_scores(text: str, expected_n: int) -> list[float] | None:
    """Extract a JSON array of N numbers from raw LLM output.

    Tolerates surrounding prose by locating the outermost ``[ ... ]`` span
    via the first ``[`` and last ``]``. Returns ``None`` when the array
    can't be parsed or the count doesn't match ``expected_n``.

    Args:
        text (str): Raw LLM response.
        expected_n (int): Number of scores expected.

    Returns:
        list[float] | None: Parsed scores, or ``None`` on any parse failure.
    """
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or len(parsed) != expected_n:
        return None
    try:
        return [float(s) for s in parsed]
    except (TypeError, ValueError):
        return None


def _format_docs_block(docs: list[str]) -> str:
    """Render docs as a numbered, single-line-per-fact block.

    Newlines inside individual docs are replaced with spaces so the LLM
    sees a clean ``i. <fact>`` structure that matches the prompt's
    output rule (one score per numbered fact).

    Args:
        docs (list[str]): Candidate documents.

    Returns:
        str: One ``i. <doc>`` per line.
    """
    flattened = (re.sub(r"\s+", " ", d).strip() for d in docs)
    return "\n".join(f"{i + 1}. {d}" for i, d in enumerate(flattened))


def score_pairs_llm(
    query: str,
    docs: list[str],
    llm_client: Any,
    prompt_manager: Any,
    timeout: float = _LLM_TIMEOUT,
) -> list[float] | None:
    """Score ``(query, doc)`` pairs via an LLM relevance-judge prompt.

    A single LLM call is made for the entire batch; the model returns one
    score per doc. The model is the LiteLLMClient's default — same model
    that drives the search agent — chosen to keep configuration simple.

    Args:
        query (str): The reranking query.
        docs (list[str]): Documents to score against ``query``.
        llm_client (Any): A ``LiteLLMClient`` (must expose ``generate_response``).
        prompt_manager (Any): A ``PromptManager`` (must expose ``render_prompt``).
        timeout (float): Per-call timeout in seconds.

    Returns:
        list[float] | None: One score per doc on success (same order as
            ``docs``); ``None`` on any failure so the caller can fall back.
    """
    if not docs:
        return []
    if llm_client is None or prompt_manager is None:
        return None
    try:
        prompt = prompt_manager.render_prompt(
            _PROMPT_ID,
            {
                "query": query,
                "docs_block": _format_docs_block(docs),
                "num_docs": str(len(docs)),
            },
        )
    except Exception as e:  # noqa: BLE001 — prompt-render failure must not break search
        _LOGGER.warning("LLM rerank: prompt render failed: %s", e)
        return None

    try:
        result = llm_client.generate_response(
            prompt,
            timeout=timeout,
            max_retries=_LLM_MAX_RETRIES,
            temperature=0.0,
        )
    except Exception as e:  # noqa: BLE001 — LLM-call failure must not break search
        _LOGGER.warning("LLM rerank: generate_response failed: %s", e)
        return None

    if not isinstance(result, str) or not result.strip():
        _LOGGER.warning(
            "LLM rerank: empty/non-string response (%r)", type(result).__name__
        )
        return None

    scores = _parse_scores(result, expected_n=len(docs))
    if scores is None:
        _LOGGER.warning(
            "LLM rerank: parse failure (n=%d, raw[:200]=%r)", len(docs), result[:200]
        )
    return scores
