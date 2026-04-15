"""Unified token accounting for both host adapters.

OpenSpace routes all LLM calls through litellm, so we hook a litellm `CustomLogger`
callback (reusing OpenSpace's `gdpval_bench.token_tracker.TokenTracker`).

Hermes uses the raw OpenAI SDK pointed at OpenRouter, so we read token totals
directly from Hermes's `run_conversation()` result dict. See `stats_from_hermes_result`.

Both paths produce the same `TokenStats` dataclass so cross-host comparison is
a straight dict diff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from benchmark.gdpval.config import ensure_openspace_importable

ensure_openspace_importable()

# Re-export OpenSpace's TokenTracker for the OpenSpace adapter.
from gdpval_bench.token_tracker import TokenTracker  # noqa: E402
from gdpval_bench.token_tracker import TokenStats as OpenSpaceTokenStats  # noqa: E402


@dataclass
class TokenStats:
    """Per-task token accounting shared by all adapters.

    Matches the schema gdpval_bench writes to its results.jsonl `tokens` field
    so downstream tools (calc_subset_performance.py) work on our output unchanged.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stats_from_openspace(os_stats: OpenSpaceTokenStats) -> TokenStats:
    """Convert OpenSpace's TokenStats (from TokenTracker.stop()) to our shared shape.

    Args:
        os_stats (OpenSpaceTokenStats): Stats returned by `TokenTracker.stop()`.

    Returns:
        TokenStats: Normalized stats with the same fields OpenSpace reports.
    """
    return TokenStats(
        prompt_tokens=os_stats.prompt_tokens,
        completion_tokens=os_stats.completion_tokens,
        total_tokens=os_stats.total_tokens,
        cost_usd=os_stats.cost_usd,
        llm_calls=os_stats.llm_calls,
        wall_time_sec=os_stats.wall_time_sec,
    )


def stats_from_hermes_result(result: Mapping[str, Any], wall_time_sec: float) -> TokenStats:
    """Build TokenStats from Hermes's `run_conversation()` result dict.

    Hermes returns usage totals directly in the result dict (see `run_agent.py:10202`),
    so no callback is needed.

    Note: Hermes's own `total_tokens` field appears to accumulate the full
    conversation history across every API call (so a 20-iteration task with
    growing context can produce a "total" that's 8-10× the sum of unique
    prompt + completion bytes). To stay apples-to-apples with OpenSpace's
    litellm-tracked numbers, we derive `total_tokens` ourselves as
    `input_tokens + output_tokens` — the unique wire-level usage.

    Args:
        result (Mapping[str, Any]): Result dict from `AIAgent.run_conversation()`.
        wall_time_sec (float): Wall-clock duration measured by the adapter.

    Returns:
        TokenStats: Normalized stats in the same shape as the OpenSpace path.
    """
    prompt = int(result.get("input_tokens") or result.get("prompt_tokens") or 0)
    completion = int(result.get("output_tokens") or result.get("completion_tokens") or 0)
    return TokenStats(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cost_usd=float(result.get("estimated_cost_usd") or 0.0),
        llm_calls=int(result.get("api_calls") or 0),
        wall_time_sec=round(wall_time_sec, 2),
    )


__all__ = ["TokenStats", "TokenTracker", "stats_from_openspace", "stats_from_hermes_result"]
