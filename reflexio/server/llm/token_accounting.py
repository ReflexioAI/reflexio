"""Plain, dependency-free per-run token accounting (OSS-safe).

Folds the per-turn token counts already present on a ToolLoopTrace into a single
run total. The enterprise billing layer converts this into a TokenUsage; OSS never
imports reflexio_ext.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RunTokenTotals:
    """Accumulated token counts for a single extraction agent run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, *, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        """Accumulate token counts, treating None as 0.

        Args:
            prompt_tokens: Prompt token count for one turn, or None if unavailable.
            completion_tokens: Completion token count for one turn, or None if unavailable.
        """
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)


def sum_trace_tokens(trace: Any) -> RunTokenTotals:
    """Fold a ToolLoopTrace's per-turn token counts into one RunTokenTotals.

    Args:
        trace: A ToolLoopTrace (or duck-typed equivalent) with a ``turns`` attribute.
            Each turn may have ``prompt_tokens`` and ``completion_tokens`` attributes.

    Returns:
        RunTokenTotals with the summed prompt and completion tokens across all turns.
    """
    totals = RunTokenTotals()
    for turn in getattr(trace, "turns", []) or []:
        totals.add(
            prompt_tokens=getattr(turn, "prompt_tokens", None),
            completion_tokens=getattr(turn, "completion_tokens", None),
        )
    return totals
