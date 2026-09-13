"""Plain, dependency-free per-run token accounting (OSS-safe).

Two producers feed the same ``RunTokenTotals`` shape, and which one is
authoritative depends on the path:

* **Live runs** use the :data:`run_token_capture` ContextVar. A generation run
  installs a fresh capture before it starts and every in-process completion adds
  into it (``_litellm_text_generation._log_token_usage``), so the total covers
  the whole run — extraction, consolidation/dedup, the playbook aggregator and
  reviewer, the should-run precheck — not just the extractor's own tool loop.
* **Durable resume** uses :func:`sum_trace_tokens`, folding the per-turn counts
  already on a ToolLoopTrace. A resumed window is decoded in a different
  process, where the ContextVar is empty by construction, so the trace sum is
  the only value available there.

**They are never summed, and the capture only wins when it observed something.**
Summing would double-count every extraction token, because in a live run the
capture is a superset of the trace fold — it sees the extractor's completions
*and* the ones after it. But a capture that observed ZERO completions has no
opinion rather than an answer of zero, so the trace fold still stands. That is
what keeps a caller who supplies ``token_totals`` by some other route from
silently billing 0, and it is why :class:`RunTokenCapture` counts observations
instead of inferring emptiness from all-zero totals.

The enterprise billing layer converts this into a TokenUsage; OSS never imports
reflexio_ext.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RunTokenTotals:
    """Accumulated token counts for a single generation run.

    The cache fields are **inclusive sub-buckets of ``prompt_tokens``**, not
    additional tokens: LiteLLM folds cache-creation and cache-read counts into
    ``prompt_tokens`` before we ever see them
    (``litellm/llms/anthropic/chat/transformation.py``), matching OpenTelemetry's
    rule that ``gen_ai.usage.cache_read.input_tokens`` SHOULD be included in
    ``gen_ai.usage.input_tokens``. They are carried separately only because the
    provider prices them differently — Anthropic charges cache-read at 0.1x and
    cache-write at 1.25x base input, a 12.5x spread hidden inside one number.

    **Never add a cache field to ``prompt_tokens``**, and never sum the four
    fields into a "total": both double-count.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_write_input_tokens: int = 0

    def add(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cache_read_input_tokens: int | None = None,
        cache_write_input_tokens: int | None = None,
    ) -> None:
        """Accumulate token counts, treating None as 0.

        Args:
            prompt_tokens: Prompt token count for one completion, or None.
            completion_tokens: Completion token count for one completion, or None.
            cache_read_input_tokens: Cached-prompt tokens read for this
                completion, or None. A sub-bucket of ``prompt_tokens``.
            cache_write_input_tokens: Prompt tokens written to the cache for this
                completion, or None. A sub-bucket of ``prompt_tokens``.
        """
        self.prompt_tokens += int(prompt_tokens or 0)
        self.completion_tokens += int(completion_tokens or 0)
        self.cache_read_input_tokens += int(cache_read_input_tokens or 0)
        self.cache_write_input_tokens += int(cache_write_input_tokens or 0)


@dataclass(slots=True)
class RunTokenCapture:
    """A run's accumulator plus how many completions it actually observed.

    ``completions`` is not decoration. A run that observed zero completions is
    NOT the same as one that observed completions reporting zero tokens — the
    ``claude_code`` and ``openclaw`` providers really do return ``Usage(0, 0)``.
    Only the first case means "this capture has no opinion", which is what lets
    a caller fall back to another producer without ever summing the two.
    """

    totals: RunTokenTotals = field(default_factory=RunTokenTotals)
    completions: int = 0

    def observe(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cache_read_input_tokens: int | None = None,
        cache_write_input_tokens: int | None = None,
    ) -> None:
        """Record one completion's usage, counting the observation itself."""
        self.completions += 1
        self.totals.add(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_write_input_tokens=cache_write_input_tokens,
        )


#: The run-scoped capture, or None when no run has installed one.
#:
#: ``default=None`` is load-bearing: a mutable default would be ONE object
#: shared by every context that never installs its own — including workers
#: submitted to a bare ``ThreadPoolExecutor`` that does not copy the context
#: (``agent_success_evaluation/regen_jobs.py``) — and that object would
#: accumulate across organizations. A None default makes those sites contribute
#: nothing, which is the correct answer for work outside a generation run.
run_token_capture: ContextVar[RunTokenCapture | None] = ContextVar(
    "reflexio_run_token_capture", default=None
)


def begin_run_token_capture() -> RunTokenCapture:
    """Install a FRESH capture for the run that is about to start.

    Returns the object so the caller can read it back at the end of the run.

    Replacing the object rather than zeroing the previous one is deliberate.
    A worker that outlived its parent — ``_execute_extractor`` does not cancel
    the thread on ``FuturesTimeoutError`` — keeps a reference to the OLD object
    and can keep adding to it. Because the next run reads a different object,
    those late writes cannot contaminate it.

    Must be called BEFORE any ``contextvars.copy_context()`` that the run
    performs: a copied context shares the capture *object*, but not a later
    ``set()``, so a capture installed inside the worker would not propagate
    back out.

    Returns:
        RunTokenCapture: The newly installed, zeroed capture.
    """
    capture = RunTokenCapture()
    run_token_capture.set(capture)
    return capture


def sum_trace_tokens(trace: Any) -> RunTokenTotals:
    """Fold a ToolLoopTrace's per-turn token counts into one RunTokenTotals.

    Used by the durable-resume path only — see the module docstring. The trace
    carries no cache breakdown, so those fields stay 0.

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
