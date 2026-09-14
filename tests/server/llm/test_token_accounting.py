"""Unit tests for the OSS per-run token accounting helpers.

Covers ``RunTokenTotals.add`` (including the ``int(x or 0)`` None->0 coercion),
``sum_trace_tokens`` (the missing/empty ``turns`` guard and summation across
turns), and the run-scoped ``RunTokenCapture`` — whose three load-bearing
properties are the absent-by-default ContextVar, the fresh object per run, and
the observation COUNT that distinguishes "saw nothing" from "saw zeros".
"""

import contextvars
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from reflexio.server.llm.token_accounting import (
    RunTokenCapture,
    RunTokenTotals,
    begin_run_token_capture,
    run_token_capture,
    sum_trace_tokens,
)


def test_run_token_totals_default_zero() -> None:
    """A fresh RunTokenTotals starts at zero on both counters."""
    totals = RunTokenTotals()
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0


def test_add_accumulates_across_calls() -> None:
    """Repeated .add calls accumulate prompt and completion tokens independently."""
    totals = RunTokenTotals()
    totals.add(prompt_tokens=10, completion_tokens=3)
    totals.add(prompt_tokens=5, completion_tokens=7)
    assert totals.prompt_tokens == 15
    assert totals.completion_tokens == 10


def test_add_coerces_none_to_zero() -> None:
    """None token values are coerced to 0 via int(x or 0), not raising."""
    totals = RunTokenTotals(prompt_tokens=4, completion_tokens=2)
    totals.add(prompt_tokens=None, completion_tokens=None)
    assert totals.prompt_tokens == 4
    assert totals.completion_tokens == 2


def test_add_mixed_none_and_value() -> None:
    """A None on one axis coerces to 0 while the other axis still accumulates."""
    totals = RunTokenTotals()
    totals.add(prompt_tokens=None, completion_tokens=9)
    totals.add(prompt_tokens=6, completion_tokens=None)
    assert totals.prompt_tokens == 6
    assert totals.completion_tokens == 9


def test_sum_trace_tokens_empty_turns() -> None:
    """A trace whose turns list is empty folds to zeros."""
    trace = SimpleNamespace(turns=[])
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0


def test_sum_trace_tokens_missing_turns_attribute() -> None:
    """A trace with no ``turns`` attribute at all folds to zeros (getattr default)."""
    trace = SimpleNamespace()
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0


def test_sum_trace_tokens_none_turns() -> None:
    """A trace whose ``turns`` is None folds to zeros via the ``or []`` guard."""
    trace = SimpleNamespace(turns=None)
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0


def test_sum_trace_tokens_sums_multiple_turns() -> None:
    """Token counts are summed across multiple turns."""
    trace = SimpleNamespace(
        turns=[
            SimpleNamespace(prompt_tokens=10, completion_tokens=2),
            SimpleNamespace(prompt_tokens=20, completion_tokens=5),
            SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        ]
    )
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 31
    assert totals.completion_tokens == 8


def test_sum_trace_tokens_turn_missing_token_attrs() -> None:
    """Turns missing token attributes contribute 0 (getattr default -> None -> 0)."""
    trace = SimpleNamespace(
        turns=[
            SimpleNamespace(prompt_tokens=4, completion_tokens=3),
            SimpleNamespace(),  # no prompt_tokens / completion_tokens
        ]
    )
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 4
    assert totals.completion_tokens == 3


def test_sum_trace_tokens_turn_none_token_values() -> None:
    """Turns with explicit None token values contribute 0 via .add coercion."""
    trace = SimpleNamespace(
        turns=[
            SimpleNamespace(prompt_tokens=None, completion_tokens=None),
            SimpleNamespace(prompt_tokens=8, completion_tokens=6),
        ]
    )
    totals = sum_trace_tokens(trace)
    assert totals.prompt_tokens == 8
    assert totals.completion_tokens == 6


# ── The run-scoped capture (Change 1) ───────────────────────────────────────


def test_cache_buckets_accumulate_separately_from_prompt_tokens() -> None:
    """The cache fields are carried alongside, never folded into, prompt_tokens.

    LiteLLM has already added cache-creation and cache-read into prompt_tokens
    before we see them, so adding them again here would double-count the most
    expensive half of the bill.
    """
    totals = RunTokenTotals()
    totals.add(
        prompt_tokens=1000,
        completion_tokens=50,
        cache_read_input_tokens=800,
        cache_write_input_tokens=100,
    )
    assert totals.prompt_tokens == 1000
    assert totals.cache_read_input_tokens == 800
    assert totals.cache_write_input_tokens == 100


def test_capture_counts_observations_not_just_tokens() -> None:
    """A completion reporting 0/0 still counts as an observation.

    ``claude_code`` and ``openclaw`` genuinely return ``Usage(0, 0)``. Inferring
    "nothing was observed" from all-zero totals would make those runs fall back
    to another producer, which is the bug this counter exists to prevent.
    """
    capture = RunTokenCapture()
    assert capture.completions == 0
    capture.observe(prompt_tokens=0, completion_tokens=0)
    assert capture.completions == 1
    assert capture.totals.prompt_tokens == 0


def test_begin_run_token_capture_installs_a_fresh_object_each_time() -> None:
    """Each run gets a NEW object, so a leaked writer cannot reach the next run.

    ``_execute_extractor`` does not cancel its worker on a timeout, so an
    orphaned thread can keep calling ``observe`` after the parent has moved on.
    It holds the old object; replacing rather than zeroing is what makes those
    late writes harmless.
    """
    first = begin_run_token_capture()
    first.observe(prompt_tokens=10, completion_tokens=1)

    second = begin_run_token_capture()
    assert second is not first
    assert second.completions == 0
    assert second.totals.prompt_tokens == 0

    # The "orphan" writes into the object it still holds; the live run is unmoved.
    first.observe(prompt_tokens=999, completion_tokens=999)
    assert run_token_capture.get() is second
    assert second.totals.prompt_tokens == 0


def test_capture_is_absent_by_default() -> None:
    """Outside a generation run there is no capture at all — not a shared one.

    A mutable ContextVar default would be one process-global object collecting
    tokens across organisations, because a worker submitted to a bare
    ThreadPoolExecutor never copies the context.
    """

    def read_in_uncopied_worker() -> RunTokenCapture | None:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(run_token_capture.get).result()

    begin_run_token_capture()
    # The worker inherits nothing: a new thread starts from an empty context.
    assert read_in_uncopied_worker() is None


def test_copied_context_shares_the_capture_object() -> None:
    """The install must precede ``copy_context()`` — this pins why.

    A copied context shares the object, so a worker's ``observe`` is visible to
    the parent. It does NOT share a later ``set()``, which is why installing the
    capture inside the worker would silently produce zero.
    """
    capture = begin_run_token_capture()
    ctx = contextvars.copy_context()

    def work() -> None:
        inner = run_token_capture.get()
        assert inner is capture
        inner.observe(prompt_tokens=7, completion_tokens=2)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(ctx.run, work).result()

    assert capture.totals.prompt_tokens == 7
    assert capture.completions == 1
