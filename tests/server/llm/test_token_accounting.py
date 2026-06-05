"""Unit tests for the OSS per-run token accounting helpers.

Covers ``RunTokenTotals.add`` (including the ``int(x or 0)`` None->0 coercion)
and ``sum_trace_tokens`` (the missing/empty ``turns`` guard and summation across
turns).
"""

from types import SimpleNamespace

from reflexio.server.llm.token_accounting import RunTokenTotals, sum_trace_tokens


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
