"""Caller-imposed absolute deadline for LLM calls.

Provider admission and embedding boundaries share this context through
``remaining_call_time``. Boundary checks reject expired dispatch and late
results; they do not forcibly interrupt synchronous transport or SDK work.

An enclosing operation with its own wall-clock bound (the generation service's
per-extractor timeout) installs a deadline here; every ladder walk started
inside it plans its rungs and repair turns against
``min(own budget, time left before the deadline)`` and stops itself before the
caller gives up, instead of being abandoned mid-call by an outer timeout while
it keeps spending provider calls.

A ``ContextVar`` rather than a parameter because the walk sits several layers
below the caller (extractor -> agent -> tool loop -> client), and the extractor
already runs in a copied context (``_execute_extractor``), so the value reaches
exactly the calls that run on the caller's behalf.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

# Absolute deadline on the ``time.monotonic()`` clock, or None for no bound.
llm_call_deadline: ContextVar[float | None] = ContextVar(
    "llm_call_deadline", default=None
)


class CallDeadlineExceededError(TimeoutError):
    """The enclosing compute attempt has no time left for more work."""


def remaining_call_time(default: float = 30.0) -> float:
    """Tighten a wait by the existing compute deadline, refusing expired work.

    This is a cooperative boundary check, not an interrupt of synchronous I/O
    or SDK processing. Callers must also check before accepting a result.
    """
    deadline = llm_call_deadline.get()
    if deadline is None:
        return default
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise CallDeadlineExceededError("LLM operation deadline exceeded")
    return min(default, remaining)


@contextmanager
def llm_deadline(deadline: float) -> Iterator[None]:
    """Share ``deadline`` with LLM walks, admission and embedding boundaries.

    Nested scopes can only tighten the bound: the effective deadline is the
    earlier of this one and any already installed.

    Args:
        deadline (float): Absolute ``time.monotonic()`` deadline.

    Yields:
        None
    """
    current = llm_call_deadline.get()
    token = llm_call_deadline.set(
        deadline if current is None else min(current, deadline)
    )
    try:
        yield
    finally:
        llm_call_deadline.reset(token)
