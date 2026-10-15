"""Caller-imposed absolute deadline for LLM calls.

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

# Absolute deadline on the ``time.monotonic()`` clock, or None for no bound.
llm_call_deadline: ContextVar[float | None] = ContextVar(
    "llm_call_deadline", default=None
)


@contextmanager
def llm_deadline(deadline: float) -> Iterator[None]:
    """Bound every LLM ladder walk inside the block by ``deadline``.

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
