"""One verdict for a failed extraction window: what the worker does about it.

:func:`classify_window_failure` reads the exception a window raised and returns
a closed, frozen :data:`WindowDisposition`. The worker executes it; it never
inspects the exception again to decide.

Precedence, highest first:

- :class:`YieldTurn` -- an execution input is locked elsewhere
  (``ExecutionInputBusyError`` on an explicit ``__cause__`` link,
  :func:`execution_input_is_busy`). Nothing failed; the turn yields with its
  computation and budgets intact.
- :class:`DeferSetup` -- the outermost exception is
  ``ReadinessUnavailableError``: required infrastructure is not qualified yet.
- :class:`Suspend` -- the outermost exception is ``OverflowSuspendedError``:
  saved overflow work needs another turn.
- :class:`Recompute` -- a ``ComputationEvidenceChangedError`` anywhere in the
  chain: the cached computation is stale and only recomputing it can pass.
- :class:`HoldInput` -- the first input-budget link in the chain decides the
  reason. ``InputBudgetConfigurationError`` is
  ``input_budget_configuration_error``, retried with backoff up to the window
  ceiling, and parked at once when a ``SavedOutputIncompatibleError`` is
  anywhere in the chain. A capacity overrun (``InputBudgetExceededError``,
  ``ExtractionInputBudgetExceededError``,
  ``OverflowAttemptBudgetExceededError``) is ``input_budget_exceeded`` and is
  held without replay.
- :class:`ProviderWait` -- :func:`classify_llm_failure` says
  :class:`OutageWait`: the provider is down for everyone, so the window waits
  without spending an attempt.
- :class:`Retry` -- anything else is the window's own failure: a charged retry,
  which storage parks at the attempt ceiling.

The chain is read through :func:`exception_chain`, the one walk owned by
``reflexio/server/llm/failure.py`` (``__cause__``, else ``__context__`` unless
``raise ... from None``; at most ``MAX_CHAIN_LINKS`` links).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reflexio.server.llm.failure import (
    OutageWait,
    classify_llm_failure,
    exception_chain,
)
from reflexio.server.llm.input_budget import (
    ComputationEvidenceChangedError,
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    SavedOutputIncompatibleError,
)
from reflexio.server.services.extraction.input_preparation import (
    ExtractionInputBudgetExceededError,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowSuspendedError,
)
from reflexio.server.services.storage.error import (
    ReadinessUnavailableError,
    execution_input_is_busy,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowAttemptBudgetExceededError,
)

InputHoldReason = Literal["input_budget_configuration_error", "input_budget_exceeded"]

_CAPACITY_EXCEEDED = (
    InputBudgetExceededError,
    ExtractionInputBudgetExceededError,
    OverflowAttemptBudgetExceededError,
)


@dataclass(frozen=True)
class YieldTurn:
    """An execution input is locked elsewhere; yield, keeping every budget."""


@dataclass(frozen=True)
class DeferSetup:
    """Required infrastructure is not ready; defer setup, keeping the window."""


@dataclass(frozen=True)
class Suspend:
    """Saved overflow work needs another worker turn."""


@dataclass(frozen=True)
class Recompute:
    """The cached computation's evidence changed; discard it and recompute."""


@dataclass(frozen=True)
class HoldInput:
    """The input budget refused the window, for ``reason``.

    ``park`` asks storage to hold the window for an operator now instead of
    retrying it; storage reports whether it ended up parked.
    """

    reason: InputHoldReason
    park: bool


@dataclass(frozen=True)
class ProviderWait:
    """The provider is down for everyone: wait ``delay_seconds``, uncharged."""

    delay_seconds: float


@dataclass(frozen=True)
class Retry:
    """The failure is the window's own: a charged retry."""


WindowDisposition = (
    YieldTurn | DeferSetup | Suspend | Recompute | HoldInput | ProviderWait | Retry
)


def classify_window_failure(exc: BaseException) -> WindowDisposition:
    """Classify a failed extraction window (precedence: module docstring).

    Args:
        exc (BaseException): The failure, as the worker caught it.

    Returns:
        WindowDisposition: The action the worker takes.
    """
    if execution_input_is_busy(exc):
        return YieldTurn()
    if isinstance(exc, ReadinessUnavailableError):
        return DeferSetup()
    if isinstance(exc, OverflowSuspendedError):
        return Suspend()
    chain = list(exception_chain(exc))
    if any(isinstance(link, ComputationEvidenceChangedError) for link in chain):
        return Recompute()
    for link in chain:
        if isinstance(link, InputBudgetConfigurationError):
            return HoldInput(
                "input_budget_configuration_error",
                park=any(
                    isinstance(other, SavedOutputIncompatibleError) for other in chain
                ),
            )
        if isinstance(link, _CAPACITY_EXCEEDED):
            return HoldInput("input_budget_exceeded", park=False)
    verdict = classify_llm_failure(exc)
    if isinstance(verdict, OutageWait):
        return ProviderWait(verdict.delay_seconds)
    return Retry()
