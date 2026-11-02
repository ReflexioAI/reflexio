"""Compute admission and unretained progress; neither grants execution ownership."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock

from reflexio.server.llm._litellm_types import ProviderRequestGuardError


class ComputeAdmissionError(ProviderRequestGuardError):
    def __init__(self, reason: Exception):
        super().__init__("External compute admission refused")
        self.reason = reason


class ComputeInterruptedError(RuntimeError):
    """Unretained computation cannot be retried as a free writer-only yield."""


@dataclass
class _ComputeProgress:
    _entered: bool = False
    _completed: bool = False
    _reservation: Callable[[], None] | None = None
    _parent: _ComputeProgress | None = None
    _reserved: bool = False
    _reservation_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def reserve(self) -> None:
        if self._parent is not None:
            self._parent.reserve()
        if self._reservation is not None:
            with self._reservation_lock:
                if not self._reserved:
                    self._reservation()
                    self._reserved = True

    @property
    def unretained(self) -> bool:
        return self._entered or self._completed

    def record_result(self) -> None:
        self._completed = True

    def retain(self) -> None:
        self._entered = self._completed = False


external_compute_admission: ContextVar[Callable[[], None] | None] = ContextVar(
    "external_compute_admission", default=None
)
_compute_progress: ContextVar[_ComputeProgress | None] = ContextVar(
    "unretained_compute_progress", default=None
)


@contextmanager
def compute_attempt_scope(
    *,
    reserve: Callable[[], None] | None = None,
) -> Iterator[_ComputeProgress]:
    """Carry actual progress until its owner confirms durable retention.

    Nested work contributes only its unretained progress. This is a transient
    retry classification witness, not a lease, budget, receipt or write grant.
    An optional owner callback reserves its existing allowance before dispatch;
    nested scopes cannot bypass it or reserve it twice.
    """
    parent = _compute_progress.get()
    progress = _ComputeProgress(_reservation=reserve, _parent=parent)
    token = _compute_progress.set(progress)
    try:
        yield progress
    except Exception as error:
        from reflexio.server.services.storage.error import (
            ReadinessUnavailableError,
            execution_input_is_busy,
        )

        if (progress.unretained and execution_input_is_busy(error)) or (
            progress._completed and isinstance(error, ReadinessUnavailableError)
        ):
            # The input-only cause would incorrectly authorize a free replay;
            # preserve only the value-free interrupted-computation boundary.
            raise ComputeInterruptedError(
                "Computation interrupted before durable retention"
            ) from None
        raise
    finally:
        _compute_progress.reset(token)
        if parent is not None:
            if progress._entered:
                parent._entered = True
            if progress._completed:
                parent._completed = True


def record_compute_result() -> None:
    """Observe a returned provider response or a completed volatile result."""
    progress = _compute_progress.get()
    if progress is not None:
        progress.record_result()


def retain_compute_result() -> None:
    """Only a confirmed durable save can end this scope's replay obligation."""
    progress = _compute_progress.get()
    if progress is not None:
        progress.retain()


def reserve_compute_attempt() -> None:
    """Persist any enclosing owner's debit before computation can begin."""
    progress = _compute_progress.get()
    if progress is not None:
        progress.reserve()


def admit_external_compute() -> None:
    guard = external_compute_admission.get()
    try:
        if guard is not None:
            guard()
        reserve_compute_attempt()
    except Exception as error:
        raise ComputeAdmissionError(error) from error
    progress = _compute_progress.get()
    if progress is not None:
        progress._entered = True
