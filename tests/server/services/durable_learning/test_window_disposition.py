"""Every failure shape the window handler acts on, and exactly what it does.

``classify_window_failure`` decides; the worker executes its verdict.

``DurableLearningWorker._handle_window_failure`` turns a failed window into one
of a closed set of actions: yield the turn, defer setup, suspend, recompute,
hold or park the input, wait for the provider, or retry. This table pins each
exception class and chain shape the handler distinguishes to its observable
effects: the storage call, the metric, the log line and the returned outcome.

The rows were built from the handler's branches (busy input, readiness,
overflow suspension, changed evidence, input budget with its saved-output park,
provider outage, ordinary retry) and their precedence. The precedence rows
matter most: a chain carrying two signals must keep resolving to the same one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from reflexio.server.llm._provider_concurrency import ProviderCapSaturatedError
from reflexio.server.llm.input_budget import (
    ComputationEvidenceChangedError,
    ExtractionInputIntegrityError,
    InputBudgetConfigurationError,
    InputBudgetExceededError,
    OutputTruncatedRequestError,
    ProviderInputOverflowError,
    SavedOutputIncompatibleError,
)
from reflexio.server.llm.provider_circuit import ProviderCircuitOpenError
from reflexio.server.services.durable_learning import worker as worker_module
from reflexio.server.services.durable_learning.window_disposition import (
    DeferSetup,
    HoldInput,
    ProviderWait,
    Recompute,
    Retry,
    Suspend,
    WindowDisposition,
    YieldTurn,
    classify_window_failure,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.input_preparation import (
    ExtractionInputBudgetExceededError,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowSuspendedError,
)
from reflexio.server.services.extraction.overflow_presentation import (
    ProtectedQuoteAmbiguityError,
)
from reflexio.server.services.storage.error import (
    ExecutionInputBusyError,
    ReadinessUnavailableError,
)
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    MAX_EXTRACTION_WINDOW_ATTEMPTS,
    PROVIDER_WAIT_PREFIX,
    LeaseLostError,
    OverflowAttemptBudgetExceededError,
    Window,
)

CONFIG = "input_budget_configuration_error"
EXCEEDED = "input_budget_exceeded"
INTEGRITY = "input_integrity_error"
W, INFO, E = logging.WARNING, logging.INFO, logging.ERROR


def _window() -> Window:
    return Window(
        window_id="w1",
        user_id="u1",
        kind="profile",
        project_id="",
        predecessor=0,
        end_seq=1,
        manifest=[],
        policy={},
        force=False,
        skip_aggregation=False,
    )


def _from(outer: BaseException, inner: BaseException) -> BaseException:
    """``raise outer from inner``, without a traceback (locator ``unknown``)."""
    outer.__cause__ = inner
    return outer


def _during(outer: BaseException, inner: BaseException) -> BaseException:
    """``outer`` raised while handling ``inner`` (an implicit context link)."""
    outer.__context__ = inner
    return outer


class _Storage:
    """Records the storage calls the handler makes, minus window and token."""

    org_id = "org1"

    def __init__(
        self, *, input_parked: bool = False, retry: bool | type[Exception] = False
    ) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self._input_parked = input_parked
        self._retry = retry

    def yield_extraction_turn(self, user_id: str, token: str) -> None:
        self.calls.append(("yield_extraction_turn", user_id))

    def defer_extraction_setup(self, user_id: str, token: str) -> None:
        self.calls.append(("defer_extraction_setup", user_id))

    def suspend_extraction(self, window: Window, token: str) -> None:
        self.calls.append(("suspend_extraction",))

    def discard_extraction_computation(self, window: Window, token: str) -> None:
        self.calls.append(("discard_extraction_computation",))

    def fail_extraction_input_budget(
        self, window: Window, token: str, reason: str, *, park: bool = False
    ) -> bool:
        self.calls.append(("fail_extraction_input_budget", reason, park))
        return self._input_parked

    def retry_extraction(
        self, window: Window, token: str, error: str, **kwargs: Any
    ) -> bool:
        self.calls.append(("retry_extraction", error, kwargs))
        if isinstance(self._retry, type):
            raise self._retry("lease lost")
        return self._retry

    def retry_extraction_effects(self, window: Window) -> None:
        self.calls.append(("retry_extraction_effects",))


@dataclass(frozen=True)
class Case:
    id: str
    make: Callable[[], BaseException]
    outcome: str
    calls: list[tuple[Any, ...]]
    health: list[tuple[str, dict[str, str]]]
    logs: list[tuple[int, str]] = field(default_factory=list)
    reports: list[bool] = field(default_factory=list)
    storage: dict[str, Any] = field(default_factory=dict)


def _held(label: str, error: str) -> tuple[int, str]:
    reason = CONFIG if label == INTEGRITY else label
    return (
        W,
        f"Extraction input held kind=profile reason={label} status_reason={reason} "
        f"error={error} at=unknown",
    )


def _parked_input(error: str, label: str = CONFIG) -> tuple[int, str]:
    return (
        E,
        f"Extraction input parked kind=profile reason={label} status_reason={CONFIG} "
        f"error={error} at=unknown",
    )


def _failed(error: str) -> tuple[int, str]:
    return (W, f"Extraction window failed kind=profile error={error} at=unknown")


def _recompute(error: str) -> tuple[int, str]:
    return (
        W,
        "Extraction outcome discarded for recompute kind=profile "
        f"reason=computation_evidence_changed error={error} at=unknown",
    )


def _waiting(error: str) -> tuple[int, str]:
    return (
        INFO,
        "Extraction window waiting for provider kind=profile retry_in_seconds=0 "
        f"error={error} at=unknown",
    )


_BUSY = (
    [("yield_extraction_turn", "u1")],
    [("worker.retries", {"phase": "input_busy"})],
)
_INPUT_FAILURES = ("worker.input_budget_failures",)
_DISCARD = (
    [("discard_extraction_computation",)],
    [("worker.computation_discards", {"extractor": "profile"})],
)


def _retry_calls(error: str) -> list[tuple[Any, ...]]:
    return [("retry_extraction", f"{error}@unknown", {})]


_WINDOW_RETRY = [("worker.retries", {"phase": "window"})]


CASES: list[Case] = [
    # -- busy input: explicit __cause__ links only, and it outranks everything.
    Case("busy", lambda: ExecutionInputBusyError("x"), "input_busy", *_BUSY),
    Case(
        "busy-wrapped-by-cause",
        lambda: _from(RuntimeError("x"), ExecutionInputBusyError("x")),
        "input_busy",
        *_BUSY,
    ),
    Case(
        "busy-beats-readiness",
        lambda: _from(ReadinessUnavailableError("x"), ExecutionInputBusyError("x")),
        "input_busy",
        *_BUSY,
    ),
    Case(
        "busy-beats-input-budget",
        lambda: _from(InputBudgetConfigurationError("x"), ExecutionInputBusyError("x")),
        "input_busy",
        *_BUSY,
    ),
    Case(
        "busy-beats-outage",
        lambda: _from(ProviderCapSaturatedError("x"), ExecutionInputBusyError("x")),
        "input_busy",
        *_BUSY,
    ),
    Case(
        "busy-via-context-is-not-busy",
        lambda: _during(RuntimeError("x"), ExecutionInputBusyError("x")),
        "failure",
        _retry_calls("RuntimeError"),
        _WINDOW_RETRY,
        [_failed("RuntimeError")],
    ),
    # -- readiness and overflow suspension: the outermost exception only.
    Case(
        "readiness",
        lambda: ReadinessUnavailableError("x"),
        "readiness_deferred",
        [("defer_extraction_setup", "u1")],
        [("worker.retries", {"phase": "readiness"})],
        reports=[True],
    ),
    Case(
        "readiness-beats-input-budget",
        lambda: _from(ReadinessUnavailableError("x"), InputBudgetExceededError("x")),
        "readiness_deferred",
        [("defer_extraction_setup", "u1")],
        [("worker.retries", {"phase": "readiness"})],
        reports=[True],
    ),
    Case(
        "readiness-wrapped-retries",
        lambda: _from(RuntimeError("x"), ReadinessUnavailableError("x")),
        "failure",
        _retry_calls("RuntimeError"),
        _WINDOW_RETRY,
        [_failed("RuntimeError")],
    ),
    Case(
        "suspended",
        lambda: OverflowSuspendedError("x"),
        "suspended",
        [("suspend_extraction",)],
        [],
    ),
    Case(
        "suspended-beats-evidence-changed",
        lambda: _from(
            OverflowSuspendedError("x"), ComputationEvidenceChangedError("x")
        ),
        "suspended",
        [("suspend_extraction",)],
        [],
    ),
    Case(
        "suspended-wrapped-retries",
        lambda: _from(RuntimeError("x"), OverflowSuspendedError("x")),
        "failure",
        _retry_calls("RuntimeError"),
        _WINDOW_RETRY,
        [_failed("RuntimeError")],
    ),
    # -- changed evidence anywhere in the chain: discard and recompute.
    Case(
        "evidence-changed",
        lambda: ComputationEvidenceChangedError("x"),
        "input_budget_failure",
        *_DISCARD,
        [_recompute("ComputationEvidenceChangedError")],
    ),
    Case(
        "evidence-changed-wrapped",
        lambda: _from(RuntimeError("x"), ComputationEvidenceChangedError("x")),
        "input_budget_failure",
        *_DISCARD,
        [_recompute("RuntimeError")],
    ),
    Case(
        "evidence-changed-via-context",
        lambda: _during(RuntimeError("x"), ComputationEvidenceChangedError("x")),
        "input_budget_failure",
        *_DISCARD,
        [_recompute("RuntimeError")],
    ),
    Case(
        "evidence-changed-beats-exceeded",
        lambda: _from(
            InputBudgetExceededError("x"), ComputationEvidenceChangedError("x")
        ),
        "input_budget_failure",
        *_DISCARD,
        [_recompute("InputBudgetExceededError")],
    ),
    # -- input budget: configuration retries with backoff; capacity holds.
    Case(
        "configuration",
        lambda: InputBudgetConfigurationError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "InputBudgetConfigurationError")],
    ),
    Case(
        "configuration-at-attempt-ceiling",
        lambda: InputBudgetConfigurationError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_parked_input("InputBudgetConfigurationError")],
        storage={"input_parked": True},
    ),
    Case(
        "configuration-wrapped",
        lambda: _from(RuntimeError("x"), InputBudgetConfigurationError("x")),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "RuntimeError")],
    ),
    Case(
        "configuration-via-context",
        lambda: _during(RuntimeError("x"), InputBudgetConfigurationError("x")),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "RuntimeError")],
    ),
    Case(
        "saved-output-incompatible-parks",
        lambda: SavedOutputIncompatibleError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, True)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_parked_input("SavedOutputIncompatibleError", INTEGRITY)],
        storage={"input_parked": True},
    ),
    Case(
        "configuration-from-saved-output-parks",
        lambda: _from(
            InputBudgetConfigurationError("x"), SavedOutputIncompatibleError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, True)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_parked_input("InputBudgetConfigurationError")],
        storage={"input_parked": True},
    ),
    # -- integrity (non-capacity): the configuration hold, public reason and
    # park outcome, under its own internal label.
    Case(
        "integrity",
        lambda: ExtractionInputIntegrityError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "ExtractionInputIntegrityError")],
    ),
    Case(
        "integrity-at-attempt-ceiling",
        lambda: ExtractionInputIntegrityError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_parked_input("ExtractionInputIntegrityError", INTEGRITY)],
        storage={"input_parked": True},
    ),
    Case(
        "integrity-wrapped",
        lambda: _from(RuntimeError("x"), ExtractionInputIntegrityError("x")),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "RuntimeError")],
    ),
    Case(
        "integrity-via-context",
        lambda: _during(RuntimeError("x"), ExtractionInputIntegrityError("x")),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "RuntimeError")],
    ),
    Case(
        "integrity-from-saved-output-parks",
        lambda: _from(
            ExtractionInputIntegrityError("x"), SavedOutputIncompatibleError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, True)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_parked_input("ExtractionInputIntegrityError", INTEGRITY)],
        storage={"input_parked": True},
    ),
    Case(
        "outermost-budget-link-wins-integrity",
        lambda: _from(
            ExtractionInputIntegrityError("x"), InputBudgetConfigurationError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "ExtractionInputIntegrityError")],
    ),
    Case(
        "outermost-budget-link-wins-over-integrity",
        lambda: _from(
            InputBudgetConfigurationError("x"), ExtractionInputIntegrityError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "InputBudgetConfigurationError")],
    ),
    Case(
        "exceeded-beats-inner-integrity",
        lambda: _from(
            InputBudgetExceededError("x"), ExtractionInputIntegrityError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "InputBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "protected-quote-ambiguity-is-integrity",
        lambda: ProtectedQuoteAmbiguityError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "ProtectedQuoteAmbiguityError")],
    ),
    Case(
        "integrity-beats-outage",
        lambda: _from(
            ExtractionInputIntegrityError("x"), ProviderCapSaturatedError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": INTEGRITY})],
        [_held(INTEGRITY, "ExtractionInputIntegrityError")],
    ),
    Case(
        "exceeded",
        lambda: InputBudgetExceededError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "InputBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "output-truncated-is-exceeded",
        lambda: OutputTruncatedRequestError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "OutputTruncatedRequestError")],
        storage={"input_parked": True},
    ),
    Case(
        "provider-input-overflow-is-exceeded",
        lambda: ProviderInputOverflowError({"model": "m"}),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "ProviderInputOverflowError")],
        storage={"input_parked": True},
    ),
    Case(
        "extraction-input-exceeded",
        lambda: ExtractionInputBudgetExceededError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "ExtractionInputBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "overflow-attempts-exceeded",
        lambda: OverflowAttemptBudgetExceededError("x"),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "OverflowAttemptBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "outermost-budget-link-wins-exceeded",
        lambda: _from(
            InputBudgetExceededError("x"), InputBudgetConfigurationError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "InputBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "outermost-budget-link-wins-configuration",
        lambda: _from(
            InputBudgetConfigurationError("x"), InputBudgetExceededError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "InputBudgetConfigurationError")],
    ),
    Case(
        "exceeded-from-saved-output-does-not-park",
        lambda: _from(InputBudgetExceededError("x"), SavedOutputIncompatibleError("x")),
        "input_budget_failure",
        [("fail_extraction_input_budget", EXCEEDED, False)],
        [(_INPUT_FAILURES[0], {"reason": EXCEEDED})],
        [_held(EXCEEDED, "InputBudgetExceededError")],
        storage={"input_parked": True},
    ),
    Case(
        "input-budget-beats-outage",
        lambda: _from(
            InputBudgetConfigurationError("x"), ProviderCapSaturatedError("x")
        ),
        "input_budget_failure",
        [("fail_extraction_input_budget", CONFIG, False)],
        [(_INPUT_FAILURES[0], {"reason": CONFIG})],
        [_held(CONFIG, "InputBudgetConfigurationError")],
    ),
    # -- provider outage: wait, uncharged.
    Case(
        "cap-saturated-waits",
        lambda: ProviderCapSaturatedError("x"),
        "provider_wait",
        [
            (
                "retry_extraction",
                f"{PROVIDER_WAIT_PREFIX}ProviderCapSaturatedError@unknown",
                {"wait_seconds": 0.0},
            )
        ],
        [("worker.retries", {"phase": "provider_wait"})],
        [_waiting("ProviderCapSaturatedError")],
    ),
    Case(
        "cap-saturated-wrapped-waits",
        lambda: _from(RuntimeError("x"), ProviderCapSaturatedError("x")),
        "provider_wait",
        [
            (
                "retry_extraction",
                f"{PROVIDER_WAIT_PREFIX}RuntimeError@unknown",
                {"wait_seconds": 0.0},
            )
        ],
        [("worker.retries", {"phase": "provider_wait"})],
        [_waiting("RuntimeError")],
    ),
    Case(
        "provider-wait-lease-lost-retries-effects",
        lambda: ProviderCapSaturatedError("x"),
        "failure",
        [
            (
                "retry_extraction",
                f"{PROVIDER_WAIT_PREFIX}ProviderCapSaturatedError@unknown",
                {"wait_seconds": 0.0},
            ),
            ("retry_extraction_effects",),
        ],
        [("worker.retries", {"phase": "effects"})],
        [_waiting("ProviderCapSaturatedError")],
        storage={"retry": LeaseLostError},
    ),
    Case(
        "circuit-refusal-without-route-retries",
        lambda: ProviderCircuitOpenError("x"),
        "failure",
        _retry_calls("ProviderCircuitOpenError"),
        _WINDOW_RETRY,
        [_failed("ProviderCircuitOpenError")],
    ),
    # -- everything else is the window's own failure: charged retry.
    Case(
        "ordinary",
        lambda: ValueError("x"),
        "failure",
        _retry_calls("ValueError"),
        _WINDOW_RETRY,
        [_failed("ValueError")],
    ),
    Case(
        "ordinary-at-attempt-ceiling-parks",
        lambda: _from(RuntimeError("x"), KeyError("x")),
        "parked",
        _retry_calls("RuntimeError"),
        [],
        [
            _failed("RuntimeError"),
            (
                E,
                f"Extraction window parked after {MAX_EXTRACTION_WINDOW_ATTEMPTS} "
                "attempts kind=profile error=RuntimeError cause=KeyError at=unknown",
            ),
        ],
        storage={"retry": True},
    ),
    Case(
        "ordinary-lease-lost-retries-effects",
        lambda: ValueError("x"),
        "failure",
        [*_retry_calls("ValueError"), ("retry_extraction_effects",)],
        [("worker.retries", {"phase": "effects"})],
        [_failed("ValueError")],
        storage={"retry": LeaseLostError},
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_each_failure_shape_has_its_effects(
    case: Case, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    health: list[tuple[str, dict[str, str]]] = []
    reports: list[bool] = []
    monkeypatch.setattr(
        worker_module,
        "record_health",
        lambda name, **labels: health.append((name, labels)),
    )
    monkeypatch.setattr(
        worker_module,
        "_report_turn_failure",
        lambda _org, _user, _exc, *, readiness_deferred=False: reports.append(
            readiness_deferred
        ),
    )
    storage = _Storage(**case.storage)

    with caplog.at_level(logging.INFO, logger=worker_module.__name__):
        outcome = DurableLearningWorker._handle_window_failure(
            cast(BaseStorage, storage), _window(), "tok", cast(Exception, case.make())
        )

    assert outcome == case.outcome
    assert storage.calls == case.calls
    assert health == case.health
    assert [
        (r.levelno, r.getMessage())
        for r in caplog.records
        if r.name == worker_module.__name__
    ] == case.logs
    assert reports == case.reports


_HOLD_CONFIG = HoldInput(CONFIG, park=False)
_PARK_CONFIG = HoldInput(CONFIG, park=True)
_HOLD_EXCEEDED = HoldInput(EXCEEDED, park=False)
_HOLD_INTEGRITY = HoldInput(INTEGRITY, park=False)
_PARK_INTEGRITY = HoldInput(INTEGRITY, park=True)

#: The verdict for each row above. The effects table was pinned against the
#: worker's if-chain before this module existed, so the pair proves the move
#: kept every outcome.
DISPOSITIONS: dict[str, WindowDisposition] = {
    "busy": YieldTurn(),
    "busy-wrapped-by-cause": YieldTurn(),
    "busy-beats-readiness": YieldTurn(),
    "busy-beats-input-budget": YieldTurn(),
    "busy-beats-outage": YieldTurn(),
    "busy-via-context-is-not-busy": Retry(),
    "readiness": DeferSetup(),
    "readiness-beats-input-budget": DeferSetup(),
    "readiness-wrapped-retries": Retry(),
    "suspended": Suspend(),
    "suspended-beats-evidence-changed": Suspend(),
    "suspended-wrapped-retries": Retry(),
    "evidence-changed": Recompute(),
    "evidence-changed-wrapped": Recompute(),
    "evidence-changed-via-context": Recompute(),
    "evidence-changed-beats-exceeded": Recompute(),
    "configuration": _HOLD_CONFIG,
    "configuration-at-attempt-ceiling": _HOLD_CONFIG,
    "configuration-wrapped": _HOLD_CONFIG,
    "configuration-via-context": _HOLD_CONFIG,
    "saved-output-incompatible-parks": _PARK_INTEGRITY,
    "configuration-from-saved-output-parks": _PARK_CONFIG,
    "integrity": _HOLD_INTEGRITY,
    "integrity-at-attempt-ceiling": _HOLD_INTEGRITY,
    "integrity-wrapped": _HOLD_INTEGRITY,
    "integrity-via-context": _HOLD_INTEGRITY,
    "integrity-from-saved-output-parks": _PARK_INTEGRITY,
    "outermost-budget-link-wins-integrity": _HOLD_INTEGRITY,
    "outermost-budget-link-wins-over-integrity": _HOLD_CONFIG,
    "exceeded-beats-inner-integrity": _HOLD_EXCEEDED,
    "protected-quote-ambiguity-is-integrity": _HOLD_INTEGRITY,
    "integrity-beats-outage": _HOLD_INTEGRITY,
    "exceeded": _HOLD_EXCEEDED,
    "output-truncated-is-exceeded": _HOLD_EXCEEDED,
    "provider-input-overflow-is-exceeded": _HOLD_EXCEEDED,
    "extraction-input-exceeded": _HOLD_EXCEEDED,
    "overflow-attempts-exceeded": _HOLD_EXCEEDED,
    "outermost-budget-link-wins-exceeded": _HOLD_EXCEEDED,
    "outermost-budget-link-wins-configuration": _HOLD_CONFIG,
    "exceeded-from-saved-output-does-not-park": _HOLD_EXCEEDED,
    "input-budget-beats-outage": _HOLD_CONFIG,
    "cap-saturated-waits": ProviderWait(0.0),
    "cap-saturated-wrapped-waits": ProviderWait(0.0),
    "provider-wait-lease-lost-retries-effects": ProviderWait(0.0),
    "circuit-refusal-without-route-retries": Retry(),
    "ordinary": Retry(),
    "ordinary-at-attempt-ceiling-parks": Retry(),
    "ordinary-lease-lost-retries-effects": Retry(),
}


def test_every_row_has_a_verdict() -> None:
    assert sorted(DISPOSITIONS) == sorted(case.id for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_each_failure_shape_has_its_verdict(case: Case) -> None:
    assert classify_window_failure(case.make()) == DISPOSITIONS[case.id]


def test_every_verdict_is_exercised() -> None:
    """A verdict no row reaches would go untested."""
    kinds = {type(verdict) for verdict in DISPOSITIONS.values()}
    assert kinds == {
        YieldTurn,
        DeferSetup,
        Suspend,
        Recompute,
        HoldInput,
        ProviderWait,
        Retry,
    }


def test_from_none_severs_the_chain() -> None:
    """``raise ... from None`` hides the handled error on purpose.

    The chain is read by ``exception_chain`` (Python's traceback rule), so a
    deliberately severed input-budget error is not the window's verdict. The
    worker's earlier private walk followed the suppressed context; no raise
    site on the window path severs a chain over these errors.
    """
    severed = RuntimeError("x")
    severed.__context__ = InputBudgetConfigurationError("x")
    severed.__suppress_context__ = True
    assert classify_window_failure(severed) == Retry()


def test_integrity_holds_keep_the_public_reason() -> None:
    """Owner decision: the persisted ``last_error`` and public ``reason`` stay.

    Clients wait on ``input_budget_configuration_error`` (the documented
    learning-status example), so every hold that is not a capacity overrun
    reports it, whatever its internal label.
    """
    for case_id, verdict in DISPOSITIONS.items():
        if isinstance(verdict, HoldInput):
            expected = EXCEEDED if verdict.label == EXCEEDED else CONFIG
            assert verdict.reason == expected, case_id
    assert HoldInput(INTEGRITY, park=False).reason == CONFIG
    assert HoldInput(INTEGRITY, park=True).reason == CONFIG


def test_integrity_is_held_exactly_like_configuration() -> None:
    """Same storage calls and outcome; only the label differs."""
    pairs = [
        ("integrity", "configuration"),
        ("integrity-at-attempt-ceiling", "configuration-at-attempt-ceiling"),
        ("integrity-wrapped", "configuration-wrapped"),
        ("integrity-via-context", "configuration-via-context"),
        ("integrity-from-saved-output-parks", "configuration-from-saved-output-parks"),
        ("integrity-beats-outage", "input-budget-beats-outage"),
    ]
    by_id = {case.id: case for case in CASES}
    for integrity, configuration in pairs:
        left, right = by_id[integrity], by_id[configuration]
        assert (left.outcome, left.calls, left.storage) == (
            right.outcome,
            right.calls,
            right.storage,
        )
        assert left.health != right.health
        left_verdict = DISPOSITIONS[integrity]
        right_verdict = DISPOSITIONS[configuration]
        assert isinstance(left_verdict, HoldInput)
        assert isinstance(right_verdict, HoldInput)
        assert (left_verdict.reason, left_verdict.park) == (
            right_verdict.reason,
            right_verdict.park,
        )
