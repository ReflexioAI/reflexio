"""A terminal failure must name the exception that produced it.

Seven exception classes converge on the single ``infrastructure_failure``
reason. The reason vocabulary is a persisted contract and stays exactly as it
is, so these tests assert that the *diagnostic* recorded alongside it tells the
classes apart, reaches the wrapped cause underneath them, and carries no
customer content.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from reflexio.server.services.offline_tuner.open_world.diagnostics import (
    MAX_CHAIN_DEPTH,
    MAX_VALIDATION_ERRORS,
    TERMINAL_DIAGNOSTIC_METADATA_KEY,
    build_terminal_failure_diagnostic,
    diagnostic_log_fragment,
    diagnostic_payload,
)
from reflexio.server.services.offline_tuner.open_world.models import DiscoveryMemo
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldInvocationLeaseConflictError,
    StorageError,
)
from tests.server.services.offline_tuner.open_world.test_rule_evidence_native import (
    source as source,
)
from tests.server.services.offline_tuner.open_world.test_runner import (
    runtime as runtime,
)

# A string that stands in for evidence text, a transcript, or a memo body. It
# must never appear in anything the runner records.
SENTINEL = "CUSTOMER-TRANSCRIPT-DO-NOT-RECORD"


def test_new_method_failure_uses_reserved_content_free_diagnostic(runtime):
    runner, transport, storage, kwargs = runtime

    def fail(**_request):
        raise StorageError(SENTINEL)

    transport.complete = fail
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert SENTINEL not in json.dumps(storage.failure)
    assert (
        storage.failure[TERMINAL_DIAGNOSTIC_METADATA_KEY]["chain"][0]["exception_class"]
        == "StorageError"
    )


def test_advisory_diagnostic_failure_does_not_change_terminal_reason(runtime):
    runner, transport, storage, kwargs = runtime

    def fail(**_request):
        raise StorageError(SENTINEL)

    def diagnostic_unavailable(*_args, **_kwargs):
        raise StorageError("diagnostic store unavailable")

    transport.complete = fail
    storage.merge_playbook_optimization_job_metadata = diagnostic_unavailable
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    assert storage.job.stage == "failed"


def _discovery_memo_validation_error(payload: dict[str, Any]) -> ValidationError:
    """Produce the real defect: a provider memo that fails its own schema."""
    try:
        DiscoveryMemo.model_validate(payload)
    except ValidationError as exc:
        return exc
    raise AssertionError("payload unexpectedly validated")


def test_chain_depth_is_bounded() -> None:
    current: BaseException = StorageError("root")
    for index in range(40):
        wrapper = StorageError(f"layer {index}")
        wrapper.__cause__ = current
        current = wrapper

    diagnostic = build_terminal_failure_diagnostic("infrastructure_failure", current)

    # A literal ceiling, not the constant under test: asserting against
    # MAX_CHAIN_DEPTH alone would stay green if the bound were raised.
    assert len(diagnostic.chain) <= 8
    assert len(diagnostic.chain) == MAX_CHAIN_DEPTH


def test_a_cyclic_chain_terminates() -> None:
    first = StorageError("first")
    second = StorageError("second")
    first.__cause__ = second
    second.__cause__ = first

    diagnostic = build_terminal_failure_diagnostic("infrastructure_failure", first)

    assert len(diagnostic.chain) == 2


def test_validation_error_count_is_bounded() -> None:
    payload: dict[str, Any] = {f"unexpected_{index}": index for index in range(40)}
    validation_error = _discovery_memo_validation_error(payload)

    diagnostic = build_terminal_failure_diagnostic(
        "infrastructure_failure", validation_error
    )
    payload_json = json.dumps(diagnostic_payload(diagnostic))

    assert len(diagnostic.chain) == 1
    # Literal ceiling first, for the same reason as the chain-depth bound.
    assert len(diagnostic.chain[0].validation_errors) <= 16
    assert len(diagnostic.chain[0].validation_errors) == MAX_VALIDATION_ERRORS
    assert len(payload_json) < 4000, len(payload_json)


def test_validation_error_count_reports_the_pre_truncation_total() -> None:
    """A truncated tail must not read as an absent one.

    This is the gap the org-43 incident exposed. ``DiscoveryMemo`` has nine
    required fields, so a response of the wrong SHAPE spends nine of the ten
    listed slots on ``missing`` before describing a single model-chosen extra
    key. The recorded list then ends after exactly one ``extra_forbidden``
    entry, which reads as "the model emitted one unexpected key" when the
    truth may be "the model emitted forty and the list ran out".

    The count is what tells those apart, so it is asserted against the number
    of errors pydantic actually raised -- not against the truncated list, which
    is the thing under suspicion.
    """
    payload: dict[str, Any] = {f"unexpected_{index}": index for index in range(40)}
    validation_error = _discovery_memo_validation_error(payload)
    raised = len(validation_error.errors())

    diagnostic = build_terminal_failure_diagnostic(
        "infrastructure_failure", validation_error
    )
    link = diagnostic.chain[0]

    # 40 unexpected keys plus the nine required fields none of them supply.
    assert raised == 49, raised
    assert link.validation_error_count == raised
    # The count must be the UNtruncated total, or it says nothing the list did
    # not already say.
    assert link.validation_error_count > len(link.validation_errors)
    assert len(link.validation_errors) == MAX_VALIDATION_ERRORS

    # It has to survive serialization, because the stored row is what a reader
    # actually sees.
    stored = diagnostic_payload(diagnostic)
    assert stored["chain"][0]["validation_error_count"] == raised
    # An integer count carries no provider content, so the record stays small.
    assert len(json.dumps(stored)) < 4000


def test_validation_error_count_is_zero_for_a_non_validation_error() -> None:
    """Only a pydantic ValidationError has a count to report."""
    diagnostic = build_terminal_failure_diagnostic(
        "infrastructure_failure", StorageError("read failed")
    )
    link = diagnostic.chain[0]

    assert link.validation_errors == ()
    assert link.validation_error_count == 0
    assert diagnostic_payload(diagnostic)["chain"][0]["validation_error_count"] == 0


# --- the candidate arm's terminal outcome ------------------------------------

# ``_run`` catches ``(TypeError, ValueError)`` around each of its three analyst
# calls. Two of them attribute the failure -- ``as exc``, reason
# ``analyst_unqualified``. The candidate arm was left as a bare
# ``except (TypeError, ValueError): return self._abstain(...)``, which is wrong
# twice over:
#
#   * it is reachable ONLY after ``discovery.memo.decision ==
#     "grounded_hypothesis"``, and that memo is durably persisted by then, so
#     ``_abstain`` recorded ``no_grounded_hypothesis`` about an attempt whose
#     grounded hypothesis is on disk -- a job row that contradicts its own
#     artifacts;
#   * no ``as exc``, so ``_record_terminal_diagnostic`` was handed
#     ``cause=None`` and the exception class was lost entirely. For a reason
#     several classes converge on, that is the whole attribution.


def test_an_infrastructure_refusal_is_named_in_the_log_detail_and_record(runtime):
    """Which of the six refusals behind ``infrastructure_failure`` fired.

    In one observed run most attempts ended
    ``terminal_failure_infrastructure_failure`` with no way to tell, from the
    logs, a lost lease from an integrity refusal. The persisted reason stays
    as it is; the refusal and the wrapped cause ride beside it.
    """
    runner, transport, storage, kwargs = runtime

    def fail(**_request):
        try:
            raise TimeoutError(SENTINEL)
        except TimeoutError as cause:
            raise OpenWorldInvocationLeaseConflictError(SENTINEL) from cause

    transport.complete = fail
    result = runner.run(**kwargs)
    assert result.reason == "infrastructure_failure"
    detail = result.failure_detail
    assert detail is not None
    assert detail.startswith(
        "refusal=lease_lost exception_type=OpenWorldInvocationLeaseConflictError "
    ), detail
    assert " cause_type=TimeoutError " in detail, detail
    assert SENTINEL not in detail
    record = storage.failure[TERMINAL_DIAGNOSTIC_METADATA_KEY]
    assert record["refusal"] == "lease_lost"
    assert record["schema_version"] == 3


def test_a_non_refusal_failure_says_so():
    diagnostic = build_terminal_failure_diagnostic(
        "infrastructure_failure", StorageError(SENTINEL)
    )
    assert diagnostic.refusal is None
    assert diagnostic_payload(diagnostic)["refusal"] is None
    fragment = diagnostic_log_fragment(diagnostic)
    assert fragment.startswith("refusal=none exception_type=StorageError raised_at=")
    assert fragment.endswith(" cause_type=none cause_at=none")
    assert SENTINEL not in fragment
