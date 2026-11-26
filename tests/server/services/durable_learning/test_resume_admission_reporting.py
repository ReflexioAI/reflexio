"""Resume admission keeps one registered failure/success scope and typed holds."""

import hashlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest

from reflexio.server.llm.compute_admission import (
    ComputeAdmissionError,
    ComputeTransitionInterruptedError,
)
from reflexio.server.services.durable_learning import resume_selection
from reflexio.server.services.storage.storage_base import ResumeCursor
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext

pytestmark = pytest.mark.unit


def test_resume_admission_reports_its_registered_descriptor_scope(monkeypatch):
    report = Mock()
    monkeypatch.setattr(resume_selection, "report_failure_at_interval", report)
    descriptor_key = hashlib.sha256(b"original-run").hexdigest()[:12]
    error = ValueError("Private exception text is not a scope identity")
    resume_selection._report_resume_admission(
        error, org_id="org", descriptor_key=descriptor_key
    )
    report.assert_called_once()
    args, kwargs = report.call_args
    assert args == (resume_selection.logger, "resume_admission_failed", error)
    assert kwargs["scope"] == f"durable-resume-admission:org:{descriptor_key}"
    assert kwargs["interval_seconds"] is None and kwargs["org_id"] == "org"
    assert kwargs["detail"] is resume_selection._failure_detail
    assert "Private" not in kwargs["detail"](error)


@pytest.mark.parametrize("kind", ["direct", "wrapped", "interrupted"])
def test_typed_transition_holds_do_not_enter_background_failure_reporting(
    monkeypatch, kind
):
    report = Mock()
    monkeypatch.setattr(resume_selection, "report_failure_at_interval", report)
    error = LearningTransitionBlockedError("Private transition detail")
    if kind == "wrapped":
        wrapped = ComputeAdmissionError(error)
        wrapped.__cause__ = error
        error = wrapped
    elif kind == "interrupted":
        error = ComputeTransitionInterruptedError("Value-free transition hold")
    with pytest.raises(type(error)) as raised:
        resume_selection._report_resume_admission(
            error, org_id="org", descriptor_key="a" * 12
        )
    assert raised.value is error
    report.assert_not_called()


def test_success_resets_the_same_descriptor_scope_as_failure(monkeypatch):
    failure, success = Mock(), Mock()
    monkeypatch.setattr(resume_selection, "report_failure_at_interval", failure)
    monkeypatch.setattr(resume_selection, "report_background_success", success)
    key = hashlib.sha256(b"original-run").hexdigest()[:12]
    resume_selection._report_resume_admission(
        ValueError("failure"), org_id="org", descriptor_key=key
    )
    scope = failure.call_args.kwargs["scope"]
    resume_selection._report_resume_admission(None, org_id="org", descriptor_key=key)
    success.assert_called_once_with(scope)


def test_failed_original_parent_observation_does_not_clear_admission_streak(
    monkeypatch,
):
    failure, success = Mock(), Mock()
    monkeypatch.setattr(resume_selection, "report_failure_at_interval", failure)
    monkeypatch.setattr(resume_selection, "report_background_success", success)
    storage = Mock()
    storage.org_id = "org"
    storage.claim_extraction.return_value = None
    storage._resume_candidate_page.side_effect = [
        (
            ResumeCursor(through_id="original-run"),
            {"id": "original-run", "org_id": "org", "project_id": ""},
        ),
        (ResumeCursor(), None),
    ]
    storage.read_learning_transition.return_value = SimpleNamespace(mode="open")
    storage.get_agent_run.return_value = SimpleNamespace(
        binding=SimpleNamespace(org_id="org", user_id="user", extractor_kind="profile")
    )
    storage.claim_user_extraction.return_value = None
    storage._observe_live_user_work_parent.side_effect = ValueError(
        "Owner observation failed"
    )
    context = SimpleNamespace(storage=storage, configurator=Mock())
    assert (
        resume_selection.ResumeSelection().claim_user(
            cast("RequestContext", context), worker_id="worker", lease_seconds=60
        )
        is None
    )
    failure.assert_called_once()
    admission = failure.call_args.kwargs["scope"]
    assert admission.startswith("durable-resume-admission:org:")
    assert all(call.args[0] != admission for call in success.call_args_list)


def test_unrelated_handled_transition_does_not_make_a_later_failure_held():
    from reflexio.server.services.durable_learning.learning_admission import (
        learning_transition_is_held,
    )

    hold = LearningTransitionBlockedError("Handled operational hold")
    failure = RuntimeError("Independent method failure")
    failure.__context__ = hold
    assert not learning_transition_is_held(failure)
    failure.__cause__ = hold
    assert learning_transition_is_held(failure)
