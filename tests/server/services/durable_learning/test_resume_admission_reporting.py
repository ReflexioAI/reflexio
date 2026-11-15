"""Resume admission keeps one registered failure/success scope and typed holds."""

import hashlib
from unittest.mock import Mock

import pytest

from reflexio.server.llm.compute_admission import (
    ComputeAdmissionError,
    ComputeTransitionInterruptedError,
)
from reflexio.server.services.durable_learning import resume_selection
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)

pytestmark = pytest.mark.unit


def test_resume_admission_reports_its_registered_descriptor_scope(monkeypatch):
    report = Mock()
    monkeypatch.setattr(resume_selection, "report_failure_at_interval", report)
    descriptor_key = hashlib.sha256(b"original-run").hexdigest()[:12]
    error = ValueError("Private exception text is not a scope identity")
    resume_selection._report_resume_admission_failure(
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
        resume_selection._report_resume_admission_failure(
            error, org_id="org", descriptor_key="a" * 12
        )
    assert raised.value is error
    report.assert_not_called()
