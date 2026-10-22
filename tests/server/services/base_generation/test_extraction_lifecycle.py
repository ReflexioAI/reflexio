"""Severity of extractor failures in ``ExtractionRunLifecycleMixin``.

A failed automatic window is retried forever with capped backoff, so early
attempts must not page; anything that is not a known-retried window keeps
paging on its first failure.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.base_generation._extraction_lifecycle import (
    WINDOW_FAILURE_ERROR_ATTEMPT,
    _failure_template,
)
from reflexio.server.services.base_generation_service import ExtractorExecutionError
from tests.server.services.test_base_generation_service_characterization import (
    _CharRequest,
    _CharService,
    _llm_client,
    _request_context,
)

_LOGGER = "reflexio.server.services.base_generation._extraction_lifecycle"


def _failing_service() -> _CharService:
    service = _CharService(
        _llm_client(),
        _request_context(MagicMock()),
        results_by_user={"u": RuntimeError("provider temporarily unavailable")},
    )
    service.service_config = _CharRequest(user_id="u", request_id="r")
    return service


def _failure_levels(service: _CharService, caplog: pytest.LogCaptureFixture) -> list:
    with (
        caplog.at_level(logging.WARNING, logger=_LOGGER),
        pytest.raises(ExtractorExecutionError),
    ):
        service._execute_extractor(service._load_extractor_config(), "u")
    return [
        record.levelno
        for record in caplog.records
        if record.getMessage().startswith("Extractor failed for")
    ]


@pytest.mark.parametrize(
    ("prior_attempts", "expected"),
    [
        (0, logging.WARNING),
        (WINDOW_FAILURE_ERROR_ATTEMPT - 2, logging.WARNING),
        (WINDOW_FAILURE_ERROR_ATTEMPT - 1, logging.ERROR),
        (WINDOW_FAILURE_ERROR_ATTEMPT + 5, logging.ERROR),
    ],
)
def test_window_failure_severity_follows_attempt_count(
    caplog, prior_attempts, expected
):
    service = _failing_service()
    service.set_extraction_window(
        [], service._load_extractor_config(), "w", prior_attempts=prior_attempts
    )

    assert _failure_levels(service, caplog) == [expected]


def test_failure_outside_a_window_still_pages(caplog):
    assert _failure_levels(_failing_service(), caplog) == [logging.ERROR]


def test_window_with_unknown_attempts_still_pages(caplog):
    service = _failing_service()
    service.set_extraction_window([], service._load_extractor_config(), "w")

    assert _failure_levels(service, caplog) == [logging.ERROR]


def _records(caplog: pytest.LogCaptureFixture, prefix: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith(prefix)]


def test_failure_log_template_is_constant_across_identifiers(caplog):
    """The identifier and exception text are log ARGS, never the template.

    Error trackers group on ``LogRecord.msg``; an interpolated identifier made
    every occurrence its own group.
    """
    templates = set()
    for identifier in ("rqa1FTwfHfKWMmIil7VUS3NBEbxivFV9", "u-2", "Zz9"):
        caplog.clear()
        service = _failing_service()
        service._results_by_user = {
            "u": RuntimeError(f"no structured output for {identifier}")
        }
        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            pytest.raises(ExtractorExecutionError) as raised,
        ):
            service._execute_extractor(service._load_extractor_config(), identifier)
        (record,) = _records(caplog, "Extractor failed")
        templates.add(record.msg)
        assert identifier not in record.msg
        # Rendered text and the raised error keep their exact historical wording.
        expected = (
            f"Extractor failed for profile_generation_char identifier={identifier}: "
            f"no structured output for {identifier} (type=RuntimeError)"
        )
        assert record.getMessage() == str(raised.value) == expected
    assert templates == {
        "Extractor failed for profile_generation_char identifier=%s: %s "
        "(type=RuntimeError)"
    }


def test_failure_template_keeps_exception_type_distinct(caplog):
    service = _failing_service()
    service._results_by_user = {"u": ValueError("bad")}
    with (
        caplog.at_level(logging.WARNING, logger=_LOGGER),
        pytest.raises(ExtractorExecutionError),
    ):
        service._execute_extractor(service._load_extractor_config(), "u")
    (record,) = _records(caplog, "Extractor failed")
    assert str(record.msg).endswith("(type=ValueError)")


def test_timeout_log_template_is_constant_across_identifiers(caplog, monkeypatch):
    class _Slow:
        def run(self):
            time.sleep(0.5)
            return ["late"]

    templates = set()
    for identifier in ("first-identifier-1", "second-identifier-2"):
        caplog.clear()
        service = _failing_service()
        service._create_extractor = lambda *_: _Slow()  # type: ignore[method-assign]
        monkeypatch.setattr(service, "_extractor_timeout_seconds", lambda: 0.05)
        with (
            caplog.at_level(logging.WARNING, logger=_LOGGER),
            pytest.raises(ExtractorExecutionError) as raised,
        ):
            service._execute_extractor(service._load_extractor_config(), identifier)
        (record,) = _records(caplog, "Extractor timed out")
        templates.add(record.msg)
        assert record.getMessage() == str(raised.value)
        assert identifier in record.getMessage()
    assert templates == {
        "Extractor timed out after 0.05 seconds for profile_generation_char "
        "identifier=%s"
    }


def test_failure_template_escapes_literal_percent():
    assert _failure_template("100% of ", "%s", " (type=X%)") == (
        "100%% of %s (type=X%%)"
    )


def test_failure_the_caller_reports_is_logged_at_warning(caplog):
    """A caller that retries or escalates the failure itself owns its one report."""
    service = _failing_service()
    service._caller_reports_extractor_failure = True

    assert _failure_levels(service, caplog) == [logging.WARNING]


@pytest.mark.parametrize(
    ("caller_reports", "expected"),
    [(False, logging.ERROR), (True, logging.WARNING)],
)
def test_run_generation_failure_level_follows_the_reporting_owner(
    caplog, monkeypatch, caller_reports, expected
):
    service = _failing_service()
    service._caller_reports_extractor_failure = caller_reports
    monkeypatch.setattr(
        service,
        "compute_generation",
        MagicMock(side_effect=ExtractorExecutionError("judge failed")),
    )

    with (
        caplog.at_level(
            logging.WARNING, logger="reflexio.server.services.base_generation_service"
        ),
        pytest.raises(ExtractorExecutionError),
    ):
        service._run_generation(_CharRequest(user_id="u", request_id="r"))

    assert [
        record.levelno
        for record in caplog.records
        if record.getMessage().startswith("Failed to run")
    ] == [expected]


def test_run_generation_non_extractor_failure_stays_error_when_caller_reports(
    caplog, monkeypatch
):
    """Only an ``ExtractorExecutionError`` reaches the caller; anything else is
    swallowed here, so the caller cannot report it and it stays at ERROR."""
    service = _failing_service()
    service._caller_reports_extractor_failure = True
    monkeypatch.setattr(
        service, "compute_generation", MagicMock(side_effect=RuntimeError("bug"))
    )

    with caplog.at_level(
        logging.WARNING, logger="reflexio.server.services.base_generation_service"
    ):
        service._run_generation(_CharRequest(user_id="u", request_id="r"))

    assert [
        record.levelno
        for record in caplog.records
        if record.getMessage().startswith("Failed to run")
    ] == [logging.ERROR]
