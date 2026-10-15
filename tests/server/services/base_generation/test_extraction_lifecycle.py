"""Severity of extractor failures in ``ExtractionRunLifecycleMixin``.

A failed automatic window is retried forever with capped backoff, so early
attempts must not page; anything that is not a known-retried window keeps
paging on its first failure.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.base_generation._extraction_lifecycle import (
    WINDOW_FAILURE_ERROR_ATTEMPT,
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
