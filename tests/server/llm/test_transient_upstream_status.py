"""Transient classification by upstream HTTP status.

LiteLLM 1.103 raises a MiniMax 529 overload as a generic ``APIError`` with
``status_code=529`` and a 502 as ``BadGatewayError`` -- neither name is in
``_TRANSIENT_LLM_ERROR_NAMES``, so before the status check every MiniMax overload
logged the request-end failure at ERROR (Sentry PYTHON-FASTAPI-56). The status
must also survive the isolated-worker pickle boundary, where the outer type is
always ``LiteLLMClientError``.
"""

import logging
import multiprocessing
from unittest.mock import MagicMock, patch

import pytest
from litellm.exceptions import APIError, BadGatewayError
from pydantic import BaseModel

from reflexio.server.llm._litellm_subprocess import _snapshot_completion_error
from reflexio.server.llm._litellm_text_generation import (
    _is_expected_transient_llm_error,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
    LiteLLMConfig,
    _CompletionErrorSnapshot,
)


class _Answer(BaseModel):
    answer: str


def _api_error(status: int) -> APIError:
    return APIError(
        status_code=status,
        message='MinimaxException - {"type":"error","error":{"type":"overloaded_error"}}',
        llm_provider="minimax",
        model="MiniMax-M3",
    )


@pytest.mark.parametrize("status", [408, 429, 502, 503, 504, 529])
def test_retry_later_status_is_transient(status: int) -> None:
    assert _is_expected_transient_llm_error(_api_error(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 500])
def test_permanent_or_synthesized_status_is_not_transient(status: int) -> None:
    """4xx other than 408/429 are permanent. 500 on a generic ``APIError`` is
    LiteLLM's own default for a malformed 200 body (``choices: null``), not a
    measured outage; a genuine upstream 500 arrives as ``InternalServerError``."""
    assert not _is_expected_transient_llm_error(_api_error(status))


def test_bad_gateway_is_transient_by_status() -> None:
    error = BadGatewayError(message="bad gateway", llm_provider="minimax", model="m")
    assert _is_expected_transient_llm_error(error)


def test_status_is_found_through_the_cause_chain() -> None:
    try:
        try:
            raise _api_error(529)
        except APIError as exc:
            raise LiteLLMClientError("API call failed") from exc
    except LiteLLMClientError as wrapped:
        assert _is_expected_transient_llm_error(wrapped)


@pytest.mark.parametrize(("status", "transient"), [(529, True), (400, False)])
def test_isolated_wrapper_status_classifies(status: int, transient: bool) -> None:
    error = LiteLLMClientError(
        "litellm.completion failed in isolated worker",
        upstream_error_type="APIError",
        upstream_status_code=status,
    )
    assert _is_expected_transient_llm_error(error) is transient


def test_worker_snapshot_records_status_code() -> None:
    snapshot = _snapshot_completion_error(_api_error(529), {"model": "m"})
    assert snapshot.type_name == "APIError"
    assert snapshot.status_code == 529


def test_worker_snapshot_ignores_non_integer_status() -> None:
    error = RuntimeError("no status")
    error.status_code = "529"  # type: ignore[attr-defined]
    assert _snapshot_completion_error(error, {"model": "m"}).status_code is None


def test_isolated_worker_error_carries_status_to_parent(monkeypatch) -> None:
    """The parent re-raise keeps the upstream status, so a 529 that crossed the
    process boundary still classifies as transient."""
    client = LiteLLMClient(LiteLLMConfig(model="minimax/MiniMax-M3"))
    result_queue = MagicMock()
    result_queue.get.return_value = (
        "error",
        _CompletionErrorSnapshot(
            type_name="APIError",
            message="overloaded",
            model="MiniMax-M3",
            llm_provider="minimax",
            status_code=529,
        ),
    )
    process = MagicMock()
    process.is_alive.return_value = False
    process_context = MagicMock()
    process_context.Queue.return_value = result_queue
    process_context.Process.return_value = process
    monkeypatch.setattr(multiprocessing, "get_context", lambda: process_context)
    monkeypatch.setattr(
        client, "_should_process_isolate_completion", lambda *_args: True
    )
    params = {
        "model": "minimax/MiniMax-M3",
        "messages": [{"role": "user", "content": "hi"}],
        "timeout": 0.5,
    }

    with pytest.raises(LiteLLMClientError) as exc_info:
        client._completion_with_hard_timeout(params, hard_timeout=5.0)

    assert exc_info.value.upstream_status_code == 529
    assert _is_expected_transient_llm_error(exc_info.value)


@pytest.mark.parametrize(
    ("status", "level"), [(529, logging.WARNING), (400, logging.ERROR)]
)
def test_request_end_log_level_follows_status(caplog, status: int, level: int):
    client = LiteLLMClient(
        LiteLLMConfig(model="minimax/MiniMax-M3", max_retries=1, retry_delay=0)
    )
    with (
        patch("litellm.completion", side_effect=_api_error(status)),
        caplog.at_level(logging.DEBUG),
        pytest.raises(LiteLLMClientError),
    ):
        client.generate_chat_response(
            messages=[{"role": "user", "content": "test"}],
            response_format=_Answer,
        )

    ends = [
        r
        for r in caplog.records
        if "event=llm_request_end" in r.getMessage()
        and "success=False" in r.getMessage()
    ]
    assert ends, "expected a request-end failure log record"
    assert {r.levelno for r in ends} == {level}
