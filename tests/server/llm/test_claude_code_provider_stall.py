"""End-to-end test of the provider's stall-detection plumbing using a mocked subprocess."""

from __future__ import annotations

import subprocess
from datetime import UTC
from unittest.mock import patch

import pytest

from reflexio.server.llm.providers import claude_code_provider as ccp
from reflexio.server.llm.providers.claude_code_provider import (
    ClaudeCodeCLIError,
    ClaudeCodeLLM,
)


@pytest.fixture
def llm(storage):
    """ClaudeCodeLLM bound to the test storage with a resolvable fake CLI path."""
    return ClaudeCodeLLM(cli_path="/usr/local/bin/claude", storage=storage)


def _mock_run(returncode: int, stdout: str, stderr: str = ""):
    """Build a subprocess.CompletedProcess for mocking."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def test_clean_run_clears_stall(llm, storage):
    stream = '{"type":"result","result":"ok","session_id":"s"}\n'
    with patch.object(ccp.subprocess, "run", return_value=_mock_run(0, stream)):
        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert storage.get_stall_state().stalled is False


def test_billing_failure_writes_stall(llm, storage):
    stream = (
        '{"type":"system","subtype":"api_retry","error":"billing_error",'
        '"attempt":1,"max_retries":3}\n'
    )
    with patch.object(
        ccp.subprocess, "run",
        return_value=_mock_run(1, stream, "resets Mon 12:00am"),
    ), pytest.raises(ClaudeCodeCLIError):
        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )
    state = storage.get_stall_state()
    assert state.stalled is True
    assert state.reason == "billing_error"
    assert state.notified_in_cc is False


def test_transient_failure_does_not_stall(llm, storage):
    stream = (
        '{"type":"system","subtype":"api_retry","error":"rate_limit",'
        '"attempt":1,"max_retries":3}\n'
    )
    with patch.object(
        ccp.subprocess, "run", return_value=_mock_run(1, stream)
    ), pytest.raises(ClaudeCodeCLIError):
        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert storage.get_stall_state().stalled is False


def test_prior_stall_cleared_after_successful_run(llm, storage):
    """A successful completion should clear a previously recorded stall."""
    from datetime import datetime

    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=datetime.now(UTC),
        reset_estimate=None,
        error_message="old failure",
    )
    assert storage.get_stall_state().stalled is True

    stream = '{"type":"result","result":"recovered","session_id":"s"}\n'
    with patch.object(ccp.subprocess, "run", return_value=_mock_run(0, stream)):
        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )
    assert storage.get_stall_state().stalled is False
