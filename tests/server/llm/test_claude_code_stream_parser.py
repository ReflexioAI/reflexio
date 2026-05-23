"""Tests for claude_code_stream_parser: NDJSON parsing + stall classification."""

from __future__ import annotations

import pytest

from reflexio.server.llm.providers.claude_code_stream_parser import (
    classify_stall,
    parse_reset_estimate,
    parse_stream_json,
)


def test_clean_stream_returns_success():
    stream = (
        '{"type":"system","subtype":"init","session_id":"s1"}\n'
        '{"type":"result","result":"ok","session_id":"s1"}\n'
    )
    result = parse_stream_json(stream, exit_code=0)
    assert result.success is True
    assert result.terminal_text == "ok"
    assert result.stall_candidate is None


def test_billing_error_in_retry_then_stream_failure_classifies_billing():
    stream = (
        '{"type":"system","subtype":"api_retry","error":"billing_error","attempt":1,"max_retries":3}\n'
        '{"type":"system","subtype":"api_retry","error":"billing_error","attempt":2,"max_retries":3}\n'
    )
    result = parse_stream_json(stream, exit_code=1)
    assert result.success is False
    assert classify_stall(result) == "billing_error"


@pytest.mark.parametrize("err", ["authentication_failed", "oauth_org_not_allowed"])
def test_auth_categories_classify_as_auth_error(err):
    stream = f'{{"type":"system","subtype":"api_retry","error":"{err}","attempt":1,"max_retries":3}}\n'
    result = parse_stream_json(stream, exit_code=1)
    assert classify_stall(result) == "auth_error"


def test_billing_error_in_retry_but_stream_succeeds_does_not_stall():
    """False-positive guard: retry surfaced billing_error but call ultimately succeeded."""
    stream = (
        '{"type":"system","subtype":"api_retry","error":"billing_error","attempt":1,"max_retries":3}\n'
        '{"type":"result","result":"ok","session_id":"s1"}\n'
    )
    result = parse_stream_json(stream, exit_code=0)
    assert result.success is True
    assert classify_stall(result) is None


@pytest.mark.parametrize("err", ["rate_limit", "server_error"])
def test_transient_errors_do_not_classify_as_stall(err):
    stream = f'{{"type":"system","subtype":"api_retry","error":"{err}","attempt":1,"max_retries":3}}\n'
    result = parse_stream_json(stream, exit_code=1)
    assert classify_stall(result) is None


def test_malformed_ndjson_returns_failure_with_no_stall():
    result = parse_stream_json("not json at all\n{broken}\n", exit_code=1)
    assert result.success is False
    assert classify_stall(result) is None


def test_text_fallback_when_no_retry_event_present():
    """When the stream has no api_retry event but stderr contains the phrase."""
    result = parse_stream_json(
        "", exit_code=1, stderr_text="hit your weekly limit · resets Mon 12:00am"
    )
    assert classify_stall(result) == "billing_error"


@pytest.mark.parametrize(
    "text,expected_hour",
    [
        ("resets 3:45pm", 15),
        ("resets Mon 12:00am", 0),
    ],
)
def test_parse_reset_estimate_extracts_time(text, expected_hour):
    parsed = parse_reset_estimate(text)
    assert parsed is not None
    assert parsed.hour == expected_hour


def test_parse_reset_estimate_returns_none_when_no_match():
    assert parse_reset_estimate("unrelated error text") is None


@pytest.mark.parametrize(
    "text",
    [
        "resets 13:00pm",  # hour > 12 in 12-hour format
        "resets 0:30am",  # hour < 1
        "resets 10:75am",  # minute > 59
    ],
)
def test_parse_reset_estimate_rejects_out_of_range(text):
    """Reject inputs the regex accepts but that aren't valid 12-hour clock times."""
    assert parse_reset_estimate(text) is None
