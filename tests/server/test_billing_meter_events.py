"""Test live count-based learning and event-moment billing emissions."""

from __future__ import annotations

import re
from unittest.mock import patch

from reflexio.server.billing_meter import (
    record_applied_learnings,
    record_extraction_tokens,
    record_learnings_generated,
    record_search_request,
)

HOOK = "reflexio.server.billing_meter.record_usage_event"

_BATCH_KEY_RE = re.compile(r"^learn-batch:[0-9a-f-]{36}$")
_TOK_KEY_RE = re.compile(r"^tok:[0-9a-f-]{36}$")
_APPLIED_KEY_RE = re.compile(r"^applied:[0-9a-f-]{36}$")
_SEARCH_KEY_RE = re.compile(r"^search:[0-9a-f-]{36}$")


def test_batch_record_accepts_a_retry_stable_event_key():
    with patch(HOOK) as hook:
        record_learnings_generated(
            org_id="org1",
            count=2,
            platform_llm=True,
            platform_storage=None,
            event_key="learn-batch:resumable:run-1:profile",
        )

    assert hook.call_args.kwargs["event_key"] == ("learn-batch:resumable:run-1:profile")


def test_fallback_emits_one_synthesized_key_event_with_count_value_n():
    """The documented FALLBACK path: no ids, one event, count_value=N."""
    with patch(HOOK) as hook:
        record_learnings_generated(
            org_id="org1",
            count=5,
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_called_once()
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "learnings_generated"
    assert kwargs["count_value"] == 5
    assert _BATCH_KEY_RE.match(kwargs["event_key"])


def test_fallback_keys_are_unique_across_calls():
    with patch(HOOK) as hook:
        record_learnings_generated(
            org_id="org1", count=2, platform_llm=True, platform_storage=None
        )
        record_learnings_generated(
            org_id="org1", count=3, platform_llm=True, platform_storage=None
        )
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert len(keys) == len(set(keys)) == 2


def test_fallback_still_noops_for_zero_count():
    with patch(HOOK) as hook:
        record_learnings_generated(
            org_id="org1", count=0, platform_llm=True, platform_storage=None
        )
    hook.assert_not_called()


# --- Task A4: synthesized-key event-moment counters ------------------------


def test_extraction_tokens_emits_synthesized_key():
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1",
            billing_input_tokens=100,
            prompt_tokens=80,
            completion_tokens=20,
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_called_once()
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "extraction_tokens"
    assert kwargs["count_value"] == 100  # unchanged: billing_input_tokens
    assert _TOK_KEY_RE.match(kwargs["event_key"])


def test_extraction_tokens_two_emits_under_one_request_id_get_distinct_keys():
    """Two token emits under the SAME request_id must not collapse downstream."""
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1",
            billing_input_tokens=100,
            prompt_tokens=80,
            completion_tokens=20,
            platform_llm=True,
            platform_storage=None,
            request_id="req-shared",
        )
        record_extraction_tokens(
            org_id="org1",
            billing_input_tokens=50,
            prompt_tokens=40,
            completion_tokens=10,
            platform_llm=True,
            platform_storage=None,
            request_id="req-shared",
        )
    assert hook.call_count == 2
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert len(keys) == len(set(keys)) == 2
    for key in keys:
        assert _TOK_KEY_RE.match(key)
    counts = [call.kwargs["count_value"] for call in hook.call_args_list]
    assert counts == [100, 50]  # both counted, unchanged


def test_extraction_tokens_noop_for_zero_or_negative():
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1",
            billing_input_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_not_called()


def test_applied_learnings_emits_synthesized_key():
    with patch(HOOK) as hook:
        record_applied_learnings(
            org_id="org1",
            surfaced_count=3,
            caller_type="production_agent",
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_called_once()
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "learning_applied"
    assert kwargs["count_value"] == 3  # unchanged: surfaced_count
    assert _APPLIED_KEY_RE.match(kwargs["event_key"])


def test_applied_learnings_two_calls_get_distinct_keys():
    with patch(HOOK) as hook:
        record_applied_learnings(
            org_id="org1",
            surfaced_count=2,
            caller_type="production_agent",
            platform_llm=True,
            platform_storage=None,
            request_id="req-shared",
        )
        record_applied_learnings(
            org_id="org1",
            surfaced_count=2,
            caller_type="production_agent",
            platform_llm=True,
            platform_storage=None,
            request_id="req-shared",
        )
    assert hook.call_count == 2
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert len(keys) == len(set(keys)) == 2


def test_applied_learnings_noop_guards_unchanged():
    with patch(HOOK) as hook:
        record_applied_learnings(
            org_id="org1",
            surfaced_count=0,
            caller_type="production_agent",
            platform_llm=True,
            platform_storage=None,
        )
        record_applied_learnings(
            org_id="org1",
            surfaced_count=3,
            caller_type="dashboard",
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_not_called()


def test_search_request_emits_synthesized_key():
    with patch(HOOK) as hook:
        record_search_request(
            org_id="org1",
            caller_type="production_agent",
        )
    hook.assert_called_once()
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "search_request"
    assert kwargs["count_value"] == 1  # unchanged
    assert _SEARCH_KEY_RE.match(kwargs["event_key"])


def test_search_request_two_calls_same_args_get_distinct_keys():
    """The critical A4 guard: two identical-args search calls must produce
    TWO events with DISTINCT search: keys -- this is what prevents the
    request_id-collision under-count (two searches must never collapse into
    one billed event downstream).
    """
    with patch(HOOK) as hook:
        record_search_request(
            org_id="org1",
            caller_type="production_agent",
            request_id="req-shared",
            session_id="sess-shared",
        )
        record_search_request(
            org_id="org1",
            caller_type="production_agent",
            request_id="req-shared",
            session_id="sess-shared",
        )
    assert hook.call_count == 2
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert len(keys) == len(set(keys)) == 2
    for key in keys:
        assert _SEARCH_KEY_RE.match(key)
    counts = [call.kwargs["count_value"] for call in hook.call_args_list]
    assert counts == [1, 1]  # both counted, unchanged


def test_search_request_noop_for_non_production_agent():
    with patch(HOOK) as hook:
        record_search_request(org_id="org1", caller_type="dashboard")
    hook.assert_not_called()
