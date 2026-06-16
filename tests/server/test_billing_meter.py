from unittest.mock import patch

from reflexio.server.billing_meter import (
    record_applied_learnings,
    record_extraction_tokens,
    record_injection_events,
    record_learnings_generated,
)

HOOK = "reflexio.server.billing_meter.record_usage_event"


def test_record_extraction_tokens_emits_event_when_platform_llm():
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1", billing_input_tokens=1100, prompt_tokens=1200,
            completion_tokens=300, platform_llm=True, platform_storage=None, pipeline="profile",
        )
    hook.assert_called_once()
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "extraction_tokens"
    assert kwargs["event_category"] == "learning"
    assert kwargs["count_value"] == 1100   # billing_input_tokens is the metered count
    assert kwargs["prompt_tokens"] == 1200
    assert kwargs["platform_llm"] is True
    assert kwargs["caller_type"] == "internal"


def test_record_extraction_tokens_still_captures_on_byo_llm():
    # Even on BYO-LLM we capture (platform_llm=False); the rating layer drops the charge.
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1", billing_input_tokens=850, prompt_tokens=900,
            completion_tokens=100, platform_llm=False, platform_storage=None,
        )
    hook.assert_called_once()
    assert hook.call_args.kwargs["platform_llm"] is False


def test_record_extraction_tokens_noop_on_negative():
    # Negative billing_input_tokens (e.g. from a corrupt trace) must not emit an event.
    with patch(HOOK) as hook:
        record_extraction_tokens(
            org_id="org1", billing_input_tokens=-1, prompt_tokens=0,
            completion_tokens=0, platform_llm=True, platform_storage=None,
        )
    hook.assert_not_called()


def test_record_extraction_tokens_noop_when_zero_input():
    with patch(HOOK) as hook:
        record_extraction_tokens(org_id="org1", billing_input_tokens=0, prompt_tokens=0,
                                 completion_tokens=0, platform_llm=True, platform_storage=None)
    hook.assert_not_called()


def test_record_learnings_generated_uses_count_value():
    with patch(HOOK) as hook:
        record_learnings_generated(org_id="org1", count=3, platform_llm=True,
                                   platform_storage=None, pipeline="playbook")
    kwargs = hook.call_args.kwargs
    assert kwargs["event_name"] == "learnings_generated"
    assert kwargs["event_category"] == "learning"
    assert kwargs["count_value"] == 3


def test_record_learnings_generated_noop_for_zero():
    with patch(HOOK) as hook:
        record_learnings_generated(org_id="org1", count=0, platform_llm=True, platform_storage=None)
    hook.assert_not_called()


def test_record_applied_learnings_billable_only_for_production_agent():
    with patch(HOOK) as hook:
        record_applied_learnings(org_id="org1", surfaced_count=4,
                                 caller_type="production_agent", platform_llm=True, platform_storage=None)
    assert hook.call_count == 1
    assert hook.call_args.kwargs["count_value"] == 4
    assert hook.call_args.kwargs["caller_type"] == "production_agent"


def test_record_applied_learnings_noop_for_dashboard():
    with patch(HOOK) as hook:
        record_applied_learnings(org_id="org1", surfaced_count=4,
                                 caller_type="dashboard", platform_llm=True, platform_storage=None)
    hook.assert_not_called()


def test_record_applied_learnings_noop_for_empty_result():
    with patch(HOOK) as hook:
        record_applied_learnings(org_id="org1", surfaced_count=0,
                                 caller_type="production_agent", platform_llm=True, platform_storage=None)
    hook.assert_not_called()


# ---------------------------------------------------------------------------
# record_injection_events (per-entity observability)
# ---------------------------------------------------------------------------


def test_record_injection_events_emits_one_event_per_entity():
    """Each surfaced entity produces exactly one ``learning_injection`` row."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[
                ("profile", "p1"),
                ("playbook", "42"),
                ("playbook", "99"),
            ],
            contents=["profile content", "playbook 42 content", "playbook 99 content"],
            request_id="r1",
            session_id="s1",
            pipeline="unified_search",
        )
    assert hook.call_count == 3
    # First call: profile p1
    first = hook.call_args_list[0].kwargs
    assert first["event_name"] == "learning_injection"
    assert first["event_category"] == "application"
    assert first["entity_type"] == "profile"
    assert first["entity_id"] == "p1"
    assert first["count_value"] == 1
    assert first["request_id"] == "r1"
    assert first["session_id"] == "s1"
    assert first["pipeline"] == "unified_search"
    # prompt_tokens is computed from content via count_input_tokens
    assert first["prompt_tokens"] is not None and first["prompt_tokens"] >= 0
    # Second call: playbook 42
    second = hook.call_args_list[1].kwargs
    assert second["entity_type"] == "playbook"
    assert second["entity_id"] == "42"
    # Third call: playbook 99
    third = hook.call_args_list[2].kwargs
    assert third["entity_id"] == "99"


def test_record_injection_events_computes_per_entity_tokens():
    """Each entity's ``prompt_tokens`` is computed from its own content."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[("playbook", "42"), ("playbook", "99")],
            contents=["short", "this is a much longer content string that should produce more tokens"],
        )
    short_tokens = hook.call_args_list[0].kwargs["prompt_tokens"]
    long_tokens = hook.call_args_list[1].kwargs["prompt_tokens"]
    assert long_tokens > short_tokens


def test_record_injection_events_noop_for_non_production_agent():
    """Caller types other than ``production_agent`` are no-ops, even with results."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="dashboard",
            entities=[("playbook", "42")],
            contents=["content"],
        )
    hook.assert_not_called()


def test_record_injection_events_noop_for_empty_entities():
    """An empty entity list is a no-op (no search results)."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[],
        )
    hook.assert_not_called()


def test_record_injection_events_zero_tokens_when_no_contents():
    """When contents is omitted, ``prompt_tokens`` is 0 (not None, not a crash)."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[("playbook", "42")],
        )
    assert hook.call_count == 1
    assert hook.call_args.kwargs["prompt_tokens"] == 0


def test_record_injection_events_tolerates_contents_shorter_than_entities():
    """When contents is shorter than entities, missing slots are treated as empty (0 tokens)."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[("playbook", "42"), ("playbook", "99")],
            contents=["only one"],
        )
    assert hook.call_count == 2
    assert hook.call_args_list[0].kwargs["prompt_tokens"] >= 0
    assert hook.call_args_list[1].kwargs["prompt_tokens"] == 0


def test_record_injection_events_propagates_metadata():
    """Metadata kwarg is applied to every emitted row."""
    with patch(HOOK) as hook:
        record_injection_events(
            org_id="org1",
            caller_type="production_agent",
            entities=[("playbook", "42"), ("playbook", "99")],
            contents=["a", "b"],
            metadata={"channel": "claude_code"},
        )
    assert hook.call_count == 2
    assert hook.call_args_list[0].kwargs["metadata"] == {"channel": "claude_code"}
    assert hook.call_args_list[1].kwargs["metadata"] == {"channel": "claude_code"}
