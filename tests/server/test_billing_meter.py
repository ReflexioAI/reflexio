from unittest.mock import patch

from reflexio.server.billing_meter import (
    record_applied_learnings,
    record_extraction_tokens,
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
