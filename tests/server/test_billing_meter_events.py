"""Tests for per-record ``learnings_generated`` events (Task A3).

Phase A of the BYOC metering redesign: ``learnings_generated`` gains an
entity-backed emission path, ``record_learnings_generated_records`` /
``emit_learnings_generated_records``, that emits ONE event per learning id
(``count_value=1``, ``event_key=f"learn:{id}"``, ``entity_id=id``) instead of
a single aggregate event with ``count_value=N``, so downstream dedup can key
on the learning id.

The existing count-based ``record_learnings_generated`` / ``emit_learnings_generated``
remain as the documented FALLBACK for callers that genuinely lack a per-record
id list (e.g. dedup/consolidation can reduce the persisted count below the raw
extracted count, so there is no safe 1:1 id per unit of ``count``). The
fallback path now also carries a synthesized ``event_key=f"learn-batch:{uuid4()}"``
so every ``learnings_generated`` event -- record-backed or batch -- has a
dedup key.

Totals are preserved in both paths: the sum of ``count_value`` across the
per-record events equals ``len(learning_ids)``; the fallback emits exactly
``count_value=N`` in one event, unchanged from before.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from reflexio.server.billing_meter import (
    emit_learnings_generated_records,
    record_learnings_generated,
    record_learnings_generated_records,
)

HOOK = "reflexio.server.billing_meter.record_usage_event"

_BATCH_KEY_RE = re.compile(r"^learn-batch:[0-9a-f-]{36}$")


def test_records_emits_one_event_per_id():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["p1", "p2", "p3"],
            platform_llm=True,
            platform_storage=None,
            pipeline="profile",
        )
    assert hook.call_count == 3
    for call, learning_id in zip(hook.call_args_list, ["p1", "p2", "p3"], strict=True):
        kwargs = call.kwargs
        assert kwargs["event_name"] == "learnings_generated"
        assert kwargs["event_category"] == "learning"
        assert kwargs["count_value"] == 1
        assert kwargs["event_key"] == f"learn:{learning_id}"
        assert kwargs["entity_id"] == learning_id


def test_records_emits_distinct_keys():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["a", "b", "c"],
            platform_llm=True,
            platform_storage=None,
        )
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert len(keys) == len(set(keys)) == 3


def test_records_totals_preserved():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["x1", "x2", "x3", "x4"],
            platform_llm=True,
            platform_storage=None,
        )
    total = sum(call.kwargs["count_value"] for call in hook.call_args_list)
    assert total == 4  # == len(learning_ids), matching the old count semantics


def test_records_noop_for_empty_list():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=[],
            platform_llm=True,
            platform_storage=None,
        )
    hook.assert_not_called()


def test_records_forwards_carried_fields():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["pb1"],
            platform_llm=False,
            platform_storage=True,
            pipeline="playbook",
            user_id="user-1",
            request_id="req-1",
            session_id="sess-1",
            source="aggregation",
            agent_version="v9",
            playbook_name="agent_rules",
            entity_type="agent_playbook",
            metadata={"k": "v"},
        )
    kwargs = hook.call_args.kwargs
    assert kwargs["pipeline"] == "playbook"
    assert kwargs["user_id"] == "user-1"
    assert kwargs["request_id"] == "req-1"
    assert kwargs["session_id"] == "sess-1"
    assert kwargs["source"] == "aggregation"
    assert kwargs["agent_version"] == "v9"
    assert kwargs["playbook_name"] == "agent_rules"
    assert kwargs["entity_type"] == "agent_playbook"
    assert kwargs["platform_llm"] is False
    assert kwargs["platform_storage"] is True
    assert kwargs["caller_type"] == "internal"
    assert kwargs["metadata"] == {"k": "v"}


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


def _configurator(config=None):
    configurator = MagicMock()
    configurator.get_config.return_value = config
    return configurator


def test_emit_records_resolves_platform_llm_and_emits_per_id():
    with patch(HOOK) as hook:
        emit_learnings_generated_records(
            org_id="org1",
            configurator=_configurator(),
            learning_ids=["r1", "r2"],
            source="reflection",
            pipeline="reflection",
        )
    assert hook.call_count == 2
    assert all(call.kwargs["platform_llm"] is True for call in hook.call_args_list)
    assert {call.kwargs["event_key"] for call in hook.call_args_list} == {
        "learn:r1",
        "learn:r2",
    }


def test_emit_records_noop_for_empty_ids():
    with patch(HOOK) as hook:
        emit_learnings_generated_records(
            org_id="org1",
            configurator=_configurator(),
            learning_ids=[],
            source="reflection",
        )
    hook.assert_not_called()


def test_emit_records_swallows_exceptions():
    configurator = MagicMock()
    configurator.get_config.side_effect = RuntimeError("config boom")
    with patch(HOOK) as hook:
        # Must not raise despite get_config() blowing up mid-emit.
        emit_learnings_generated_records(
            org_id="org1",
            configurator=configurator,
            learning_ids=["r1"],
            source="reflection",
        )
    hook.assert_not_called()
