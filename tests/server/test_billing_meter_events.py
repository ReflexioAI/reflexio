"""Tests for per-record ``learnings_generated`` events (Task A3) and
synthesized-key event-moment counters (Task A4).

Phase A of the BYOC metering redesign: ``learnings_generated`` gains an
entity-backed emission path, ``record_learnings_generated_records`` /
``emit_learnings_generated_records``, that emits ONE event per learning id
(``count_value=1``, ``event_key=f"learn:{entity_type}:{id}"``, ``entity_id=id``)
instead of a single aggregate event with ``count_value=N``, so downstream
dedup can key on the learning id. The ``entity_type`` segment is required for
collision-freedom: ``user_playbook_id`` and ``agent_playbook_id`` are each an
autoincrement primary key in a SEPARATE table, so the same integer id can
legitimately occur in both -- without the entity-type segment those would
mint the same ``event_key`` and collapse into one event downstream.

The existing count-based ``record_learnings_generated`` / ``emit_learnings_generated``
remain for online extraction callers that have a known billable count but do
not retain per-record ids. Resumable finalization uses only the record-backed
path and skips items without durable ids. The count-based helper carries a
synthesized ``event_key=f"learn-batch:{uuid4()}"`` so every
``learnings_generated`` event -- record-backed or batch -- has a dedup key.

Totals are preserved in both paths: the sum of ``count_value`` across the
per-record events equals ``len(learning_ids)``; the fallback emits exactly
``count_value=N`` in one event, unchanged from before.

Task A4 covers the three "event-moment" counters -- ``record_extraction_tokens``,
``record_applied_learnings``, ``record_search_request`` -- which have no
durable per-record row to key on (a token batch, an applied-learnings
response, a search request are all ephemeral moments, not entities). Each now
mints a fresh ``uuid4()`` at the emit site as its ``event_key``
(``tok:``/``applied:``/``search:`` prefixes) so two calls -- even with the
exact same ``request_id`` -- produce two distinct keys and are never
collapsed by downstream dedup. ``count_value`` is unchanged.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from reflexio.server.billing_meter import (
    emit_learnings_generated_records,
    record_applied_learnings,
    record_extraction_tokens,
    record_learnings_generated,
    record_learnings_generated_records,
    record_search_request,
)

HOOK = "reflexio.server.billing_meter.record_usage_event"

_BATCH_KEY_RE = re.compile(r"^learn-batch:[0-9a-f-]{36}$")
_TOK_KEY_RE = re.compile(r"^tok:[0-9a-f-]{36}$")
_APPLIED_KEY_RE = re.compile(r"^applied:[0-9a-f-]{36}$")
_SEARCH_KEY_RE = re.compile(r"^search:[0-9a-f-]{36}$")


def test_records_emits_one_event_per_id():
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["p1", "p2", "p3"],
            platform_llm=True,
            platform_storage=None,
            pipeline="profile",
            entity_type="profile",
        )
    assert hook.call_count == 3
    for call, learning_id in zip(hook.call_args_list, ["p1", "p2", "p3"], strict=True):
        kwargs = call.kwargs
        assert kwargs["event_name"] == "learnings_generated"
        assert kwargs["event_category"] == "learning"
        assert kwargs["count_value"] == 1
        assert kwargs["event_key"] == f"learn:profile:{learning_id}"
        assert kwargs["entity_id"] == learning_id


def test_records_key_uses_placeholder_when_entity_type_is_none():
    """No entity_type -> stable '_' placeholder, never a literal 'None'."""
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["z1"],
            platform_llm=True,
            platform_storage=None,
        )
    assert hook.call_args.kwargs["event_key"] == "learn:_:z1"


def test_records_key_disambiguates_same_id_across_entity_types():
    """Regression guard for the cross-table collision finding: a
    user_playbook id and an agent_playbook id that share the same integer
    (both tables are separate AUTOINCREMENT PKs starting at 1) must produce
    DISTINCT event_keys.
    """
    with patch(HOOK) as hook:
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["11"],
            platform_llm=True,
            platform_storage=None,
            entity_type="user_playbook",
        )
        record_learnings_generated_records(
            org_id="org1",
            learning_ids=["11"],
            platform_llm=True,
            platform_storage=None,
            entity_type="agent_playbook",
        )
    keys = [call.kwargs["event_key"] for call in hook.call_args_list]
    assert keys == ["learn:user_playbook:11", "learn:agent_playbook:11"]
    assert len(set(keys)) == 2


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
            source="offline_optimizer",
            pipeline="playbook",
            entity_type="profile",
        )
    assert hook.call_count == 2
    assert all(call.kwargs["platform_llm"] is True for call in hook.call_args_list)
    assert {call.kwargs["event_key"] for call in hook.call_args_list} == {
        "learn:profile:r1",
        "learn:profile:r2",
    }


def test_emit_records_noop_for_empty_ids():
    with patch(HOOK) as hook:
        emit_learnings_generated_records(
            org_id="org1",
            configurator=_configurator(),
            learning_ids=[],
            source="offline_optimizer",
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
            source="offline_optimizer",
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
