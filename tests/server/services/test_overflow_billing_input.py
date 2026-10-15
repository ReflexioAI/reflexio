"""Source-input metering counts complete primary fields, independent of dispatch."""

import json
from types import MethodType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction.input_preparation import (
    PreparedChunk,
    PreparedExtractionInput,
    SourceSpan,
    repartition_extraction_input,
)
from reflexio.server.services.extraction.overflow_coordinator import (
    OverflowCoordinator,
)


def _fixture(kind) -> tuple[Any, ...]:
    content = "Full original diagnostic context. " * 2000 + "Late scoped approval."
    shadow = "Do not interpret draft approval as permission to deploy."
    tool = '{"tool_name":"check","output":"Permission denied; nothing deployed."}'
    split = 7301
    spans = (
        SourceSpan(1, "r", "user", "content", 0, split, content[:split], 1),
        SourceSpan(1, "r", "user", "content", split, len(content), content[split:], 1),
        SourceSpan(1, "r", "user", "shadow_content", 0, len(shadow), shadow, 1),
        SourceSpan(1, "r", "user", "tools_used[0]", 0, len(tool), tool, 1),
    )
    prepared = PreparedExtractionInput(
        "immutable-source",
        (
            PreparedChunk(0, spans[:1]),
            PreparedChunk(1, spans[1:3], spans[:1]),
            PreparedChunk(2, spans[3:], spans[2:3]),
        ),
    )
    executor: Any = WindowExecutor.__new__(WindowExecutor)
    executor.context = SimpleNamespace(
        org_id="org",
        configurator=SimpleNamespace(
            get_config=lambda: SimpleNamespace(api_key_config=None)
        ),
    )
    service = SimpleNamespace(
        _extraction_input_text=Mock(return_value="legacy clipped input")
    )
    totals = RunTokenTotals(
        prompt_tokens=20000,
        completion_tokens=400,
        cache_read_input_tokens=100,
        cache_write_input_tokens=50,
    )
    plan = SimpleNamespace(
        write_plan=SimpleNamespace(
            new_profiles=[SimpleNamespace(profile_id=11)],
            new_playbooks=[SimpleNamespace(user_playbook_id=12)],
        ),
        finalization_result=None,
        token_totals=totals,
        prepared=object(),
    )
    window = SimpleNamespace(
        kind=kind, window_id="w", user_id="u", manifest=[{"request_id": "r"}]
    )
    coordinator = SimpleNamespace(require_prepared=Mock(return_value=prepared))
    # Bind the production counter, not a stub: #657 moved the field rejoin out
    # of ``_billing_snapshot`` into ``OverflowCoordinator.billing_input_text``,
    # so this is the code whose count these tests pin.
    coordinator.billing_input_text = MethodType(
        OverflowCoordinator.billing_input_text, coordinator
    )
    return (
        executor,
        window,
        service,
        plan,
        coordinator,
        prepared,
        f"{content}\n{shadow}\n{tool}",
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_overflow_billing_counts_full_original_fields_once_across_repartition(kind):
    executor, window, service, plan, coordinator, prepared, full_source = _fixture(kind)
    snapshot = executor._billing_snapshot(window, service, plan, coordinator)
    assert snapshot["input_tokens"] == count_input_tokens(full_source)
    assert snapshot["input_tokens"] > count_input_tokens("legacy clipped input")
    service._extraction_input_text.assert_not_called()
    assert snapshot["prompt_tokens"] == plan.token_totals.prompt_tokens
    assert snapshot["completion_tokens"] == plan.token_totals.completion_tokens
    assert (
        snapshot["cache_read_input_tokens"] == plan.token_totals.cache_read_input_tokens
    )
    assert (
        snapshot["cache_write_input_tokens"]
        == plan.token_totals.cache_write_input_tokens
    )
    changed = repartition_extraction_input(prepared, chunk_index=1, split_after=1)
    coordinator.require_prepared.return_value = changed
    assert (
        executor._billing_snapshot(window, service, plan, coordinator)["input_tokens"]
        == snapshot["input_tokens"]
    )
    assert plan.token_totals.prompt_tokens == 20000


def test_overflow_billing_rejoins_token_boundaries_before_counting():
    executor, window, service, plan, coordinator, _, _ = _fixture("profile")
    coordinator.require_prepared.return_value = PreparedExtractionInput(
        "source",
        (
            PreparedChunk(0, (SourceSpan(1, "r", "user", "content", 0, 3, "hel"),)),
            PreparedChunk(1, (SourceSpan(1, "r", "user", "content", 3, 5, "lo"),)),
        ),
    )
    snapshot = executor._billing_snapshot(window, service, plan, coordinator)
    assert snapshot["input_tokens"] == count_input_tokens("hello")
    assert snapshot["input_tokens"] != count_input_tokens("hel") + count_input_tokens(
        "lo"
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_saved_billing_snapshot_replays_identical_source_and_provider_totals(
    kind, monkeypatch
):
    executor, window, service, plan, coordinator, _, _ = _fixture(kind)
    snapshot = json.loads(
        json.dumps(executor._billing_snapshot(window, service, plan, coordinator))
    )
    record_tokens = Mock()
    record_learnings = Mock()
    monkeypatch.setattr(
        "reflexio.server.metering_events.record_extraction_tokens", record_tokens
    )
    monkeypatch.setattr(
        "reflexio.server.metering_events.record_learnings_generated_records_strict",
        record_learnings,
    )
    executor._bill(window, snapshot)
    executor._bill(window, snapshot)
    first, second = record_tokens.call_args_list
    assert first == second
    assert first.kwargs["billing_input_tokens"] == snapshot["input_tokens"]
    assert first.kwargs["prompt_tokens"] == 20000
    assert first.kwargs["completion_tokens"] == 400
    assert first.kwargs["cache_read_input_tokens"] == 100
    assert first.kwargs["cache_write_input_tokens"] == 50
    assert first.kwargs["event_key"] == "tok:w"
    assert record_learnings.call_args_list[0] == record_learnings.call_args_list[1]
    service._extraction_input_text.assert_not_called()


def test_legacy_billing_preserves_formatter_and_no_plan_has_no_snapshot():
    executor, window, service, plan, _, _, _ = _fixture("profile")
    assert executor._billing_snapshot(window, service, None) is None
    service._extraction_input_text.assert_not_called()
    snapshot = executor._billing_snapshot(window, service, plan)
    assert snapshot["input_tokens"] == count_input_tokens("legacy clipped input")
    service._extraction_input_text.assert_called_once_with(plan.prepared)
