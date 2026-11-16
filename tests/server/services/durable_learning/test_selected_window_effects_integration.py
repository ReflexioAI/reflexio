"""SEALED completed-window outbox delivery uses no generation authority."""

import json
from unittest.mock import MagicMock

import pytest

from reflexio.server.services.durable_learning.selected_learning import (
    preview_learning_work,
)
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
)
from reflexio.server.services.tagging.tagging_scheduler import (
    schedule_tagging as real_schedule_tagging,
)
from reflexio.test_support.selected_window_effects import commit_window, grant, invoke
from tests.server.services.extraction.test_selected_recovery_integration import (
    context as context,
)


@pytest.fixture(autouse=True)
def deny_provider_and_new_work(monkeypatch):
    def deny(*_args, **_kwargs):
        raise AssertionError("Effects-only execution cannot generate")

    for name in ("completion", "acompletion", "embedding", "aembedding"):
        monkeypatch.setattr(f"litellm.{name}", deny)
    monkeypatch.setattr(WindowExecutor, "execute", deny)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging",
        lambda **_kwargs: None,
    )


@pytest.mark.parametrize(
    "skipped,parent,billing",
    [(True, False, False), (False, False, True), (False, True, True)],
)
def test_committed_effects_drain_while_ordinary_claims_remain_sealed(
    context, monkeypatch, skipped, parent, billing
):
    storage = context.storage
    window = commit_window(context, skipped=skipped, parent=parent, billing=billing)
    selection = preview_learning_work(context, window_id=window.window_id)
    assert selection.eligible and selection.effects_only, selection.blockers
    admitted = grant(context, selection)
    with pytest.raises(LearningTransitionBlockedError):
        storage.claim_extraction("ordinary", 60)
    with pytest.raises(LearningTransitionBlockedError):
        storage.claim_user_extraction("user", "ordinary", 60)
    prepare = MagicMock(side_effect=AssertionError("Effects cannot prepare fresh work"))
    monkeypatch.setattr(storage, "prepare_extraction", prepare)
    assert any(
        "effects_debt" in row.reasons
        for row in storage.transition_inventory_page("windows").items
    )
    result = invoke(context, selection, admitted)
    assert result.status == "completed", result.to_dict()
    assert storage.pending_extraction_effects() == []
    assert invoke(context, selection, admitted).status == "held"
    if parent:
        assert (
            storage.get_agent_run("window:" + window.window_id).status
            == AgentRunStatus.FINALIZED
        )
    prepare.assert_not_called()


def test_erased_raw_source_is_not_restored_for_committed_effects_ack(context):
    storage = context.storage
    window = commit_window(context, billing=True)
    storage.conn.execute("DELETE FROM interactions")
    storage.conn.execute("DELETE FROM requests")
    storage.conn.commit()
    selection = preview_learning_work(context, window_id=window.window_id)
    assert selection.eligible and selection.effects_only, selection.blockers
    admitted = grant(context, selection)
    assert invoke(context, selection, admitted).status == "completed"
    assert storage.pending_extraction_effects() == []
    assert storage.conn.execute("SELECT count(*) FROM interactions").fetchone()[0] == 0


def test_changed_effects_receipt_or_source_identity_cannot_use_an_old_grant(context):
    storage = context.storage
    window = commit_window(context)
    selection = preview_learning_work(context, window_id=window.window_id)
    admitted = grant(context, selection)
    storage.conn.execute("UPDATE interactions SET request_id='changed'")
    storage.conn.commit()
    assert invoke(context, selection, admitted).status == "held"
    assert len(storage.pending_extraction_effects()) == 1


def test_unsupported_native_effects_codec_remains_held(context):
    window = commit_window(context)
    context.storage.conn.execute(
        "UPDATE extraction_windows SET effects=?",
        (json.dumps({"version": 99, "skipped": True}),),
    )
    context.storage.conn.commit()
    selection = preview_learning_work(context, window_id=window.window_id)
    assert not selection.eligible and selection.effects_only


def test_ack_failure_retains_debt_and_retry_deduplicates_finances_and_handoff(
    context, monkeypatch
):
    from reflexio.server.usage_metrics import (
        UsageEventDeliveryStatus,
        configure_usage_event_recorder,
    )

    storage = context.storage
    window = commit_window(context, skipped=False, billing=True)
    selection = preview_learning_work(context, window_id=window.window_id)
    admitted = grant(context, selection)
    events, attempts, dispatched = {}, [], []

    def record(event):
        if event.event_name != "extraction_tokens":
            return UsageEventDeliveryStatus.EXEMPT
        attempts.append(event)
        prior = event.event_key in events
        events[event.event_key] = event
        return (
            UsageEventDeliveryStatus.DUPLICATE
            if prior
            else UsageEventDeliveryStatus.APPENDED
        )

    configure_usage_event_recorder(record)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.tagging_scheduler.schedule_tagging",
        lambda **kwargs: dispatched.append(kwargs),
    )
    ack = storage.ack_extraction_effects
    monkeypatch.setattr(
        storage,
        "ack_extraction_effects",
        MagicMock(side_effect=RuntimeError("synthetic ACK unavailable")),
    )
    assert invoke(context, selection, admitted).status == "held"
    assert (
        len(storage.pending_extraction_effects()) == 0
    )  # retry backoff, not lost debt
    assert tuple(
        storage.conn.execute(
            "SELECT effects_done,derived_claimed FROM extraction_windows"
        ).fetchone()
    ) == (0, 1)
    monkeypatch.setattr(storage, "ack_extraction_effects", ack)
    selection = preview_learning_work(context, window_id=window.window_id)
    state = storage.read_learning_transition()
    admitted = storage.grant_learning_transition(
        "effects",
        expected_authority_nonce=state.authority.authority_nonce,
        expected_authority_generation=state.authority.generation,
        expected_nonce=state.project.nonce,
        expected_revision=state.project.revision,
        selection_sha256=selection.fingerprint(),
    )
    assert invoke(context, selection, admitted).status == "completed"
    assert len(dispatched) == 1
    assert len(attempts) == 2 and len(events) == 1
    event = events["tok:" + window.window_id]
    assert (
        event.billing_input_tokens,
        event.prompt_tokens,
        event.completion_tokens,
        event.created_at,
    ) == (3, 5, 2, 1.0)
    assert tuple(
        storage.conn.execute(
            "SELECT effects_done,derived_claimed FROM extraction_windows"
        ).fetchone()
    ) == (1, 1)


def test_invalid_effects_flag_is_rejected_and_unknown_parent_stays_held(
    context,
):
    from reflexio.server.services.durable_learning.selected_learning import (
        LearningWorkSelection,
    )
    from reflexio.server.services.storage.storage_base.learning_transition import (
        LearningTransitionIntegrityError,
    )

    window = commit_window(context, skipped=False, parent=True)
    selection = preview_learning_work(context, window_id=window.window_id)
    with pytest.raises(LearningTransitionIntegrityError):
        LearningWorkSelection.from_dict({**selection.to_dict(), "effects_only": 1})
    context.storage.conn.execute(
        "UPDATE extraction_windows SET effects=json_set(effects, '$.extraction_run_ids', json('[\"unknown-parent\"]'))"
    )
    context.storage.conn.commit()
    assert not preview_learning_work(context, window_id=window.window_id).eligible


def test_effects_handoff_retained_by_normal_tagging_owner_after_context_reset(
    context, monkeypatch
):
    from reflexio.server.services.durable_learning.selected_learning import (
        current_selected_learning,
    )
    from reflexio.server.services.tagging import tagging_scheduler

    # Restore the actual handoff owner instead of the fixture's scheduler stub.
    scheduled = []
    scheduler = MagicMock()
    scheduler.schedule.side_effect = lambda *args, **kwargs: scheduled.append(
        (args, kwargs)
    )
    monkeypatch.setattr(
        tagging_scheduler.TaggingScheduler,
        "get_instance",
        classmethod(lambda _cls: scheduler),
    )
    monkeypatch.setattr(tagging_scheduler, "RequestContext", lambda **_kwargs: context)
    monkeypatch.setattr(tagging_scheduler, "schedule_tagging", real_schedule_tagging)
    context.configurator.get_config().profile_extractor_config.tagging_definition_prompt = "synthetic"
    window = commit_window(context)
    selection = preview_learning_work(context, window_id=window.window_id)
    admitted = grant(context, selection)
    assert invoke(context, selection, admitted).status == "completed"
    assert current_selected_learning() is None
    [(args, kwargs)] = scheduled
    assert kwargs == {}
    key, callback = args
    tagging_scheduler.TaggingScheduler._run_callback(key, callback)
    assert len(scheduled) == 2
    assert scheduled[1] == ((key, callback), {"delay_seconds": 15})
    assert context.storage.pending_extraction_effects() == []
