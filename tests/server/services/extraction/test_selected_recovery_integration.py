"""Selected recovery invariants exercised through real SQLite and the worker."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.extraction.agent_run_records import (
    source_interaction_digest,
)
from reflexio.server.services.extraction.recovery import (
    RecoveryRefusedError,
    _require_preserved_output_fields,
    inspect_run,
    selection_from_report,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    build_scope_hash,
)
from reflexio.server.usage_metrics import (
    configure_usage_event_recorder,
    exempt_usage_event_recorder,
)


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", lambda *_a, **_k: [0.0] * 512)
    monkeypatch.setattr(
        "reflexio.server.services.extraction.resume_worker.schedule_tagging",
        lambda **_k: None,
    )
    store = SQLiteStorage(org_id="org", db_path=str(tmp_path / "recovery.db"))
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id, ctx.storage, ctx.storage_base_dir = "org", store, None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extraction_definition_prompt="test"
        ),
        pending_tool_call_config=PendingToolCallConfig(enabled=False),
    )
    ctx.configurator.get_agent_context.return_value = "test"
    ctx.prompt_manager = MagicMock()
    store.add_request(
        Request(
            request_id="request",
            user_id="user",
            session_id="session",
            source="api",
            agent_version="v1",
        )
    )
    store._insert_interaction(
        Interaction(
            interaction_id=1,
            user_id="user",
            request_id="request",
            role="user",
            content="PRIVATE SOURCE",
        )
    )
    configure_usage_event_recorder(exempt_usage_event_recorder)
    yield ctx
    configure_usage_event_recorder(None)
    store.conn.close()


def seed(ctx, identifier="selected"):
    config = ctx.configurator.get_config().profile_extractor_config
    run = AgentRunRecord(
        id=identifier,
        binding=AgentBinding(
            org_id="org",
            extractor_kind="profile",
            user_id="user",
            request_id="request",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1],
            extractor_config_hash=build_scope_hash(config.model_dump(mode="json")),
        ),
        status=AgentRunStatus.AGENT_COMPLETED,
        generation_request_snapshot={
            "output_schema_name": "StructuredProfilesOutput",
            "source_interaction_digests": {
                "1": source_interaction_digest(
                    ctx.storage.get_interactions_by_ids([1])[0]
                )
            },
        },
        committed_output={"profiles": []},
    )
    ctx.storage.create_agent_run(run)
    old = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    ctx.storage.conn.execute(
        "UPDATE _agent_runs SET updated_at=? WHERE id=?", (old, identifier)
    )
    ctx.storage.conn.commit()
    return ctx.storage.get_agent_run(identifier)


def test_recovery_selects_only_named_run_with_disabled_resume(context):
    selected, other = seed(context), seed(context, "other")
    selection = selection_from_report(inspect_run(context, selected))
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    result = worker.recover_selected(selection, operation_id="op")
    assert result.status == AgentRunStatus.FINALIZED
    assert result.progress_stage == "completed"
    assert context.storage.get_agent_run_recovery_facts(result.id)["receipt_count"] == 0
    assert context.storage.get_agent_run(other.id) == other
    assert [entry["event"] for entry in result.recovery_history] == [
        "finalizing",
        "finalized",
    ]
    assert not context.configurator.get_config().pending_tool_call_config.enabled
    with pytest.raises(RecoveryRefusedError):
        worker.recover_selected(selection, operation_id="again")


def test_stale_preview_never_claims(context):
    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    context.storage.update_agent_run_status(run.id, AgentRunStatus.FINALIZATION_FAILED)
    with pytest.raises(RecoveryRefusedError, match="preview_changed"):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="op")
    assert context.storage.get_agent_run(run.id).recovery_operation_id is None


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"committed_output": None}, "saved_output_missing"),
        ({"pending_tool_call_ids": ["pending"]}, "pending_dependencies"),
        ({"status": AgentRunStatus.RUNNING}, "not_finalization_candidate"),
        (
            {"claimed_by": "other", "claimed_at": datetime.now(UTC)},
            "live_or_unknown_claim",
        ),
        ({"finalization_attempts": 3}, "retry_limit_reached"),
    ],
)
def test_blockers_are_explicit(context, change, reason):
    run = replace(seed(context), **change)
    assert reason in inspect_run(context, run)["blockers"]


def test_source_erasure_and_config_drift_block(context):
    run = seed(context)
    context.storage.conn.execute("DELETE FROM interactions WHERE interaction_id=1")
    context.storage.conn.commit()
    context.configurator.get_config().profile_extractor_config.extraction_definition_prompt = "changed"
    report = inspect_run(context, run)
    assert set(report["blockers"]) >= {
        "source_missing_or_scope_mismatch",
        "configuration_changed",
    }
    assert "PRIVATE SOURCE" not in str(report)


def test_claim_cas_and_owner_fencing(context):
    run = seed(context)
    args = {
        "run_id": run.id,
        "org_id": "org",
        "expected_updated_at": run.updated_at,
        "operation_id": "op",
    }
    claim = context.storage.claim_agent_run_for_recovery(**args, worker_id="first")
    assert claim is not None
    assert (
        context.storage.claim_agent_run_for_recovery(**args, worker_id="second") is None
    )
    result = context.storage.update_agent_run_status(
        run.id,
        AgentRunStatus.FINALIZED,
        expected_claimed_by="second",
        expected_claimed_at=claim.claimed_at,
    )
    assert result is None
    assert context.storage.get_agent_run(run.id).claimed_by == "first"


def test_receipt_and_progress_rollback_together(context):
    run = seed(context)
    with pytest.raises(RuntimeError, match="crash"), context.storage.commit_scope():
        context.storage.save_agent_run_finalization_receipt(
            run_id=run.id, entity_type="profile", learning_ids=[]
        )
        assert (
            context.storage.get_agent_run(run.id).progress_stage == "results_committed"
        )
        raise RuntimeError("crash")
    assert not context.storage.get_agent_run_recovery_facts(run.id)["receipt_exists"]
    assert context.storage.get_agent_run(run.id).progress_stage == "output_saved"


def test_crash_after_receipt_commit_can_finalize_without_duplicate(context):
    run = seed(context)
    context.storage.save_agent_run_finalization_receipt(
        run_id=run.id, entity_type="profile", learning_ids=[]
    )
    current = context.storage.get_agent_run(run.id)
    assert current.progress_stage == "results_committed"
    selection = selection_from_report(inspect_run(context, current))
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    with patch(
        "reflexio.server.services.profile.service.ProfileGenerationService._finalize_write_plan_with_outcome",
        side_effect=AssertionError("duplicate commit"),
    ):
        assert (
            worker.recover_selected(selection, operation_id="repair").status
            == AgentRunStatus.FINALIZED
        )


def test_generic_timestamp_is_not_progress(context):
    run = seed(context)
    context.storage.conn.execute(
        "UPDATE _agent_runs SET updated_at=? WHERE id=?",
        (datetime.now(UTC).isoformat(), run.id),
    )
    context.storage.conn.commit()
    assert (
        context.storage.get_agent_run(run.id).last_progress_at == run.last_progress_at
    )


@pytest.mark.parametrize("pinned", [True, False])
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"add": [{"content": "legacy operation"}]},
        {"profiles": None},
        {"profiles": [], "delete": ["legacy operation"]},
        {
            "profiles": [
                {"content": "example", "time_to_live": "infinity", "delete": True}
            ]
        },
    ],
)
def test_ambiguous_profile_output_is_preserved_without_claim(context, pinned, payload):
    stored = seed(context)
    run = replace(
        stored,
        committed_output=payload,
        generation_request_snapshot=stored.generation_request_snapshot
        if pinned
        else {},
    )
    report = inspect_run(context, run)
    assert "invalid_saved_output" in report["blockers"]
    with pytest.raises(RecoveryRefusedError, match="preview_has_blockers"):
        selection_from_report(report)
    assert context.storage.get_agent_run(run.id) == stored


def test_legacy_output_requires_original_source_evidence(context):
    run = replace(seed(context), generation_request_snapshot={})
    report = inspect_run(context, run)
    assert report["blockers"] == ["source_version_unverifiable"]
    assert not report["eligible"]


def test_same_id_source_replacement_invalidates_preview_before_claim(context):
    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    interaction = context.storage.get_interactions_by_ids([1])[0]
    context.storage._insert_interaction(
        interaction.model_copy(update={"content": "REPLACED PRIVATE SOURCE"})
    )
    report = inspect_run(context, run)
    assert report["blockers"] == ["source_changed"]
    assert "PRIVATE" not in str(report)
    with pytest.raises(RecoveryRefusedError):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="op")
    assert context.storage.get_agent_run(run.id).recovery_operation_id is None


def test_source_replacement_after_claim_cannot_enter_ordinary_retry_queue(context):
    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    claim = context.storage.claim_agent_run_for_recovery

    def replace_after_claim(**kwargs):
        claimed = claim(**kwargs)
        source = context.storage.get_interactions_by_ids([1])[0]
        context.storage._insert_interaction(
            source.model_copy(update={"content": "changed"})
        )
        return claimed

    with (
        patch.object(
            context.storage,
            "claim_agent_run_for_recovery",
            side_effect=replace_after_claim,
        ),
        pytest.raises(RecoveryRefusedError),
    ):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="op")
    refused = context.storage.get_agent_run(run.id)
    assert refused.status == AgentRunStatus.FAILED
    assert refused.last_error == "selected_recovery_refused"
    assert refused.committed_output == run.committed_output
    assert not context.storage.get_agent_run_recovery_facts(run.id)["receipt_exists"]
    assert (
        context.storage.claim_finalization_failed_agent_run(
            org_id="org", worker_id="ordinary", claim_ttl_seconds=1
        )
        is None
    )


def test_playbook_validator_cannot_discard_saved_candidates():
    payload = {"playbooks": [{"evidence_ref": "T1"}]}
    parsed = StructuredReferencedExtractedPlaybookList.model_validate(payload)
    assert parsed.playbooks == []  # Ordinary generation tolerates this shape.
    with pytest.raises(ValueError, match="candidates were discarded"):
        _require_preserved_output_fields(payload, parsed)


def test_interrupted_selected_run_keeps_source_checks_in_ordinary_retry(context):
    run = seed(context)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    selection = selection_from_report(inspect_run(context, run))
    with (
        patch.object(
            worker, "_execute_claimed_run", side_effect=RuntimeError("interrupted")
        ),
        pytest.raises(RuntimeError),
    ):
        worker.recover_selected(selection, operation_id="op")
    assert (
        context.storage.get_agent_run(run.id).status
        == AgentRunStatus.FINALIZATION_FAILED
    )
    source = context.storage.get_interactions_by_ids([1])[0]
    context.storage._insert_interaction(
        source.model_copy(update={"content": "changed"})
    )
    result = worker.run_once()
    assert result is not None
    assert result.status == AgentRunStatus.FAILED
    assert result.last_error == "selected_recovery_refused"
    assert result.committed_output == run.committed_output
    assert not context.storage.get_agent_run_recovery_facts(run.id)["receipt_exists"]
    assert worker.run_once() is None


def test_selected_receipt_replay_survives_source_change_and_attempt_ceiling(context):
    run = seed(context)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    selection = selection_from_report(inspect_run(context, run))
    with (
        patch.object(
            worker, "_execute_claimed_run", side_effect=RuntimeError("interrupted")
        ),
        pytest.raises(RuntimeError),
    ):
        worker.recover_selected(selection, operation_id="op")
    context.storage.save_agent_run_finalization_receipt(
        run_id=run.id, entity_type="profile", learning_ids=[]
    )
    context.storage.conn.execute(
        "UPDATE _agent_runs SET finalization_attempts=100 WHERE id=?", (run.id,)
    )
    context.storage.conn.commit()
    source = context.storage.get_interactions_by_ids([1])[0]
    context.storage._insert_interaction(
        source.model_copy(update={"content": "changed"})
    )
    with patch(
        "reflexio.server.services.profile.service.ProfileGenerationService._finalize_write_plan_with_outcome",
        side_effect=AssertionError("duplicate commit"),
    ):
        result = worker.run_once()
    assert result is not None and result.status == AgentRunStatus.FINALIZED
