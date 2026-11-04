"""Selected recovery invariants exercised through real SQLite and the worker."""

import json
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
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.extraction.agent_run_records import (
    UNPERSISTED_SERVICE_CONFIG_FIELDS,
    service_config_snapshot,
    source_interaction_digest,
)
from reflexio.server.services.extraction.recovery import (
    RecoveryRefusedError,
    _require_preserved_output_fields,
    inspect_run,
    payload_fingerprint,
    selection_from_report,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredReferencedExtractedPlaybookList,
)
from reflexio.server.services.profile.service import ProfileGenerationServiceConfig
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
    # These tests own a short-lived connection; scheduler execution has its own
    # gates and must not outlive this fixture or start additional inference.
    monkeypatch.setattr(
        "reflexio.server.services.playbook.service.PlaybookGenerationService._dispatch_playbook_schedulers",
        lambda *_args: None,
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
        pending_tool_call_config=PendingToolCallConfig(pending_ttl_seconds=0),
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


def seed(
    ctx,
    identifier="selected",
    service_snapshot=None,
    *,
    legacy_output=False,
    kind="profile",
    payload=None,
):
    config = (
        ctx.configurator.get_config().profile_extractor_config
        if kind == "profile"
        else ctx.configurator.get_config().user_playbook_extractor_config
    )
    schema_name = (
        "StructuredProfilesOutput" if kind == "profile" else "StructuredPlaybookList"
    )
    run = AgentRunRecord(
        id=identifier,
        binding=AgentBinding(
            org_id="org",
            extractor_kind=kind,
            user_id="user",
            request_id="request",
            agent_version="v1",
            source="api",
            source_interaction_ids=[1],
            extractor_config_hash=build_scope_hash(config.model_dump(mode="json")),
        ),
        status=AgentRunStatus.AGENT_COMPLETED
        if legacy_output
        else AgentRunStatus.RUNNING,
        generation_request_snapshot={
            "output_schema_name": schema_name,
            "source_interaction_digests": {
                "1": source_interaction_digest(
                    ctx.storage.get_interactions_by_ids([1])[0]
                )
            },
        },
        committed_output={"profiles": []} if legacy_output else None,
        service_config_snapshot=service_snapshot,
    )
    if legacy_output:
        # Literal historical state: no writer UUID or fabricated input proof.
        ctx.storage.create_agent_run(run)
    else:
        from reflexio.server.services.playbook.playbook_service_utils import (
            StructuredPlaybookList,
        )
        from reflexio.server.services.profile.profile_generation_service_utils import (
            StructuredProfilesOutput,
        )
        from tests.server.services.extraction._publication_fixtures import (
            write_completed_output,
        )

        shape = (
            StructuredProfilesOutput if kind == "profile" else StructuredPlaybookList
        )
        write_completed_output(
            ctx.storage,
            run,
            shape.model_validate(
                payload
                if payload is not None
                else {"profiles" if kind == "profile" else "playbooks": []}
            ),
        )
    old = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    ctx.storage.conn.execute(
        "UPDATE _agent_runs SET updated_at=?, claimed_by=NULL, claimed_at=NULL WHERE id=?",
        (old, identifier),
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
    assert not (
        context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds
        > 0
    )
    with pytest.raises(RecoveryRefusedError):
        worker.recover_selected(selection, operation_id="again")


# A row written before the snapshot stopped copying content (legacy) and one
# written after (trimmed). Stored rows are never rewritten, so each row's
# payload fingerprint is a hash of what that row already stores.
_LEGACY_SERVICE_SNAPSHOT = {
    "user_id": "user",
    "request_id": "request",
    "existing_data": [{"content": "OLD PROFILE"}],
    "window_interactions": [{"interactions": [{"content": "PRIVATE SOURCE"}]}],
    "frozen_extractor_config": {"extraction_definition_prompt": "test"},
}


@pytest.mark.parametrize("shape", ["legacy", "trimmed"])
def test_recovery_accepts_legacy_and_trimmed_service_snapshots(context, shape):
    snapshot = (
        _LEGACY_SERVICE_SNAPSHOT
        if shape == "legacy"
        else service_config_snapshot(
            ProfileGenerationServiceConfig(
                user_id="user",
                request_id="request",
                existing_data=[{"content": "OLD PROFILE"}],
                frozen_extractor_config={"extraction_definition_prompt": "test"},
            )
        )
    )
    run = seed(context, service_snapshot=snapshot)
    assert run.service_config_snapshot == snapshot
    if shape == "trimmed":
        assert UNPERSISTED_SERVICE_CONFIG_FIELDS.isdisjoint(snapshot)
    selection = selection_from_report(inspect_run(context, run))
    # Re-reading the untouched row reproduces the preview's fingerprint.
    reread = context.storage.get_agent_run(run.id)
    assert payload_fingerprint(reread) == selection.payload_fingerprint
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    result = worker.recover_selected(selection, operation_id="op")
    assert result.status == AgentRunStatus.FINALIZED
    # Recovery never rewrites the stored snapshot of either shape.
    assert context.storage.get_agent_run(run.id).service_config_snapshot == snapshot


def test_recovery_preview_binds_the_stored_snapshot_shape(context):
    """A preview hashes the row's stored snapshot, so a changed shape refuses."""
    run = seed(context, service_snapshot=_LEGACY_SERVICE_SNAPSHOT)
    selection = selection_from_report(inspect_run(context, run))
    trimmed = {
        key: value
        for key, value in _LEGACY_SERVICE_SNAPSHOT.items()
        if key not in UNPERSISTED_SERVICE_CONFIG_FIELDS
    }
    context.storage.conn.execute(
        "UPDATE _agent_runs SET service_config_snapshot=? WHERE id=?",
        (json.dumps(trimmed), run.id),
    )
    context.storage.conn.commit()
    with pytest.raises(RecoveryRefusedError, match="preview_changed"):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="op")


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
            {"claimed_by": "other"},
            "run_revision_changed",
        ),
        ({"finalization_attempts": 3}, "retry_limit_reached"),
    ],
)
def test_blockers_are_explicit(context, change, reason):
    # A supplied logical witness must match the actual scoped row, even when
    # its timestamp was not advanced by a superseding legacy writer.
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
    from reflexio.server.services.durable_learning.user_lease import (
        user_extraction_lease,
    )

    run = seed(context)
    args = {
        "expected": run,
        "operation_id": "op",
    }
    with user_extraction_lease(context.storage, "user", max_wait_seconds=0):
        claim = context.storage.claim_agent_run_for_recovery(**args, worker_id="first")
        assert claim is not None
        assert (
            context.storage.claim_agent_run_for_recovery(**args, worker_id="second")
            is None
        )
        result = context.storage.update_agent_run_status(
            run.id,
            AgentRunStatus.FINALIZED,
            expected_claimed_by="second",
            expected_claimed_at=claim.claimed_at,
        )
        assert result is None
        assert context.storage.get_agent_run(run.id).claimed_by == claim.claimed_by
        assert claim.claimed_by.startswith("first:")


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
        "reflexio.server.services.profile.service.ProfileGenerationService._persist_write_plan",
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
    assert context.storage._resume_candidate_page("finalization")[1] is None


def test_playbook_validator_cannot_discard_saved_candidates():
    payload = {"playbooks": [{"evidence_ref": "T1"}]}
    parsed = StructuredReferencedExtractedPlaybookList.model_validate(payload)
    assert parsed.playbooks == []  # Ordinary generation tolerates this shape.
    with pytest.raises(ValueError, match="candidates were discarded"):
        _require_preserved_output_fields(payload, parsed)


def test_interrupted_selected_run_keeps_source_checks_in_ordinary_retry(context):
    context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds = (
        86_400
    )
    run = seed(context)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    selection = selection_from_report(inspect_run(context, run))
    # Interrupt after real admission/adoption, so cleanup must remain inside
    # the actual owner instead of fabricating an unowned post-release write.
    with (
        patch.object(
            worker, "_retry_finalization", side_effect=RuntimeError("interrupted")
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
    DurableLearningWorker(lambda _: context).drain_org("org", 1, 300)
    result = context.storage.get_agent_run(run.id)
    assert result is not None
    assert result.status == AgentRunStatus.FAILED
    assert result.last_error == "selected_recovery_refused"
    assert result.committed_output == run.committed_output
    assert not context.storage.get_agent_run_recovery_facts(run.id)["receipt_exists"]
    assert DurableLearningWorker(lambda _: context).drain_org("org", 1, 300) == 0
    assert context.storage.get_agent_run(run.id) == result


def _publish_with_pending_delivery(context, *, kind="profile", selected=True):
    from reflexio.models.config_schema import PlaybookConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.playbook.components.reviewer import (
        PlaybookCandidateReviewer,
    )
    from reflexio.server.usage_metrics import UsageEventDeliveryStatus

    context.configurator.get_config().user_playbook_extractor_config = PlaybookConfig(
        extraction_definition_prompt="test"
    )
    context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds = (
        86_400
    )
    context.prompt_manager = PromptManager()
    entry = (
        {"content": "Synthetic result", "time_to_live": "infinity"}
        if kind == "profile"
        else {
            "content": "Require approval",
            "trigger": "before deployment",
            "rationale": "Agreed procedure",
        }
    )
    run = seed(
        context,
        kind=kind,
        payload={"profiles" if kind == "profile" else "playbooks": [entry]},
    )
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    attempts = []

    def fail_delivery(event):
        if event.event_name == "learnings_generated":
            attempts.append(event)
            return UsageEventDeliveryStatus.FAILED
        return UsageEventDeliveryStatus.APPENDED

    configure_usage_event_recorder(fail_delivery)
    with (
        patch(
            "reflexio.server.services.profile.components.consolidator.ProfileConsolidator.deduplicate",
            side_effect=lambda items, *_a, **_kw: (items, [], []),
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator.deduplicate",
            side_effect=lambda groups, *_a, **_kw: (
                [item for group in groups for item in group],
                [],
                [],
            ),
        ),
        patch.object(
            PlaybookCandidateReviewer,
            "review",
            side_effect=lambda **kw: kw["candidates"],
        ),
    ):
        if selected:
            result = worker.recover_selected(
                selection_from_report(inspect_run(context, run)), operation_id="op"
            )
        else:
            DurableLearningWorker(lambda _: context).drain_org("org", 1, 300)
            result = context.storage.get_agent_run(run.id)
    assert result is not None and result.status == AgentRunStatus.FINALIZED
    assert result.committed_output is not None
    assert result.finalization_attempts == 0 and len(attempts) == 1
    with context.storage._stream_sql() as db:
        receipt = context.storage._get_output_publication_receipt(
            db, run.id, result.committed_output["publication"]["id"]
        )
    assert (
        receipt is not None
        and receipt.delivery_state == "pending"
        and len(receipt.learning_ids) == 1
    )
    return result, receipt, worker


@pytest.mark.parametrize("cached_preparation", [False, True])
def test_profile_recovery_authors_from_bound_source_and_reuses_saved_preparation(
    context, monkeypatch, cached_preparation
):
    import time

    import litellm

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.profile.service import ProfileGenerationService

    quote = "I require separate approval before deploying."
    source = context.storage.get_interactions_by_ids([1])[0]
    context.storage._insert_interaction(source.model_copy(update={"content": quote}))
    context.storage._insert_interaction(
        source.model_copy(
            update={
                "interaction_id": 2,
                "content": "Unrelated source outside this run.",
            }
        )
    )
    context.prompt_manager = PromptManager()
    context.configurator.get_config().pending_tool_call_config.pending_ttl_seconds = (
        86_400
    )
    run = seed(
        context,
        payload={
            "profiles": [
                {"content": quote, "source_span": quote, "time_to_live": "infinity"}
            ]
        },
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o-mini":128000}')
    monkeypatch.setattr(
        "reflexio.server.services.deduplication_utils.resolve_model_name",
        lambda *_a, **_k: "gpt-4o-mini",
    )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=8192, fallback_models=[])
    )
    calls = []

    def author(params, _timeout):
        payload = json.loads(params["messages"][0]["content"].rsplit("\n\n", 1)[1])
        interactions = [
            item
            for group in payload["original_window"]
            for item in group["interactions"]
        ]
        assert [item["interaction_id"] for item in interactions] == [1]
        assert interactions[0]["role"] == "user"
        excerpt = interactions[0]["content"]["excerpts"][0]
        assert excerpt["text"] == quote
        assert payload["EXISTING"] == [] and len(payload["NEW"]) == 1
        output = {
            "snapshot": payload["snapshot"],
            "groups": [
                {
                    "new_ids": [payload["NEW"][0]["id"]],
                    "writes": [
                        {
                            "clauses": [
                                {
                                    "action": "assert",
                                    "source_new_id": payload["NEW"][0]["id"],
                                    "content": quote,
                                    "evidence": [
                                        {
                                            "excerpt_id": excerpt["excerpt_id"],
                                            "stance": "assertion",
                                        }
                                    ],
                                }
                            ]
                        }
                    ],
                    "forget": [],
                }
            ],
        }
        calls.append(output)
        return litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": json.dumps(output)},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", author)
    worker = ExtractionResumeWorker(request_context=context, llm_client=client)
    if cached_preparation:
        with (
            patch.object(
                ProfileGenerationService,
                "_persist_write_plan",
                side_effect=RuntimeError("publication paused after preparation"),
            ),
            pytest.raises(RecoveryRefusedError, match="recovery_not_verified"),
        ):
            worker.recover_selected(
                selection_from_report(inspect_run(context, run)),
                operation_id="first",
            )
        run = context.storage.get_agent_run(run.id)
        with context.storage._stream_sql(read_only=True) as db:
            preparations = db.query(
                "SELECT preparation FROM _agent_output_preparations "
                "WHERE org_id=? AND run_id=? AND project_id=? LIMIT 1",
                (context.org_id, run.id, ""),
            )
        assert len(preparations) == 1 and preparations[0]["preparation"] is not None
        assert context.storage.get_user_profile("user") == []
        monkeypatch.setattr(
            "reflexio.server.services.extraction.resume_worker._request_interaction_models_from_ids",
            MagicMock(
                side_effect=AssertionError("Saved preparation cannot reload source")
            ),
        )
        # Respect the real first-failure backoff; do not reset attempts or due
        # timestamps to make the cached retry eligible.
        deadline = time.monotonic() + 5
        while "retry_not_due" in inspect_run(context, run)["blockers"]:
            assert time.monotonic() < deadline, (
                "Prepared publication retry stayed deferred"
            )
            time.sleep(0.02)
    result = worker.recover_selected(
        selection_from_report(inspect_run(context, run)), operation_id="finish"
    )
    assert result.status == AgentRunStatus.FINALIZED and len(calls) == 1
    assert result.committed_output is not None and run.committed_output is not None
    assert result.committed_output["output"] == run.committed_output["output"]
    rows = context.storage.get_user_profile("user")
    assert len(rows) == 1 and rows[0].content == quote
    assert rows[0].source_interaction_ids == [1]
    assert rows[0].custom_features["rsi_clauses_v14"]
    assert DurableLearningWorker(lambda _: context).drain_org("org", 1, 300) == 0
    assert len(calls) == 1


def test_current_receipt_delivery_survives_source_change_and_attempt_ceiling(context):
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.usage_metrics import UsageEventDeliveryStatus

    run, receipt, worker = _publish_with_pending_delivery(context)
    context.storage.conn.execute(
        "UPDATE _agent_runs SET finalization_attempts=100 WHERE id=?", (run.id,)
    )
    context.storage.conn.commit()
    source = context.storage.get_interactions_by_ids([1])[0]
    context.storage._insert_interaction(
        source.model_copy(update={"content": "changed"})
    )
    before = context.storage.get_agent_run(run.id)
    events = []
    configure_usage_event_recorder(
        lambda event: events.append(event) or UsageEventDeliveryStatus.APPENDED
    )
    with patch.object(
        ExtractionResumeWorker,
        "_items_from_committed_output",
        side_effect=AssertionError("must not reconstruct execution"),
    ):
        _deliver(context.storage, receipt)
    assert context.storage.get_agent_run(run.id) == before
    assert len(events) == 1 and events[0].entity_id == receipt.learning_ids[0]


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize(
    "kind,entity_type", [("profile", "profile"), ("playbook", "user_playbook")]
)
def test_current_receipt_bills_without_erased_sources_or_output_reconstruction(
    context, selected, kind, entity_type
):
    from reflexio.server.services.extraction.publication import _deliver
    from reflexio.server.usage_metrics import UsageEventDeliveryStatus

    result, receipt, worker = _publish_with_pending_delivery(
        context, kind=kind, selected=selected
    )
    context.storage.conn.execute("DELETE FROM interactions WHERE interaction_id=1")
    context.storage.conn.execute("DELETE FROM requests WHERE request_id='request'")
    context.storage.conn.commit()
    events = []
    configure_usage_event_recorder(
        lambda event: events.append(event) or UsageEventDeliveryStatus.APPENDED
    )
    with (
        patch.object(
            context.storage,
            "get_agent_run",
            side_effect=AssertionError("must not read run"),
        ),
        patch.object(
            context.storage,
            "get_interactions_by_ids",
            side_effect=AssertionError("must not read source"),
        ),
        patch.object(
            context.configurator,
            "get_config",
            side_effect=AssertionError("must not read config"),
        ),
        patch.object(
            ExtractionResumeWorker,
            "_items_from_committed_output",
            side_effect=AssertionError("must not convert output"),
        ),
    ):
        _deliver(context.storage, receipt)
    assert context.storage.get_agent_run(result.id) == result
    assert (
        len(events) == 1
        and events[0].event_key == f"learn:{entity_type}:{receipt.learning_ids[0]}"
    )
    assert result.committed_output is not None
    with context.storage._stream_sql() as db:
        accepted = context.storage._get_output_publication_receipt(
            db, result.id, result.committed_output["publication"]["id"]
        )
    assert (
        accepted is not None
        and accepted.delivery_state == "accepted"
        and accepted.financial_facts is None
    )


def test_unpublished_output_cannot_borrow_historical_receipt_after_source_erasure(
    context,
):
    from reflexio.models.api_schema.service_schemas import UserProfile
    from reflexio.server.usage_metrics import UsageEventDeliveryStatus

    run = seed(context)
    profile = UserProfile(
        profile_id="historical-result",
        user_id="user",
        content="Synthetic old result",
        last_modified_timestamp=1_700_000_000,
        generated_from_request_id="request",
        embedding=[0.0] * 512,
    )
    context.storage.add_user_profile("user", [profile], skip_embedding=True)
    before_profiles = context.storage.get_user_profile("user")
    context.storage.save_agent_run_finalization_receipt(
        run_id=run.id, entity_type="profile", learning_ids=[profile.profile_id]
    )
    selection = selection_from_report(
        inspect_run(context, context.storage.get_agent_run(run.id))
    )
    claim = context.storage.claim_agent_run_for_recovery
    claims = []

    def erase_after_claim(**kwargs):
        claimed = claim(**kwargs)
        claims.append(claimed)
        context.storage.conn.execute("DELETE FROM interactions WHERE interaction_id=1")
        context.storage.conn.commit()
        return claimed

    events = []
    configure_usage_event_recorder(
        lambda event: events.append(event) or UsageEventDeliveryStatus.APPENDED
    )
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    with (
        patch.object(
            context.storage,
            "claim_agent_run_for_recovery",
            side_effect=erase_after_claim,
        ),
        pytest.raises(RecoveryRefusedError, match="eligibility_changed"),
    ):
        worker.recover_selected(selection, operation_id="op")
    assert len(claims) == 1 and claims[0] is not None
    assert not [event for event in events if event.event_name == "learnings_generated"]
    assert context.storage.get_user_profile("user") == before_profiles
    assert (
        context.storage.get_agent_run_finalization_receipt(
            run_id=run.id, entity_type="profile"
        )
        is None
    )
    historical = context.storage.conn.execute(
        "SELECT entity_type,learning_ids FROM _agent_run_finalization_receipts WHERE run_id=?",
        (run.id,),
    ).fetchone()
    assert historical["entity_type"] == "profile" and json.loads(
        historical["learning_ids"]
    ) == [profile.profile_id]
    with context.storage._stream_sql() as db:
        assert (
            context.storage._get_output_publication_receipt(
                db, run.id, run.committed_output["publication"]["id"]
            )
            is None
        )


@pytest.mark.parametrize(
    "ttl,age_seconds,held",
    [(600, 590, False), (600, 610, True), (1800, 610, False), (60, 70, True)],
)
def test_selected_recovery_follows_actual_owner_across_legacy_timing_values(
    context, monkeypatch, ttl, age_seconds, held
):
    monkeypatch.setenv("REFLEXIO_RESUME_CLAIM_TTL_SECONDS", str(ttl))
    run = seed(context)
    now = datetime.now(UTC)
    old = (now - timedelta(seconds=age_seconds)).isoformat()
    context.storage.conn.execute(
        "UPDATE _agent_runs SET updated_at=? WHERE id=?", (old, run.id)
    )
    context.storage.conn.commit()
    token = (
        context.storage.claim_user_extraction("user", "actual-owner", 300)
        if held
        else None
    )
    try:
        current = context.storage.get_agent_run(run.id)
        assert current is not None
        report = inspect_run(context, current, now=now)
        assert ("live_user_work_owner" in report["blockers"]) is held
        assert report["eligible"] is not held
        if held:
            assert token is not None
            context.storage.fence_user_extraction("user", token)
    finally:
        if token is not None:
            context.storage.release_user_extraction("user", token)


@pytest.mark.parametrize("retired_context", [None, "Historical playbook instructions"])
def test_selected_recovery_keeps_historical_playbook_context_inert(
    context, retired_context
):
    import json

    from reflexio.models.config_schema import UserPlaybookExtractorConfig

    config = UserPlaybookExtractorConfig(extraction_definition_prompt="test")
    context.configurator.get_config().user_playbook_extractor_config = config
    stored = seed(context, kind="playbook")
    frozen = {**config.model_dump(mode="json"), "context_prompt": retired_context}
    snapshot = {
        **stored.generation_request_snapshot,
        "extractor_config": frozen,
        "output_schema_name": "StructuredPlaybookList",
    }
    context.storage.conn.execute(
        "UPDATE _agent_runs SET extractor_kind='playbook',extractor_config_hash=?,"
        "generation_request_snapshot=? WHERE id=?",
        (build_scope_hash(frozen), json.dumps(snapshot), stored.id),
    )
    context.storage.conn.commit()
    run = context.storage.get_agent_run(stored.id)
    assert "configuration_changed" not in inspect_run(context, run)["blockers"]
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    selection = selection_from_report(inspect_run(context, run))
    result = worker.recover_selected(selection, operation_id="historical-context")
    assert result.status == AgentRunStatus.FINALIZED
    assert result.generation_request_snapshot == snapshot
    assert result.binding.extractor_config_hash == build_scope_hash(frozen)


@pytest.mark.parametrize(
    "change", ["prompt", "missing_snapshot", "tampered_snapshot", "profile"]
)
def test_retired_context_does_not_relax_recovery_policy_witnesses(context, change):
    from reflexio.models.config_schema import UserPlaybookExtractorConfig

    stored = seed(context)
    if change == "profile":
        config = context.configurator.get_config().profile_extractor_config
        kind = "profile"
    else:
        config = UserPlaybookExtractorConfig(extraction_definition_prompt="test")
        context.configurator.get_config().user_playbook_extractor_config = config
        kind = "playbook"
    frozen = {**config.model_dump(mode="json"), "context_prompt": "Historical context"}
    snapshot = {**stored.generation_request_snapshot, "extractor_config": frozen.copy()}
    original_hash = build_scope_hash(frozen)
    if change == "prompt":
        config.extraction_definition_prompt = "Changed live policy"
    elif change == "missing_snapshot":
        snapshot.pop("extractor_config")
    elif change == "tampered_snapshot":
        snapshot["extractor_config"]["context_prompt"] = (
            "Not witnessed by original hash"
        )
    run = replace(
        stored,
        binding=replace(
            stored.binding, extractor_kind=kind, extractor_config_hash=original_hash
        ),
        generation_request_snapshot=snapshot,
    )
    assert "configuration_changed" in inspect_run(context, run)["blockers"]


@pytest.mark.parametrize("busy", ["lease", "capacity"])
def test_busy_recovery_does_not_mutate_the_selected_run(context, monkeypatch, busy):
    from reflexio.server.services.durable_learning import user_lease

    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    held = None
    if busy == "lease":
        held = context.storage.claim_user_extraction("user", "live_customer", 300)
        assert held is not None
    else:
        monkeypatch.setattr(user_lease, "_reserve", lambda: False)
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    try:
        with pytest.raises(RecoveryRefusedError):
            worker.recover_selected(selection, operation_id="must_not_be_recorded")
        assert context.storage.get_agent_run(run.id) == run
        if held is not None:
            context.storage.fence_user_extraction("user", held)
    finally:
        if held is not None:
            context.storage.release_user_extraction("user", held)


def test_recovery_claim_and_owner_adoption_are_one_transaction(context, monkeypatch):
    from reflexio.server.services.storage.storage_base import LeaseLostError

    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    monkeypatch.setattr(
        context.storage, "_adopt_owned_user_work_run", lambda *_a, **_k: None
    )
    with pytest.raises(LeaseLostError, match="adoption was refused"):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="rolled_back_claim")
    assert context.storage.get_agent_run(run.id) == run


@pytest.mark.parametrize(
    ("change", "reason", "blockers"),
    [
        ("run_revision", "preview_changed", []),
        ("source", "eligibility_changed", ["source_changed"]),
        ("configuration", "eligibility_changed", ["configuration_changed"]),
    ],
)
def test_changes_during_admission_refuse_before_claim(
    context, monkeypatch, change, reason, blockers
):
    from contextlib import contextmanager

    from reflexio.server.services.durable_learning import user_lease
    from reflexio.server.services.extraction import recovery

    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    admission = user_lease.user_extraction_lease
    observed = []
    reports = []

    def observe_inspection(*args, **kwargs):
        report = inspect_run(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(recovery, "inspect_run", observe_inspection)

    @contextmanager
    def changed_before_admission_returns(*args, **kwargs):
        with admission(*args, **kwargs) as owner:
            if change == "run_revision":
                context.storage.conn.execute(
                    "UPDATE _agent_runs SET updated_at=? WHERE id=?",
                    (datetime.now(UTC).isoformat(), run.id),
                )
                context.storage.conn.commit()
            elif change == "source":
                source = context.storage.get_interactions_by_ids([1])[0]
                context.storage._insert_interaction(
                    source.model_copy(update={"content": "changed after preview"})
                )
            else:
                context.configurator.get_config.return_value.profile_extractor_config = ProfileExtractorConfig(
                    extraction_definition_prompt="changed after preview"
                )
            observed.append(context.storage.get_agent_run(run.id))
            yield owner

    monkeypatch.setattr(
        user_lease, "user_extraction_lease", changed_before_admission_returns
    )
    with pytest.raises(RecoveryRefusedError, match=f"^{reason}$"):
        ExtractionResumeWorker(
            request_context=context, llm_client=MagicMock()
        ).recover_selected(selection, operation_id="must_not_be_recorded")
    assert len(observed) == 1
    assert len(reports) == 2
    assert reports[0]["blockers"] == []
    assert reports[1]["blockers"] == blockers
    assert context.storage.get_agent_run(run.id) == observed[0]
    assert observed[0].recovery_operation_id is None


def test_selected_claim_has_one_actual_admission_and_commits_before_effects(
    context, monkeypatch
):
    from reflexio.server.services.durable_learning import user_lease

    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    counts = {"reserve": 0, "release": 0, "claim": 0, "lease_release": 0}
    for label, target, name in (
        ("reserve", user_lease, "_reserve"),
        ("release", user_lease, "_release"),
        ("lease_release", context.storage, "release_user_extraction"),
    ):
        original = getattr(target, name)

        def track(*args, _original=original, _label=label, **kwargs):
            counts[_label] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(target, name, track)
    claim = context.storage.claim_agent_run_for_recovery

    def claim_under_actual_owner(**kwargs):
        owner = user_lease._require_user_work_owner(context.storage)
        assert owner.user_id == "user"
        owner.fence(context.storage)
        counts["claim"] += 1
        return claim(**kwargs)

    monkeypatch.setattr(
        context.storage, "claim_agent_run_for_recovery", claim_under_actual_owner
    )
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    execute = worker._execute_owned_claim

    def effects_after_commit(claimed, **kwargs):
        assert not context.storage.conn.in_transaction
        owner = user_lease._require_user_work_owner(context.storage)
        parent = context.storage.conn.execute(
            "SELECT user_work_token FROM _agent_runs WHERE id=?", (claimed.id,)
        ).fetchone()[0]
        assert parent == owner.token
        return execute(claimed, **kwargs)

    monkeypatch.setattr(worker, "_execute_owned_claim", effects_after_commit)
    assert (
        worker.recover_selected(selection, operation_id="one_admission").status
        == AgentRunStatus.FINALIZED
    )
    assert counts == {"reserve": 1, "release": 1, "claim": 1, "lease_release": 1}


def test_interruption_at_owned_entry_uses_the_already_adopted_claim(
    context, monkeypatch
):
    run = seed(context)
    selection = selection_from_report(inspect_run(context, run))
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())

    def interrupted_before_entry(*_args, **_kwargs):
        raise RuntimeError("controlled entry interruption")

    monkeypatch.setattr(worker, "_execute_owned_claim", interrupted_before_entry)
    with pytest.raises(RuntimeError, match="controlled entry interruption"):
        worker.recover_selected(selection, operation_id="interrupted_entry")
    after = context.storage.get_agent_run(run.id)
    assert after.status == AgentRunStatus.FINALIZATION_FAILED
    assert after.last_error == "selected_recovery_interrupted"
    assert after.committed_output == run.committed_output
    assert after.recovery_operation_id == "interrupted_entry"


@pytest.mark.parametrize("historical_ids", [None, [], ["historical-result"]])
def test_unqualified_historical_publication_refuses_before_claim(
    context, historical_ids
):
    run = seed(context, legacy_output=True)
    if historical_ids is not None:
        context.storage.save_agent_run_finalization_receipt(
            run_id=run.id, entity_type="profile", learning_ids=historical_ids
        )
    before = context.storage.get_agent_run(run.id)
    report = inspect_run(context, before)
    assert "publication_input_proof_unavailable" in report["blockers"]
    worker = ExtractionResumeWorker(request_context=context, llm_client=MagicMock())
    from reflexio.server.services.extraction.recovery import RecoverySelection

    with pytest.raises(RecoveryRefusedError, match="preview_has_blockers"):
        selection_from_report(report)
    # Even a manually constructed/stale manifest must fail apply before claim.
    selection = RecoverySelection.model_validate(
        {key: report[key] for key in RecoverySelection.model_fields}
    )
    with (
        patch.object(
            context.storage,
            "claim_agent_run_for_recovery",
            side_effect=AssertionError("must refuse before claim"),
        ),
        pytest.raises(RecoveryRefusedError, match="eligibility_changed"),
    ):
        worker.recover_selected(selection, operation_id="op")
    assert context.storage.get_agent_run(run.id) == before
    assert (
        context.storage.get_agent_run_finalization_receipt(
            run_id=run.id, entity_type="profile"
        )
        == historical_ids
    )
    assert "PRIVATE SOURCE" not in str(report)


@pytest.mark.parametrize(
    "damage",
    ["id", "dependency_inputs", "answer_inputs", "source_inputs", "subset", "digest"],
)
def test_current_output_with_incomplete_publication_proof_is_not_claimable(
    context, damage
):
    run = seed(context)
    output = json.loads(json.dumps(run.committed_output))
    if damage in ("id", "dependency_inputs", "answer_inputs", "source_inputs"):
        del output["publication"][damage]
    elif damage == "subset":
        output["publication"]["dependency_inputs"] = {"missing-answer": "a" * 64}
    else:
        output["publication"]["answer_inputs"] = {"invalid-answer": "x" * 64}
    context.storage.conn.execute(
        "UPDATE _agent_runs SET committed_output=? WHERE id=?",
        (json.dumps(output), run.id),
    )
    context.storage.conn.commit()
    before = context.storage.get_agent_run(run.id)
    report = inspect_run(context, before)
    assert "publication_input_proof_unavailable" in report["blockers"]
    with pytest.raises(RecoveryRefusedError, match="preview_has_blockers"):
        selection_from_report(report)
    assert context.storage.get_agent_run(run.id) == before
