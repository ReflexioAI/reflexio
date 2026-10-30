"""Saved-output configuration retries retain finalization and publication fences."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.extraction.resumable_agent import ResumableExtractionAgent
from reflexio.server.services.playbook.components.reviewer import (
    PlaybookCandidateReviewer,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.test_support.skip_decorators import skip_in_precommit
from tests.server.services.test_durable_window_pipeline import (
    _saved_legacy_window,
    _scripted_audited_response,
    _seed_adopted_window,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
recovery_pipeline = imported_pipeline


def _prepare_failed_saved_output(pipeline, monkeypatch, *, kind, cached):
    (engine, storage, window, token, executor, _, _, _, _, run) = _saved_legacy_window(
        pipeline, monkeypatch, kind=kind
    )
    prepared = _seed_adopted_window(storage, window, token, run)
    span = prepared.chunks[0].spans[0]
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(PlaybookCandidateReviewer, "is_enabled", lambda _: True)
    stages = []

    def respond(params, hard_timeout):
        prompt = json.dumps(params["messages"], ensure_ascii=False)
        audited = _scripted_audited_response(params)
        if audited is not None:
            stages.append("consolidate")
            result = audited
        elif "Select original evidence needed" in prompt:
            stages.append("protect")
            result = {
                "candidate_evidence": [
                    {
                        "candidate_ref": "C1",
                        "kind": "instruction",
                        "evidence_refs": [span.evidence_ref],
                    }
                ]
            }
        elif "reason_code" in prompt:
            stages.append("review")
            result = {
                "decisions": [
                    {
                        "id": "C1",
                        "decision": "accept",
                        "reason_code": "grounded_useful",
                        "evidence_ids": ["C1-E1"],
                    }
                ]
            }
            if "supporting_evidence" in prompt:
                result["supporting_evidence"] = [
                    {
                        "id": "C1",
                        "excerpts": [{"evidence_id": "C1-E1", "text": span.text}],
                    }
                ]
        elif "unique_ids" in prompt:
            stages.append("dedup")
            result = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        else:
            raise AssertionError("Saved extraction must not redispatch raw extraction")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(result), tool_calls=None, refusal=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                total_tokens=120,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model="minimax/MiniMax-M3",
            _hidden_params={"custom_llm_provider": "minimax"},
        )

    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(respond)
    )
    target = "complete_extraction" if cached else "save_extraction_outcome"
    original = getattr(storage, target)
    monkeypatch.setattr(
        storage,
        target,
        Mock(
            side_effect=InputBudgetConfigurationError(
                "Correctable configuration rejected saved-output finalization"
            )
        ),
    )
    # The fixture represents a prior process that saved an output then exited.
    with storage._stream_sql() as db:
        db.query("UPDATE learning_work SET lease_until=0,due_at=0 WHERE user_id='u'")
    _, worker, http = pipeline
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert http.get(
        "/api/learning_status", params={"request_id": "legacy-saved"}
    ).json() == {
        "status": "failed",
        "reason": "input_budget_configuration_error",
    }
    failed = storage.get_agent_run(run.id)
    assert failed is not None and failed.status == AgentRunStatus.FAILED
    assert failed.committed_output == run.committed_output
    assert storage.count_all_profiles() == 0
    assert storage.get_user_playbooks(user_id=window.user_id) == []
    with storage._stream_sql() as db:
        row = db.query(
            "SELECT outcome,completed FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
        assert bool(row["outcome"]) == cached and not row["completed"]
        db.query("UPDATE learning_work SET due_at=0 WHERE user_id='u'")
        db.query("UPDATE extraction_cursors SET retry_at=0 WHERE user_id='u'")
    monkeypatch.setattr(storage, target, original)
    return engine, storage, window, executor, run, worker, http, stages


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("cached", [False, True])
@skip_in_precommit
def test_corrected_saved_output_retry_publishes_and_becomes_discoverably_finalized(
    recovery_pipeline, monkeypatch, kind, cached
):
    _, storage, window, _, run, worker, http, stages = _prepare_failed_saved_output(
        recovery_pipeline, monkeypatch, kind=kind, cached=cached
    )
    first = storage.get_agent_run(run.id)
    assert first is not None
    attempts = first.generation_request_snapshot["overflow"]["attempts"]
    assert worker.drain_org(storage.org_id, 2, 300) == 2
    assert http.get(
        "/api/learning_status", params={"request_id": "legacy-saved"}
    ).json() == {"status": "done", "reason": "covered"}
    finalized = storage.get_latest_finalized_agent_run_for_request(
        org_id=storage.org_id,
        extractor_kind=kind,
        user_id=window.user_id,
        request_id="legacy-saved",
    )
    assert finalized is not None and finalized.id == run.id
    assert finalized.status == AgentRunStatus.FINALIZED
    assert finalized.committed_output == run.committed_output
    assert (
        finalized.generation_request_snapshot["overflow"]["attempts"][: len(attempts)]
        == attempts
    )
    assert stages.count("protect") == 1
    # Playbook publication always passes through audited consolidation.
    assert ("consolidate" in stages) == (kind == "playbook")
    assert storage.count_all_profiles() == (1 if kind == "profile" else 0)
    assert len(storage.get_user_playbooks(user_id=window.user_id)) == (
        1 if kind == "playbook" else 0
    )
    assert storage.pending_extraction_effects() == []
    assert worker.drain_org(storage.org_id, 2, 300) == 0


@pytest.mark.parametrize("failure", ["lease", "cas", "after_restore"])
@skip_in_precommit
def test_recovered_parent_and_learning_writes_roll_back_on_lost_ownership_or_commit(
    recovery_pipeline, monkeypatch, failure
):
    _, storage, window, executor, run, _, _, _ = _prepare_failed_saved_output(
        recovery_pipeline, monkeypatch, kind="profile", cached=True
    )
    user, token = storage.claim_extraction("corrected-retry", 300)
    assert user == window.user_id
    retry = storage.prepare_extraction(user, token)
    assert retry is not None
    before = storage.get_agent_run(run.id)
    assert before is not None
    if failure == "lease":
        token = "stale-token"
    elif failure == "cas":
        original = storage.update_agent_run_status

        def miss(run_id, status, **kwargs):
            if status == AgentRunStatus.AGENT_COMPLETED:
                # Real SQLite's status-CAS miss returns its unchanged FAILED row.
                return original(
                    run_id, status, expected_statuses=(AgentRunStatus.RUNNING,)
                )
            return original(run_id, status, **kwargs)

        monkeypatch.setattr(storage, "update_agent_run_status", miss)
    else:
        monkeypatch.setattr(
            storage,
            "complete_extraction",
            Mock(
                side_effect=RuntimeError(
                    "commit failed after restoration and learning writes"
                )
            ),
        )
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        LeaseLostError,
        OverflowCheckpointConflictError,
    )

    expected = {
        "lease": LeaseLostError,
        "cas": OverflowCheckpointConflictError,
        "after_restore": RuntimeError,
    }[failure]
    with pytest.raises(expected):
        executor.execute(retry, token)
    after = storage.get_agent_run(run.id)
    assert after is not None and after.status == AgentRunStatus.FAILED
    assert after.committed_output == before.committed_output
    assert after.generation_request_snapshot == before.generation_request_snapshot
    assert storage.count_all_profiles() == 0
    assert storage.extraction_status(window.user_id, "legacy-saved")["status"] != "done"
    assert storage.pending_extraction_effects() == []


@skip_in_precommit
def test_custom_extractor_saved_output_retry_restores_finalization(
    recovery_pipeline, monkeypatch
):
    from reflexio.server.services.profile.components.extractor import ProfileExtractor
    from reflexio.server.services.profile.profile_generation_service_utils import (
        StructuredProfilesOutput,
    )
    from reflexio.server.services.profile.service import ProfileGenerationService

    (_, storage, window, token, _, _, _, _, _, run) = _saved_legacy_window(
        recovery_pipeline, monkeypatch, kind="profile"
    )
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)

    class CustomSavedExtractor(ProfileExtractor):
        def _generate_raw_updates_from_sessions(
            self, request_interaction_data_models, existing_profiles
        ):
            result = ResumableExtractionAgent(
                client=self.client, storage=storage
            ).start(
                run=run,
                messages=[],
                output_schema=StructuredProfilesOutput,
            )
            self._last_resumable_run_id = result.run_id
            self._last_model_provenance = result.model_provenance
            assert isinstance(result.output, StructuredProfilesOutput)
            return result.output.model_dump(mode="json")["profiles"]

    def factory(service, extractor_config, service_config):
        return CustomSavedExtractor(
            request_context=service.request_context,
            llm_client=service.client,
            extractor_config=extractor_config,
            service_config=service_config,
            agent_context=service.configurator.get_agent_context(),
        )

    monkeypatch.setattr(ProfileGenerationService, "_create_extractor", factory)
    # Dedup's deterministic suite transport remains mocked; raw saved extraction
    # above goes through the actual resumable-agent existing-output branch.
    storage.fail_extraction_input_budget(
        window, token, "input_budget_configuration_error"
    )
    with storage._stream_sql() as db:
        db.query("UPDATE extraction_cursors SET retry_at=0 WHERE user_id='u'")
    _, worker, _ = recovery_pipeline
    assert worker.drain_org(storage.org_id, 2, 300) == 2
    finalized = storage.get_latest_finalized_agent_run_for_request(
        org_id=storage.org_id,
        extractor_kind="profile",
        user_id=window.user_id,
        request_id="legacy-saved",
    )
    assert finalized is not None and finalized.status == AgentRunStatus.FINALIZED
    assert finalized.committed_output == run.committed_output
    assert "overflow" not in finalized.generation_request_snapshot
    assert storage.count_all_profiles() == 1


@skip_in_precommit
def test_cached_recovery_preserves_pending_tool_progress(
    recovery_pipeline, monkeypatch
):
    _, storage, window, _, run, worker, _, _ = _prepare_failed_saved_output(
        recovery_pipeline, monkeypatch, kind="profile", cached=True
    )
    advanced = storage.update_agent_run_status(
        run.id,
        AgentRunStatus.FINALIZED_PENDING_TOOL,
        pending_tool_call_ids=["retained-human-question"],
    )
    assert advanced is not None
    assert worker.drain_org(storage.org_id, 2, 300) == 2
    after = storage.get_agent_run(run.id)
    assert after is not None and after.status == AgentRunStatus.FINALIZED_PENDING_TOOL
    assert after.pending_tool_call_ids == ["retained-human-question"]
    assert after.committed_output == run.committed_output
    assert storage.extraction_status(window.user_id, "legacy-saved")["status"] == "done"
    assert storage.count_all_profiles() == 1
