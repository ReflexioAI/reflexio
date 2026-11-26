"""Actual inline/manual retained-output lifecycle for the selected method."""

import json
from unittest.mock import Mock

import litellm
import pytest

from reflexio.models.api_schema.domain.entities import Status
from reflexio.models.config_schema import LLMConfig
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    agent_run_user_work,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookGenerationRequest,
)
from reflexio.server.services.playbook.procedural.inline import retained_plan
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.storage.storage_base import AgentRunStatus
from tests.server.services.playbook.test_procedural_window_integration import setup
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline

pytestmark = pytest.mark.integration


def republish(worker, storage, run):
    with agent_run_user_work(storage, run):
        adopted = adopt_agent_run_for_execution(storage, run)
        if adopted.status == AgentRunStatus.FINALIZATION_FAILED:
            # The scheduler ordinarily makes this transition after claiming.
            # Drive the same exact owned CAS for this finalizer integration test.
            adopted = storage.update_agent_run_status(
                adopted.id,
                AgentRunStatus.FINALIZING,
                expected_statuses=(adopted.status,),
                expected_claimed_by=adopted.claimed_by,
                expected_claimed_at=adopted.claimed_at,
            )
            assert adopted is not None
        worker._active_claim = adopted
        return worker._publish_retained_output(adopted)


def inline_setup(pipeline, monkeypatch, *, retire=False, change_before_commit=None):
    engine, storage, window, token, client, calls, existing = setup(
        pipeline, monkeypatch, retire=retire, change_before_commit=change_before_commit
    )
    storage.release_user_extraction(window.user_id, token)
    config = engine.request_context.configurator.get_config().model_copy(deep=True)
    config.llm_config = LLMConfig(generation_model_name="minimax/MiniMax-M3")
    engine.request_context.configurator.set_config(config)
    monkeypatch.setenv("MINIMAX_API_KEY", "fixture-no-provider")
    service = PlaybookGenerationService(
        client, engine.request_context, skip_aggregation=True
    )
    return engine, storage, service, client, calls, existing


@pytest.mark.parametrize("retire", [False, True])
def test_manual_inline_add_or_retirement_preserves_native_publication(
    pipeline, monkeypatch, retire
):
    engine, storage, service, client, calls, _ = inline_setup(
        pipeline, monkeypatch, retire=retire
    )
    service.run(
        PlaybookGenerationRequest(
            user_id="u",
            agent_version="v",
            request_id="procedural",
            source="test",
            auto_run=False,
            force_extraction=True,
        )
    )
    assert len(calls) == 3
    assert [call["model"] for call in calls] == [
        "minimax/MiniMax-M3",
        "minimax/MiniMax-M3.1-Flash",
        "minimax/MiniMax-M3",
    ]
    with storage._stream_sql(read_only=True) as db:
        saved = db.query(
            "SELECT id FROM _agent_runs WHERE user_id=? AND id NOT LIKE 'window:%' LIMIT 2",
            ("u",),
        )
    assert len(saved) == 1
    run = storage.get_agent_run(saved[0]["id"])
    assert run.committed_output is not None
    plan = retained_plan(run)
    assert plan is not None
    assert plan.skip_aggregation is True
    rows = storage.get_user_playbooks(user_id="u", agent_version="v")
    assert len(rows) == (0 if retire else 1)
    if not retire:
        assert "Require explicit approval" in rows[0].content
    worker = ExtractionResumeWorker(
        request_context=engine.request_context, llm_client=client
    )
    republish(worker, storage, run)
    assert len(calls) == 3
    assert len(storage.get_user_playbooks(user_id="u", agent_version="v")) == (
        0 if retire else 1
    )


def test_initial_inline_publication_failure_reuses_reviewed_output(
    pipeline, monkeypatch
):
    engine, storage, service, client, calls, _ = inline_setup(pipeline, monkeypatch)
    original_save = storage.save_user_playbooks
    monkeypatch.setattr(
        storage,
        "save_user_playbooks",
        Mock(side_effect=RuntimeError("synthetic inline write failure")),
    )
    service.run(
        PlaybookGenerationRequest(
            user_id="u",
            agent_version="v",
            request_id="procedural",
            source="test",
            auto_run=False,
            force_extraction=True,
        )
    )
    assert len(calls) == 3
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    monkeypatch.setattr(storage, "save_user_playbooks", original_save)
    with storage._stream_sql(read_only=True) as db:
        saved = db.query(
            "SELECT id FROM _agent_runs WHERE user_id=? AND id NOT LIKE 'window:%' LIMIT 2",
            ("u",),
        )
    assert len(saved) == 1
    run = storage.get_agent_run(saved[0]["id"])
    assert run.committed_output is not None
    worker = ExtractionResumeWorker(
        request_context=engine.request_context, llm_client=client
    )
    republish(worker, storage, run)
    assert len(calls) == 3
    assert len(storage.get_user_playbooks(user_id="u", agent_version="v")) == 1


def test_inline_noop_keeps_incumbent_and_skips_downstream_models(pipeline, monkeypatch):
    _, storage, service, client, calls, existing = inline_setup(
        pipeline, monkeypatch, retire=True
    )
    original = client._completion_with_hard_timeout

    def noop(params, timeout):
        original(params, timeout)  # Record the actual source-reader dispatch.
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "noop",
                                "type": "function",
                                "function": {
                                    "name": "finish_batch",
                                    "arguments": json.dumps(
                                        {
                                            "status": "complete",
                                            "reason": "No reusable change selected.",
                                        }
                                    ),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", noop)
    service.run(
        PlaybookGenerationRequest(
            user_id="u",
            agent_version="v",
            request_id="procedural",
            source="test",
            auto_run=False,
            force_extraction=True,
        )
    )
    assert len(calls) == 1
    current = storage.get_user_playbooks(user_id="u", agent_version="v")
    assert len(current) == 1
    assert current[0].content == existing[0].content
    assert current[0].user_playbook_id == existing[0].user_playbook_id


def test_inline_pending_and_skip_aggregation_flags_survive_retained_plan(
    pipeline, monkeypatch
):
    engine, storage, _, client, calls, _ = inline_setup(pipeline, monkeypatch)
    service = PlaybookGenerationService(
        client,
        engine.request_context,
        output_pending_status=True,
        skip_aggregation=True,
    )
    schedule = Mock(side_effect=AssertionError("Pending/skip must not aggregate"))
    monkeypatch.setattr(service, "_trigger_playbook_aggregation", schedule)
    service.run(
        PlaybookGenerationRequest(
            user_id="u",
            agent_version="v",
            request_id="procedural",
            source="test",
            auto_run=False,
            force_extraction=True,
        )
    )
    assert len(calls) == 3
    rows = storage.get_user_playbooks(
        user_id="u", agent_version="v", status_filter=[Status.PENDING]
    )
    assert len(rows) == 1
    run = service._last_extraction_run
    assert run is not None
    plan = retained_plan(run)
    assert plan is not None
    assert plan.output_pending_status is True and plan.skip_aggregation is True
    schedule.assert_not_called()


@pytest.mark.parametrize("change", ["history", "source"])
def test_inline_changed_original_inputs_cannot_retire_incumbent(
    pipeline, monkeypatch, change
):
    _, storage, service, _, calls, existing = inline_setup(
        pipeline, monkeypatch, retire=True, change_before_commit=change
    )
    service.run(
        PlaybookGenerationRequest(
            user_id="u",
            agent_version="v",
            request_id="procedural",
            source="test",
            auto_run=False,
            force_extraction=True,
        )
    )
    assert len(calls) == 3
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
            (existing[0].user_playbook_id,),
        )
    assert rows[0]["status"] is None and rows[0]["superseded_by"] is None
