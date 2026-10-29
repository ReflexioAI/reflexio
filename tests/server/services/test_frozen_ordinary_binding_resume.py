"""Upgrade retries through the ordinary window service retain admitted state."""

import copy
import json
from dataclasses import replace
from unittest.mock import Mock

import pytest

from reflexio.models.config_schema import (
    DeduplicationConfig,
    UserPlaybookExtractorConfig,
)
from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.extraction import resumable_agent
from reflexio.server.services.extraction.agent_run_records import (
    build_extractor_agent_run_record,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookList,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base import (
    AgentRunStatus,
    build_scope_hash,
)
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def setup_ordinary(pipeline, monkeypatch, kind, state):
    engine, _, http = pipeline
    if kind == "playbook":
        configurator = engine.request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Extract deployment instructions",
                        deduplication_config=DeduplicationConfig(),
                    ),
                }
            )
        )
    assert http.post(
        "/api/publish_interaction", json=durable.payload("ordinary-upgrade", force=True)
    ).json()["success"]
    storage = engine.get_storage()
    user, token = storage.claim_extraction("ordinary-upgrade-worker", 600)
    window = storage.prepare_extraction(user, token)
    policy = copy.deepcopy(window.policy)
    # Reproduce deployed label/strategy and warning-threshold serialization.
    policy["extractor"]["extractor_name"] = None
    if kind == "playbook":
        policy["extractor"]["deduplication_config"].pop("strategy", None)
        policy["extractor"]["deduplication_config"]["max_unified_content_chars"] = 37
    with storage.conn:
        storage.conn.execute(
            "UPDATE extraction_windows SET policy=? WHERE window_id=?",
            (json.dumps(policy), window.window_id),
        )
    window = storage.prepare_extraction(user, token)
    executor = WindowExecutor(engine.request_context, Mock())
    service, request, config = executor.service(window)
    # compute_generation rebuilds service_config here. The frozen policy must
    # survive that rebuild, not merely the executor's first configuration.
    monkeypatch.setattr(service, "_should_run_before_extraction", lambda _: True)
    assert service._prepare_generation_run(request) is not None
    assert service.service_config.frozen_extractor_config == policy["extractor"]
    schema = StructuredProfilesOutput if kind == "profile" else StructuredPlaybookList
    output = schema.model_validate(
        {"profiles" if kind == "profile" else "playbooks": []}
    )
    run = build_extractor_agent_run_record(
        org_id=storage.org_id,
        extractor_kind=kind,
        user_id=user,
        generation_request_id=request.request_id,
        agent_version=getattr(request, "agent_version", None),
        source=request.source,
        request_interaction_data_models=service.service_config.window_interactions,
        extractor_config=config,
        service_config=service.service_config,
        agent_context="original context",
        output_schema_name=schema.__name__,
    )
    snapshot = copy.deepcopy(run.generation_request_snapshot)
    snapshot["extractor_config"] = copy.deepcopy(policy["extractor"])
    run = replace(
        run,
        id=f"window:{window.window_id}",
        binding=replace(
            run.binding, extractor_config_hash=build_scope_hash(policy["extractor"])
        ),
        status=AgentRunStatus(state),
        generation_request_snapshot=snapshot,
        max_steps_remaining=3,
        resume_attempts=2,
        pending_tool_call_ids=["saved-pending-answer"]
        if state == "agent_completed"
        else [],
        committed_output=(
            resumable_agent.encode_committed_output(output, None)
            if state == "agent_completed"
            else None
        ),
    )
    storage.create_agent_run(run)
    loop = Mock(
        return_value=ToolLoopResult(
            ctx=None,
            trace=ToolLoopTrace(finished=True),
            finished_reason="structured_output",
            messages=[],
            structured_output=output,
            max_steps_remaining=2,
        )
    )
    monkeypatch.setattr(resumable_agent, "run_tool_loop", loop)
    monkeypatch.setattr(resumable_agent, "pending_tool_calls_enabled", lambda _: False)

    def execute():
        return resumable_agent.run_resumable_extraction_agent(
            request_context=engine.request_context,
            client=Mock(),
            extractor_kind=kind,
            user_id=user,
            request_id=request.request_id,
            agent_version=getattr(request, "agent_version", None),
            source=request.source,
            request_interaction_data_models=service.service_config.window_interactions,
            extractor_config=config,
            service_config=service.service_config,
            agent_context="original context",
            messages=[{"role": "user", "content": "source"}],
            output_schema=schema,
            log_label="ordinary-upgrade",
        )

    return storage, window, run, service, execute, loop, output, token


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("state", ["failed", "running", "agent_completed"])
def test_ordinary_old_schema_resumes_or_replays_without_rewriting_admission(
    pipeline, monkeypatch, kind, state
):
    storage, window, run, _, execute, loop, output, _ = setup_ordinary(
        pipeline, monkeypatch, kind, state
    )
    original = copy.deepcopy(storage.get_agent_run(run.id))
    compatible = resumable_agent.matches_frozen_extractor_binding
    monkeypatch.setattr(
        resumable_agent, "matches_frozen_extractor_binding", lambda *_: False
    )
    with pytest.raises(RuntimeError, match="Extraction window agent binding changed"):
        execute()
    assert storage.get_agent_run(run.id) == original
    loop.assert_not_called()
    monkeypatch.setattr(resumable_agent, "matches_frozen_extractor_binding", compatible)
    result = execute()
    assert result.output == output
    assert result.run_id == original.id
    if state == "agent_completed":
        assert result.pending_tool_call_ids == original.pending_tool_call_ids
        assert (
            storage.get_agent_run(run.id).pending_tool_call_ids
            == original.pending_tool_call_ids
        )
    assert execute().output == output
    saved = storage.get_agent_run(run.id)
    assert saved.binding == original.binding
    assert saved.generation_request_snapshot == original.generation_request_snapshot
    assert saved.service_config_snapshot == original.service_config_snapshot
    assert saved.agent_context_snapshot == original.agent_context_snapshot
    assert saved.resume_attempts == original.resume_attempts
    assert loop.call_count == (0 if state == "agent_completed" else 1)
    if loop.called:
        # A no-output retry restarts from the seed prompt, so it gets the full
        # step budget rather than the saved run's spent counter (3).
        assert loop.call_args.kwargs["max_steps"] == 8
    else:
        assert saved.committed_output == original.committed_output
    assert (
        storage.extraction_status(window.user_id, "ordinary-upgrade")["status"]
        != "done"
    )
    assert storage.pending_extraction_effects() == []


@pytest.mark.parametrize(
    "tamper", ["no_policy", "policy", "prompt", "search_policy", "identity", "unknown"]
)
def test_ordinary_compatibility_rejects_unproven_or_changed_work(
    pipeline, monkeypatch, tamper
):
    storage, _, run, service, execute, loop, _, _ = setup_ordinary(
        pipeline, monkeypatch, "playbook", "agent_completed"
    )
    original = copy.deepcopy(storage.get_agent_run(run.id))
    if tamper == "no_policy":
        service.service_config.frozen_extractor_config = None
    elif tamper == "policy":
        service.service_config.frozen_extractor_config["context_prompt"] = "changed"
    elif tamper == "prompt":
        service._window_extractor_config.extraction_definition_prompt = (
            "Changed instructions"
        )
    elif tamper == "search_policy":
        service._window_extractor_config.deduplication_config.search_threshold = 0.9
    elif tamper == "identity":
        service.service_config.window_interactions[0].interactions[
            0
        ].interaction_id += 999
    else:
        service.service_config.frozen_extractor_config["unknown_removed_setting"] = True
    with pytest.raises(RuntimeError, match="Extraction window agent binding changed"):
        execute()
    loop.assert_not_called()
    assert storage.get_agent_run(run.id) == original


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_ordinary_saved_output_retry_commits_window_once(
    pipeline, monkeypatch, kind, caplog
):
    storage, window, run, _, _, loop, _, token = setup_ordinary(
        pipeline, monkeypatch, kind, "agent_completed"
    )
    engine, _, _ = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "false")
    original = copy.deepcopy(storage.get_agent_run(run.id))
    executor = WindowExecutor(engine.request_context, Mock())
    with caplog.at_level("INFO", logger="reflexio.ops"):
        executor.execute(window, token)
    saved = storage.get_agent_run(run.id)
    assert saved.binding == original.binding
    assert saved.generation_request_snapshot == original.generation_request_snapshot
    assert saved.committed_output == original.committed_output
    assert saved.pending_tool_call_ids == original.pending_tool_call_ids
    loop.assert_not_called()
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT completed_seq FROM extraction_cursors WHERE user_id=? AND kind=?",
            (window.user_id, kind),
        )[0]
    assert cursor["completed_seq"] == window.end_seq
    effects = storage.pending_extraction_effects()
    assert len(effects) == 1
    assert effects[0][0].window_id == window.window_id
    assert (
        storage.extraction_status(window.user_id, "ordinary-upgrade")["status"]
        == "done"
    )
    assert (
        f"event=extraction_window_committed window_id={window.window_id}" in caplog.text
    )
