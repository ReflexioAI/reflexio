"""Real HTTP/window/SQLite publication with deterministic provider wire replies."""

import json
from unittest.mock import Mock

import litellm
import pytest
import yaml

from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.input_budget import ExtractionInputIntegrityError
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)
from reflexio.server.services.playbook.procedural import native
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


def setup(pipeline, monkeypatch, *, retire=False, change_before_commit=None):
    engine, _, http = pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-no-provider")
    monkeypatch.setenv("REFLEXIO_LLM_CONTEXT_LIMITS_JSON", '{"gpt-4o-mini":128000}')
    monkeypatch.setattr(
        "reflexio.server.llm._litellm_text_generation.resolve_model_name",
        lambda **_: "gpt-4o-mini",
    )
    configured = engine.request_context.configurator.get_config().model_copy(deep=True)
    configured.profile_extractor_config = None
    configured.user_playbook_extractor_config = UserPlaybookExtractorConfig(
        extraction_definition_prompt="Extract reusable deployment procedures."
    )
    engine.request_context.configurator.set_config(configured)
    storage = engine.get_storage()
    monkeypatch.setattr(storage, "_get_embedding", Mock(return_value=[0.0] * 512))
    existing = []
    if retire:
        from reflexio.models.api_schema.domain.entities import UserPlaybook

        row = UserPlaybook(
            user_id="u",
            agent_version="v",
            request_id="old",
            content="Use the obsolete deploy command.",
            trigger="When deploying.",
        )
        storage.save_user_playbooks([row], skip_embedding=True)
        existing = [row]
    monkeypatch.setattr(
        PlaybookConsolidator, "retrieve_existing_playbooks", lambda *_a, **_k: existing
    )
    request = durable.payload("procedural", count=5, force=True)
    request["agent_version"] = "v"
    for interaction in request["interaction_data_list"]:
        interaction["content"] = (
            "Require explicit approval before deploying. The obsolete deploy command is no longer valid."
        )
    assert http.post("/api/publish_interaction", json=request).json()["success"]
    user, token = storage.claim_extraction("procedural-test", 600)
    window = storage.prepare_extraction(user, token)
    assert window.kind == "playbook"
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o-mini", max_tokens=16384, fallback_models=[])
    )
    calls = []

    def wire(params, _timeout):
        calls.append(params)
        names = {tool["function"]["name"] for tool in params["tools"]}
        body = yaml.safe_load(params["messages"][1]["content"])
        if "record_observation" in names:
            evidence = body["sources"][0]["segments"][0]["excerpts"][0]["id"]
            name, arguments = (
                "record_observation",
                {
                    "topic": "Deployment",
                    "observation": "The deployment policy changed.",
                    "evidence": evidence,
                },
            )
        elif "create_procedure" in names or "source_observations" in body:
            evidence = body["source_observations"][0]["evidence"]
            if retire:
                name, arguments = (
                    "retire_procedure",
                    {"group": "deployment", "target": "P1", "evidence": evidence},
                )
            else:
                name, arguments = (
                    "create_procedure",
                    {
                        "group": "deployment",
                        "name": "deployment-approval",
                        "trigger": "When deploying.",
                        "entry": "Require explicit approval before deploying.",
                        "evidence": evidence,
                    },
                )
        elif "review_result" in names:
            if change_before_commit == "history":
                storage._execute(
                    "UPDATE user_playbooks SET content=? WHERE user_playbook_id=?",
                    ("Changed by another owner", existing[0].user_playbook_id),
                )
            elif change_before_commit == "source":
                storage._execute(
                    "UPDATE interactions SET content=? WHERE interaction_id=?",
                    (
                        "Changed original source",
                        body["source"]["sources"][0]["interaction_id"],
                    ),
                )
            name, arguments = (
                "review_result",
                {
                    "unit": "U0",
                    "decision": "accept",
                    "result_support": "pass",
                    "scope_conditions": "pass",
                    "useful_nonduplicate": "pass",
                    "preservation_and_correction": "pass",
                    "resource_consistency": "pass",
                    "reason": "The source directly supports the complete change.",
                },
            )
        else:
            raise AssertionError(f"Retired model stage invoked: {names}")
        if "return_procedural_batch" in names:
            arguments = {
                "calls": [
                    {"id": "entry", "name": name, "arguments": arguments},
                    {
                        "id": "done",
                        "name": "finish_batch",
                        "arguments": {
                            "status": "complete",
                            "reason": "Complete declaration batch.",
                        },
                    },
                ]
            }
            name = "return_procedural_batch"
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(arguments),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    return engine, storage, window, token, client, calls, existing


@pytest.mark.parametrize("retire", [False, True])
@pytest.mark.parametrize("fail_commit", [False, True])
def test_procedural_native_window_publishes_exact_plan_and_replays_without_models(
    pipeline, monkeypatch, retire, fail_commit
):
    engine, storage, window, token, client, calls, existing = setup(
        pipeline, monkeypatch, retire=retire
    )
    executor = WindowExecutor(engine.request_context, client)
    monkeypatch.setattr(
        executor,
        "_prepare_overflow_context",
        Mock(side_effect=AssertionError("Retired protected-record stage must not run")),
    )
    original_complete = storage.complete_extraction
    if fail_commit:
        monkeypatch.setattr(
            storage,
            "complete_extraction",
            Mock(side_effect=RuntimeError("synthetic interrupted commit")),
        )
        with pytest.raises(RuntimeError, match="interrupted commit"):
            executor.execute(window, token)
        assert len(calls) == 3
        monkeypatch.setattr(storage, "complete_extraction", original_complete)
        # Refresh actual saved outcome, as the real worker does on continuation.
        window = storage.prepare_extraction(window.user_id, token)
    executor.execute(window, token)
    assert len(calls) == 3
    run = storage.get_agent_run(f"window:{window.window_id}")
    checkpoint = run.generation_request_snapshot["overflow"]["checkpoint"]
    assert (
        run.generation_request_snapshot["overflow"]["preparation_version"]
        == native.PREPARATION_VERSION
    )
    assert native.KEY in checkpoint
    assert len(checkpoint[native.KEY]["reader_steps"]) == 1
    assert len(run.generation_request_snapshot["overflow"]["attempts"]) == 3
    rows = storage.get_user_playbooks(user_id="u", agent_version="v")
    if retire:
        assert rows == []
        with storage._stream_sql(read_only=True) as db:
            retired = db.query(
                "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
                (existing[0].user_playbook_id,),
            )
        assert len(retired) == 1
        assert (
            retired[0]["status"] is not None or retired[0]["superseded_by"] is not None
        )
    else:
        assert len(rows) == 1
        assert "Require explicit approval" in rows[0].content
        assert rows[0].source_interaction_ids
        assert rows[0].expanded_terms is None
    assert storage.extraction_status("u", "procedural")["status"] == "done"


@pytest.mark.parametrize("change", ["history", "source"])
def test_changed_original_evidence_refuses_native_publication(
    pipeline, monkeypatch, change
):
    engine, storage, window, token, client, calls, existing = setup(
        pipeline, monkeypatch, retire=True, change_before_commit=change
    )
    from reflexio.server.llm.input_budget import ComputationEvidenceChangedError
    from reflexio.server.services.storage.storage_base._extraction_stream import (
        WindowInputsDeletedError,
    )

    with pytest.raises(
        (
            ComputationEvidenceChangedError,
            WindowInputsDeletedError,
            ExtractionInputIntegrityError,
        )
    ):
        WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            "SELECT status, superseded_by FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
            (existing[0].user_playbook_id,),
        )
    assert rows[0]["status"] is None and rows[0]["superseded_by"] is None
    assert storage.extraction_status("u", "procedural")["status"] != "done"


def test_embedding_retry_restores_all_three_completed_model_stages(
    pipeline, monkeypatch
):
    engine, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
    from reflexio.server.llm.providers.embedding_service_provider import (
        EmbeddingUnavailableError,
    )

    original = storage.precompute_user_playbook_embeddings
    monkeypatch.setattr(
        storage,
        "precompute_user_playbook_embeddings",
        Mock(side_effect=EmbeddingUnavailableError("synthetic unavailable index")),
    )
    with pytest.raises(EmbeddingUnavailableError, match="unavailable index"):
        WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    monkeypatch.setattr(storage, "precompute_user_playbook_embeddings", original)
    WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    assert len(storage.get_user_playbooks(user_id="u", agent_version="v")) == 1


def test_all_empty_automatic_window_keeps_its_real_execution_owner_without_model_calls(
    pipeline, monkeypatch
):
    engine, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
    storage._execute("UPDATE interactions SET content='' WHERE user_id=?", ("u",))
    WindowExecutor(engine.request_context, client).execute(window, token)
    assert calls == []
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run is not None and run.status.value == "finalized"
    assert run.committed_output is not None
    assert run.committed_output["publication"]["origin"] == "initial_window"
    assert run.committed_output["model_provenance"] is None
    assert run.committed_output["trace"]["turns"] == []
    assert set(run.binding.source_interaction_ids) == {1, 2, 3, 4, 5}
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    assert storage.extraction_status("u", "procedural")["status"] == "done"


def cli_setup(pipeline, monkeypatch, tmp_path):
    from reflexio.models.config_schema import LLMConfig
    from reflexio.server.llm.providers.cli_tool_protocol import parse_tool_use
    from reflexio.server.llm.providers.native_cli_provider import (
        _build_model_response_with_tool_call,
    )
    from reflexio.server.services.playbook.procedural.transport import BATCH_TOOL

    engine, storage, window, token, _, _, existing = setup(
        pipeline, monkeypatch, retire=True
    )
    model = "local-cli/codex/fixture-model"
    monkeypatch.setenv("REFLEXIO_NATIVE_CLI_MODEL", model)
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    monkeypatch.delenv("CLAUDE_SMART_CLI_MODEL", raising=False)
    executable = tmp_path / "fixture-codex"
    executable.write_text("#!/bin/sh\nexit 77\n")
    executable.chmod(0o700)
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(executable))
    configured = engine.request_context.configurator.get_config().model_copy(deep=True)
    configured.llm_config = LLMConfig(generation_model_name=model)
    engine.request_context.configurator.set_config(configured)
    client = LiteLLMClient(LiteLLMConfig(model=model, fallback_models=[]))
    calls = []

    def wire(params, _timeout):
        calls.append(params)
        assert params["model"] == model
        assert {item["function"]["name"] for item in params["tools"]} == {BATCH_TOOL}
        body = yaml.safe_load(params["messages"][1]["content"])
        if "result_review_units" in body:
            name, arguments = (
                "review_result",
                {
                    "unit": "U0",
                    "decision": "accept",
                    "result_support": "pass",
                    "scope_conditions": "pass",
                    "useful_nonduplicate": "pass",
                    "preservation_and_correction": "pass",
                    "resource_consistency": "pass",
                    "reason": "The source directly supports retirement.",
                },
            )
        elif "source_observations" in body:
            name, arguments = (
                "retire_procedure",
                {
                    "group": "deployment",
                    "target": "P1",
                    "evidence": body["source_observations"][0]["evidence"],
                },
            )
        else:
            name, arguments = (
                "record_observation",
                {
                    "topic": "Deployment",
                    "observation": "The old command is no longer valid.",
                    "evidence": body["sources"][0]["segments"][0]["excerpts"][0]["id"],
                },
            )
        text = json.dumps(
            {
                "tool": BATCH_TOOL,
                "args": {
                    "calls": [
                        {"id": "member", "name": name, "arguments": arguments},
                        {
                            "id": "done",
                            "name": "finish_batch",
                            "arguments": {
                                "status": "complete",
                                "reason": "Complete declaration batch.",
                            },
                        },
                    ]
                },
            }
        )
        parsed = parse_tool_use(text, {BATCH_TOOL})
        assert parsed is not None
        return _build_model_response_with_tool_call(
            model=model, terminal_text=text, elapsed_seconds=0, tool_use=parsed
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    return engine, storage, window, token, client, calls, existing, wire


@pytest.mark.parametrize("change_at_writer", [None, "history", "source"])
def test_cli_retirement_only_window_keeps_native_owner_and_atomic_retirement(
    pipeline, monkeypatch, tmp_path, change_at_writer
):
    engine, storage, window, token, client, calls, existing, _ = cli_setup(
        pipeline, monkeypatch, tmp_path
    )
    if change_at_writer is not None:
        from reflexio.server.llm.input_budget import (
            ComputationEvidenceChangedError,
            ExtractionInputIntegrityError,
        )
        from reflexio.server.services.playbook.service import PlaybookGenerationService
        from reflexio.server.services.storage.storage_base._extraction_stream import (
            WindowInputsDeletedError,
        )

        persist = PlaybookGenerationService.persist_generation

        def changed_before_persist(service, plan):
            if change_at_writer == "history":
                storage._execute(
                    "UPDATE user_playbooks SET content=? WHERE user_playbook_id=?",
                    ("Changed after compute", existing[0].user_playbook_id),
                )
            else:
                storage._execute(
                    "UPDATE interactions SET content=? WHERE user_id=? AND interaction_id=?",
                    ("Changed after compute", "u", 1),
                )
            return persist(service, plan)

        monkeypatch.setattr(
            PlaybookGenerationService, "persist_generation", changed_before_persist
        )
        with pytest.raises(
            (
                ComputationEvidenceChangedError,
                ExtractionInputIntegrityError,
                WindowInputsDeletedError,
            )
        ):
            WindowExecutor(engine.request_context, client).execute(window, token)
        assert len(calls) == 3
        assert storage.get_user_playbooks(user_id="u", agent_version="v")
        monkeypatch.setattr(PlaybookGenerationService, "persist_generation", persist)
        window = storage.prepare_extraction(window.user_id, token)
    WindowExecutor(engine.request_context, client).execute(window, token)
    assert len(calls) == 3
    run = storage.get_agent_run(f"window:{window.window_id}")
    assert run is not None and run.committed_output is not None
    assert run.status.value == "finalized"
    assert run.generation_request_snapshot["overflow"]["preparation_version"] == 5
    assert (
        run.generation_request_snapshot["overflow"]["checkpoint"][native.KEY]["adapter"]
        == "inline"
    )
    assert run.committed_output["publication"]["origin"] == "initial_window"
    assert storage.get_user_playbooks(user_id="u", agent_version="v") == []
    with storage._stream_sql(read_only=True) as db:
        retired = db.query(
            "SELECT status FROM user_playbooks WHERE user_playbook_id=? LIMIT 1",
            (existing[0].user_playbook_id,),
        )
    assert retired[0]["status"] is not None
    assert storage.extraction_status("u", "procedural")["status"] == "done"


@pytest.mark.parametrize("nonempty", [False, True])
def test_marker_free_legacy_window_replays_its_actual_retained_schema_without_v8(
    pipeline, monkeypatch, nonempty
):
    from dataclasses import replace

    from reflexio.server.llm.tools import ToolLoopResult, ToolLoopTrace
    from reflexio.server.services.durable_learning.user_lease import (
        owned_user_work_context,
    )
    from reflexio.server.services.extraction.agent_run_records import (
        build_extractor_agent_run_record,
    )
    from reflexio.server.services.extraction.history_inputs import HistoricalInputs
    from reflexio.server.services.extraction.resumable_agent import (
        ResumableExtractionAgent,
    )
    from reflexio.server.services.extraction.source_inputs import capture_source_inputs
    from reflexio.server.services.playbook.playbook_service_utils import (
        StructuredPlaybookList,
    )

    engine, storage, window, token, client, calls, _ = setup(pipeline, monkeypatch)
    executor = WindowExecutor(engine.request_context, client)
    service, _, config = executor.service(window)
    groups = service.service_config.window_interactions
    extractor = service._create_extractor(config, service.service_config)
    run = build_extractor_agent_run_record(
        org_id=engine.request_context.org_id,
        extractor_kind="playbook",
        user_id="u",
        generation_request_id="procedural",
        agent_version="v",
        source=service.service_config.source,
        request_interaction_data_models=groups,
        extractor_config=config,
        service_config=service.service_config,
        agent_context=extractor.agent_context,
        output_schema_name=StructuredPlaybookList.__name__,
    )
    run = replace(run, id=f"window:{window.window_id}")
    with owned_user_work_context(storage, "u", token, 300):
        saved = ResumableExtractionAgent(client=client, storage=storage).start(
            run=run,
            publication_origin="initial_window",
            messages=[],
            history_inputs=HistoricalInputs(),
            source_inputs=capture_source_inputs(storage, run, groups),
            output_schema=StructuredPlaybookList,
            workflow=lambda _run: ToolLoopResult(
                ctx=None,
                trace=ToolLoopTrace(finished=True),
                finished_reason="structured_output",
                structured_output=StructuredPlaybookList.model_validate(
                    {
                        "playbooks": [
                            {
                                "content": "Require explicit approval before deploying.",
                                "trigger": "When deploying.",
                            }
                        ]
                        if nonempty
                        else []
                    }
                ),
            ),
        )
        assert saved.output is not None
        assert saved.execution_run is not None
        entries = extractor.extract_playbook_entries(groups)
        assert len(entries) == int(nonempty)
        if nonempty:
            assert entries[0].content == "Require explicit approval before deploying."
            assert entries[0].source_interaction_ids == list(
                saved.execution_run.binding.source_interaction_ids
            )
    assert calls == []
    retained = storage.get_agent_run(run.id)
    assert retained is not None
    assert "overflow" not in retained.generation_request_snapshot
    assert (
        retained.generation_request_snapshot["output_schema_name"]
        == "StructuredPlaybookList"
    )
    assert saved.execution_run is not None
    assert retained.committed_output == saved.execution_run.committed_output
    # Supply the independent classic consolidator's unchanged verdict. Exercise
    # the actual window publisher, not a direct storage insertion or V8 fallback.
    monkeypatch.setattr(
        PlaybookConsolidator,
        "deduplicate",
        lambda _self, results, *_args, **_kwargs: (
            [item for group in results for item in group],
            [],
            [],
        ),
    )
    executor.execute(window, token)
    rows = storage.get_user_playbooks(user_id="u", agent_version="v")
    assert len(rows) == int(nonempty)
    if nonempty:
        assert rows[0].content == "Require explicit approval before deploying."
        assert rows[0].source_interaction_ids == list(
            run.binding.source_interaction_ids
        )
    assert storage.extraction_status("u", "procedural")["status"] == "done"
    assert calls == []


def test_retirement_window_delivers_zero_learning_quantity_and_keeps_token_charges_on_ack_retry(
    pipeline, monkeypatch
):
    from reflexio.server import metering_events

    engine, storage, window, token, client, calls, _ = setup(
        pipeline, monkeypatch, retire=True
    )
    executor = WindowExecutor(engine.request_context, client)
    events = []
    monkeypatch.setattr(
        metering_events,
        "record_usage_event_strict",
        lambda **event: events.append(event),
    )
    executor.execute(window, token)
    effect_user, token = storage.claim_extraction("procedural-effects", 600)
    assert effect_user == window.user_id
    [(pending, effects)] = storage.pending_extraction_effects(
        user_id=effect_user, token=token
    )
    assert pending.window_id == window.window_id
    assert effects["billing"]["ids"] == []
    assert effects["billing"]["prompt_tokens"] == 300
    assert effects["billing"]["completion_tokens"] == 60
    dispatch = Mock()
    monkeypatch.setattr(
        "reflexio.server.services.playbook.service.PlaybookGenerationService.emit_generation_side_effects",
        dispatch,
    )
    ack = storage.ack_extraction_effects
    monkeypatch.setattr(
        storage,
        "ack_extraction_effects",
        Mock(side_effect=RuntimeError("synthetic acknowledgement failure")),
    )
    from reflexio.server.services.durable_learning.user_lease import (
        owned_user_work_context,
    )

    with owned_user_work_context(storage, window.user_id, token, 300):
        with pytest.raises(RuntimeError, match="acknowledgement failure"):
            executor.deliver(pending, effects, token=token)
        assert storage.pending_extraction_effects()
        monkeypatch.setattr(storage, "ack_extraction_effects", ack)
        executor.deliver(pending, effects, token=token)
    assert len(calls) == 3
    assert storage.pending_extraction_effects() == []
    assert dispatch.call_count == 1
    assert not any(event["event_name"] == "learnings_generated" for event in events)
    charges = [event for event in events if event["event_name"] == "extraction_tokens"]
    # Receipt delivery may repeat before acknowledgement; stable keys make these
    # the same financial fact, rather than a newly minted charge.
    assert charges and len({event["event_key"] for event in charges}) == 1
    assert all(event["event_key"] == f"tok:{window.window_id}" for event in charges)
    assert all(
        event["prompt_tokens"] == 300 and event["completion_tokens"] == 60
        for event in charges
    )


def test_cli_partial_execution_resumes_reader_and_preserves_physical_allowance(
    pipeline, monkeypatch, tmp_path
):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError

    engine, storage, window, token, client, calls, _, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )
    interrupted = False

    def outage(params, timeout):
        nonlocal interrupted
        body = yaml.safe_load(params["messages"][1]["content"])
        if "source_observations" in body and not interrupted:
            interrupted = True
            calls.append(params)
            raise ProviderRequestGuardError("synthetic provider suspension")
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", outage)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises(ProviderRequestGuardError, match="synthetic"):
        executor.execute(window, token)
    partial = storage.get_agent_run(f"window:{window.window_id}")
    assert partial is not None and partial.committed_output is None
    checkpoint = partial.generation_request_snapshot["procedural_execution"]
    assert checkpoint["physical_calls"] == 2
    assert len(checkpoint["completed"]) == 1
    assert len(calls) == 2
    window = storage.prepare_extraction(window.user_id, token)
    executor.execute(window, token)
    final = storage.get_agent_run(partial.id)
    assert final is not None and final.status.value == "finalized"
    checkpoint = final.generation_request_snapshot["procedural_execution"]
    assert checkpoint["physical_calls"] == 4
    assert len(checkpoint["completed"]) == 3
    assert len(calls) == 4
    assert (
        len(
            [
                params
                for params in calls
                if "source_observations"
                not in yaml.safe_load(params["messages"][1]["content"])
                and "result_review_units"
                not in yaml.safe_load(params["messages"][1]["content"])
            ]
        )
        == 1
    )


@pytest.mark.parametrize("change", ["source", "history", "profile"])
def test_cli_partial_source_history_or_profile_change_holds_before_redraw(
    pipeline, monkeypatch, tmp_path, change
):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError

    engine, storage, window, token, client, calls, existing, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )

    def outage(params, timeout):
        if "source_observations" in yaml.safe_load(params["messages"][1]["content"]):
            calls.append(params)
            raise ProviderRequestGuardError("synthetic suspension")
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", outage)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises(ProviderRequestGuardError):
        executor.execute(window, token)
    if change == "source":
        storage._execute(
            "UPDATE interactions SET content=? WHERE user_id=? AND interaction_id=?",
            ("Changed source", "u", 1),
        )
    elif change == "history":
        storage._execute(
            "UPDATE user_playbooks SET content=? WHERE user_playbook_id=?",
            ("Changed history", existing[0].user_playbook_id),
        )
    else:
        config = engine.request_context.configurator.get_config().model_copy(deep=True)
        config.llm_config.generation_model_name = "local-cli/codex/different-model"
        engine.request_context.configurator.set_config(config)
    window = storage.prepare_extraction(window.user_id, token)
    with pytest.raises((ExtractionInputIntegrityError, RuntimeError)):
        executor.execute(window, token)
    assert len(calls) == 2
    retained = storage.get_agent_run(f"window:{window.window_id}")
    assert retained is not None and retained.committed_output is None
    assert (
        retained.generation_request_snapshot["procedural_execution"]["physical_calls"]
        == 2
    )


def test_cli_invalid_terminal_output_is_not_rerolled_by_worker_retry(
    pipeline, monkeypatch, tmp_path
):
    engine, storage, window, token, client, calls, _, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )

    def malformed(params, timeout):
        if "source_observations" in yaml.safe_load(params["messages"][1]["content"]):
            calls.append(params)
            return litellm.ModelResponse(
                model=params["model"],
                choices=[
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Invalid terminal result",
                        },
                        "finish_reason": "stop",
                    }
                ],
            )
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", malformed)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises(RuntimeError, match="did not complete"):
        executor.execute(window, token)
    count = len(calls)
    partial = storage.get_agent_run(f"window:{window.window_id}")
    assert partial is not None and partial.committed_output is None
    assert partial.generation_request_snapshot["procedural_execution"]["failed"]
    window = storage.prepare_extraction(window.user_id, token)
    with pytest.raises(RuntimeError, match="did not complete"):
        executor.execute(window, token)
    assert len(calls) == count


def test_cli_physical_limit_cannot_reset_across_six_worker_turns(
    pipeline, monkeypatch, tmp_path
):
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError

    engine, storage, window, token, client, calls, _, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )

    def outage(params, timeout):
        if "source_observations" in yaml.safe_load(params["messages"][1]["content"]):
            calls.append(params)
            raise ProviderRequestGuardError("synthetic suspension")
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", outage)
    executor = WindowExecutor(engine.request_context, client)
    for index in range(6):
        with pytest.raises(ProviderRequestGuardError, match="synthetic|allowance"):
            executor.execute(window, token)
        saved = storage.get_agent_run(f"window:{window.window_id}")
        assert saved is not None and saved.committed_output is None
        assert saved.generation_request_snapshot["procedural_execution"][
            "physical_calls"
        ] == min(index + 2, 6)
        assert len(calls) == min(index + 2, 6)
        window = storage.prepare_extraction(window.user_id, token)


def test_cli_checkpoint_rejects_stale_claim_and_checkpoint_revision(
    pipeline, monkeypatch, tmp_path
):
    from copy import deepcopy

    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.services.durable_learning.user_lease import (
        adopt_agent_run_for_execution,
        owned_user_work_context,
    )
    from reflexio.server.services.storage.storage_base import LeaseLostError

    engine, storage, window, token, client, calls, _, wire = cli_setup(
        pipeline, monkeypatch, tmp_path
    )

    def outage(params, timeout):
        if "source_observations" in yaml.safe_load(params["messages"][1]["content"]):
            calls.append(params)
            raise ProviderRequestGuardError("synthetic suspension")
        return wire(params, timeout)

    monkeypatch.setattr(client, "_completion_with_hard_timeout", outage)
    with pytest.raises(ProviderRequestGuardError):
        WindowExecutor(engine.request_context, client).execute(window, token)
    old = storage.get_agent_run(f"window:{window.window_id}")
    assert old is not None
    checkpoint = old.generation_request_snapshot["procedural_execution"]
    storage.release_user_extraction(window.user_id, token)
    user, replacement = storage.claim_extraction("replacement-owner", 600)
    assert user == window.user_id
    with owned_user_work_context(storage, user, replacement, 300):
        current = adopt_agent_run_for_execution(storage, old)
        value = deepcopy(checkpoint)
        value["revision"] += 1
        with pytest.raises(LeaseLostError):
            storage.checkpoint_procedural_execution(
                old, expected=checkpoint, save=value
            )
        stale = deepcopy(checkpoint)
        stale["revision"] -= 1
        value["revision"] = checkpoint["revision"]
        with pytest.raises(LeaseLostError):
            storage.checkpoint_procedural_execution(current, expected=stale, save=value)
    saved = storage.get_agent_run(old.id)
    assert saved is not None
    assert saved.generation_request_snapshot["procedural_execution"] == checkpoint
    assert len(calls) == 2


@pytest.mark.parametrize("cached_replay", [False, True])
@pytest.mark.parametrize("extra_prior_usage", [False, True])
@pytest.mark.parametrize("extra_current_usage", [False, True])
def test_mixed_transport_resume_and_ack_keep_cumulative_usage_exactly_once(
    pipeline,
    monkeypatch,
    tmp_path,
    extra_current_usage,
    extra_prior_usage,
    cached_replay,
):
    from reflexio.models.config_schema import LLMConfig
    from reflexio.server import metering_events
    from reflexio.server.llm._litellm_types import ProviderRequestGuardError
    from reflexio.server.services.durable_learning.user_lease import (
        owned_user_work_context,
    )
    from reflexio.server.services.playbook.procedural.transport import BATCH_TOOL

    engine, storage, window, token, _, _, _, _ = cli_setup(
        pipeline, monkeypatch, tmp_path
    )
    config = engine.request_context.configurator.get_config().model_copy(deep=True)
    config.llm_config = LLMConfig(generation_model_name="gpt-4o-mini")
    engine.request_context.configurator.set_config(config)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-4o-mini",
            max_tokens=16384,
            fallback_models=["local-cli/codex/fixture-model"],
        )
    )
    calls = []
    suspended = False

    def wire(params, _timeout):
        nonlocal suspended
        calls.append(params)
        assert params["model"] == "gpt-4o-mini"
        assert {item["function"]["name"] for item in params["tools"]} == {BATCH_TOOL}
        body = yaml.safe_load(params["messages"][1]["content"])
        if "result_review_units" in body:
            name = "review_result"
            arguments = {
                "unit": "U0",
                "decision": "accept",
                "result_support": "pass",
                "scope_conditions": "pass",
                "useful_nonduplicate": "pass",
                "preservation_and_correction": "pass",
                "resource_consistency": "pass",
                "reason": "The source supports retirement.",
            }
        elif "source_observations" in body:
            if not suspended:
                suspended = True
                raise ProviderRequestGuardError("synthetic compiler suspension")
            name = "retire_procedure"
            arguments = {
                "group": "deployment",
                "target": "P1",
                "evidence": body["source_observations"][0]["evidence"],
            }
            if extra_current_usage:
                # Exercise the native observer for completion usage outside the
                # procedural stage trace; no provider is called here.
                client._log_token_usage(
                    params,
                    litellm.ModelResponse(
                        usage={
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                        }
                    ),
                )
        else:
            if extra_prior_usage:
                client._log_token_usage(
                    params,
                    litellm.ModelResponse(
                        usage={
                            "prompt_tokens": 11,
                            "completion_tokens": 5,
                            "total_tokens": 16,
                            "cache_read_input_tokens": 4,
                            "cache_creation_input_tokens": 2,
                        }
                    ),
                )
            name = "record_observation"
            arguments = {
                "topic": "Deployment",
                "observation": "The old command is no longer valid.",
                "evidence": body["sources"][0]["segments"][0]["excerpts"][0]["id"],
            }
        payload = {
            "calls": [
                {"id": "member", "name": name, "arguments": arguments},
                {
                    "id": "done",
                    "name": "finish_batch",
                    "arguments": {"status": "complete", "reason": "Complete batch."},
                },
            ]
        }
        return litellm.ModelResponse(
            model=params["model"],
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(calls)}",
                                "type": "function",
                                "function": {
                                    "name": BATCH_TOOL,
                                    "arguments": json.dumps(payload),
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        )

    monkeypatch.setattr(client, "_completion_with_hard_timeout", wire)
    executor = WindowExecutor(engine.request_context, client)
    with pytest.raises(ProviderRequestGuardError, match="compiler suspension"):
        executor.execute(window, token)
    partial = storage.get_agent_run(f"window:{window.window_id}")
    assert partial is not None
    trace = partial.generation_request_snapshot["procedural_execution"]["trace"]
    assert sum(turn["prompt_tokens"] or 0 for turn in trace["turns"]) == 100
    prior_observed = partial.generation_request_snapshot["procedural_execution"][
        "observed_usage"
    ]
    assert prior_observed["prompt_tokens"] == (111 if extra_prior_usage else 100)
    assert prior_observed["completion_tokens"] == (25 if extra_prior_usage else 20)
    assert prior_observed["cache_read_input_tokens"] == (4 if extra_prior_usage else 0)
    assert prior_observed["cache_write_input_tokens"] == (2 if extra_prior_usage else 0)
    window = storage.prepare_extraction(window.user_id, token)
    if cached_replay:
        from reflexio.server.services.playbook.service import PlaybookGenerationService

        resolve = PlaybookGenerationService._resolve_write_plan

        def fail_preparation(service, results):
            resolve(service, results)
            raise RuntimeError("synthetic post-review preparation failure")

        monkeypatch.setattr(
            PlaybookGenerationService, "_resolve_write_plan", fail_preparation
        )
        with pytest.raises(RuntimeError, match="post-review preparation failure"):
            executor.execute(window, token)
        assert len(calls) == 4
        monkeypatch.setattr(PlaybookGenerationService, "_resolve_write_plan", resolve)
        window = storage.prepare_extraction(window.user_id, token)
    executor.execute(window, token)
    assert len(calls) == 4
    effect_user, token = storage.claim_extraction("mixed-effects-owner", 600)
    assert effect_user == window.user_id
    [(pending, effects)] = storage.pending_extraction_effects(
        user_id=effect_user, token=token
    )
    expected_prompt = (
        300 + (7 if extra_current_usage else 0) + (11 if extra_prior_usage else 0)
    )
    expected_completion = (
        60 + (3 if extra_current_usage else 0) + (5 if extra_prior_usage else 0)
    )
    assert effects["billing"]["ids"] == []
    assert effects["billing"]["prompt_tokens"] == expected_prompt
    assert effects["billing"]["completion_tokens"] == expected_completion
    assert effects["billing"]["cache_read_input_tokens"] == (
        4 if extra_prior_usage else 0
    )
    assert effects["billing"]["cache_write_input_tokens"] == (
        2 if extra_prior_usage else 0
    )
    events = []
    monkeypatch.setattr(
        metering_events,
        "record_usage_event_strict",
        lambda **event: events.append(event),
    )
    ack = storage.ack_extraction_effects
    monkeypatch.setattr(
        storage,
        "ack_extraction_effects",
        Mock(side_effect=RuntimeError("synthetic effects acknowledgement failure")),
    )
    with owned_user_work_context(storage, effect_user, token, 300):
        with pytest.raises(RuntimeError, match="acknowledgement failure"):
            executor.deliver(pending, effects, token=token)
        monkeypatch.setattr(storage, "ack_extraction_effects", ack)
        executor.deliver(pending, effects, token=token)
    charges = [event for event in events if event["event_name"] == "extraction_tokens"]
    assert len(charges) == 2
    assert len({event["event_key"] for event in charges}) == 1
    assert all(
        event["prompt_tokens"] == expected_prompt
        and event["completion_tokens"] == expected_completion
        for event in charges
    )
    assert all(
        event["cache_read_input_tokens"] == (4 if extra_prior_usage else 0)
        and event["cache_write_input_tokens"] == (2 if extra_prior_usage else 0)
        for event in charges
    )
    assert not any(event["event_name"] == "learnings_generated" for event in events)
    assert storage.pending_extraction_effects() == []
    assert len(calls) == 4
