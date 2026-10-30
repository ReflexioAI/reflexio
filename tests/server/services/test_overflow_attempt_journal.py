"""Saved quota is validated before native execution and locked provider admission."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from reflexio.lib._base import create_generation_litellm_client
from reflexio.server.llm._litellm_types import (
    LiteLLMConfig,
    ModelProvenance,
    ProviderRequestGuardError,
)
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.window_executor import (
    WindowExecutor,
    load_window_inputs,
)
from reflexio.server.services.extraction.chunk_gate import judge_extraction_chunks
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.overflow_coordinator import OverflowCoordinator
from reflexio.server.services.extraction.overflow_presentation import (
    present_extraction_chunk,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    OverflowCheckpointConflictError,
    validate_extraction_overflow_manifest,
)
from tests.server.services.test_durable_window_pipeline import (
    _overflow_coordinator,
    _prepared_presentation_text,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
journal_pipeline = imported_pipeline
UNKNOWN = {"id": "old", "usage": None, "model_provenance": None}
USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "cache_read_input_tokens": 0,
    "cache_write_input_tokens": 0,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [("version", True), ("version", 1.0), ("attempt_limit", 128.0)],
)
def test_protocol_header_requires_integer_literals(field, value):
    manifest = OverflowCoordinator.initial_manifest("source")
    assert validate_extraction_overflow_manifest(manifest) is manifest
    manifest[field] = value
    with pytest.raises(OverflowCheckpointConflictError, match="overflow marker"):
        validate_extraction_overflow_manifest(manifest)


@pytest.mark.parametrize(
    "attempts",
    [
        [{"id": "old"}],
        [{"id": "old", "usage": None}],
        [{**UNKNOWN, "id": ""}],
        [UNKNOWN, UNKNOWN],
        [{**UNKNOWN, "usage": {"prompt_tokens": 10}}],
        [{**UNKNOWN, "usage": {**USAGE, "prompt_tokens": True}}],
        [{**UNKNOWN, "usage": {**USAGE, "completion_tokens": -1}}],
        [{**UNKNOWN, "model_provenance": {"model_name": 42}}],
        [{**UNKNOWN, "model_provenance": {"model": "invented"}}],
        [{**UNKNOWN, "id": str(index)} for index in range(129)],
    ],
)
def test_malformed_saved_journal_blocks_executor_constructor_and_locked_reservation(
    journal_pipeline, monkeypatch, attempts
):
    engine, _, _ = journal_pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    storage, window, token = _overflow_coordinator(journal_pipeline, attempts=attempts)
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    before = parent.generation_request_snapshot
    dispatch = Mock(side_effect=AssertionError("Malformed quota must not dispatch"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_deadline", dispatch)
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    with pytest.raises(InputBudgetConfigurationError, match="attempt|provenance"):
        executor.execute(window, token)
    with pytest.raises(OverflowCheckpointConflictError, match="attempt|provenance"):
        OverflowCoordinator(
            storage,
            window,
            token,
            before["overflow"],
            source_digest="captured-source",
            context_limits={},
        )
    with pytest.raises(OverflowCheckpointConflictError, match="attempt|provenance"):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            reserve_attempt_id="new-paid-attempt",
        )
    after = storage.get_agent_run(parent.id)
    assert after is not None and after.generation_request_snapshot == before
    assert storage.count_all_profiles() == 0
    dispatch.assert_not_called()


@pytest.mark.parametrize("corruption", [None, "journal", "checkpoint"])
def test_actual_client_guard_rechecks_locked_journal_and_preserves_unknown_usage(
    journal_pipeline, monkeypatch, corruption
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-provider")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    storage, window, token = _overflow_coordinator(journal_pipeline, attempts=[UNKNOWN])
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={"gpt-4o": 128000},
    )
    assert coordinator.token_totals.prompt_tokens == 0
    if corruption is not None:
        snapshot = parent.generation_request_snapshot
        if corruption == "journal":
            snapshot["overflow"]["attempts"] = [{"id": "old"}]
        else:
            del snapshot["overflow"]["checkpoint"]
        with storage._stream_sql() as db:
            db.query(
                "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
                (json.dumps(snapshot), parent.id),
            )
    before = storage.get_agent_run(parent.id)
    assert before is not None
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"profiles":[]}', tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    wire = Mock(return_value=response)
    monkeypatch.setattr(client, "_completion_with_deadline", wire)
    with coordinator.provider_scope():
        if corruption is not None:
            with pytest.raises(ProviderRequestGuardError):
                client.generate_chat_response_with_provenance(
                    messages=[{"role": "user", "content": "Verified original source"}],
                    response_format=StructuredProfilesOutput,
                )
        else:
            result = client.generate_chat_response_with_provenance(
                messages=[{"role": "user", "content": "Verified original source"}],
                response_format=StructuredProfilesOutput,
            )
            assert result.value == StructuredProfilesOutput(profiles=[])
    after = storage.get_agent_run(parent.id)
    assert after is not None
    if corruption is not None:
        wire.assert_not_called()
        assert after.generation_request_snapshot == before.generation_request_snapshot
    else:
        wire.assert_called_once()
        manifest = after.generation_request_snapshot["overflow"]
        assert manifest["attempts"][0] == UNKNOWN
        assert len(manifest["attempts"]) == 2
        assert manifest["attempts"][1]["usage"] == USAGE
        assert coordinator.token_totals.prompt_tokens == 10
        assert coordinator.token_totals.completion_tokens == 2
    assert storage.count_all_profiles() == 0


@pytest.mark.parametrize(
    "checkpoint",
    [
        "missing",
        None,
        {"stage": "prepare"},
        {"completed_steps": {}},
        {"stage": "", "completed_steps": {}},
        {"stage": "prepare", "completed_steps": {str(i): {} for i in range(129)}},
        {"stage": "prepare", "completed_steps": {"paid-step": None}},
        {"stage": "prepare", "completed_steps": {"paid-step": {"output": False}}},
        {
            "stage": "prepare",
            "completed_steps": {"paid-step": {"output": [], "model_provenance": None}},
        },
        {
            "stage": "prepare",
            "completed_steps": {"": {"output": False, "model_provenance": {}}},
        },
    ],
)
def test_incomplete_checkpoint_blocks_load_and_locked_updates(
    journal_pipeline, monkeypatch, checkpoint
):
    engine, _, _ = journal_pipeline
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    storage, window, token = _overflow_coordinator(journal_pipeline, attempts=[UNKNOWN])
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    snapshot = parent.generation_request_snapshot
    if checkpoint == "missing":
        del snapshot["overflow"]["checkpoint"]
    else:
        snapshot["overflow"]["checkpoint"] = checkpoint
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    dispatch = Mock(side_effect=AssertionError("Damaged checkpoint must not dispatch"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_deadline", dispatch)
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    with pytest.raises(InputBudgetConfigurationError, match="checkpoint|receipt"):
        executor.execute(window, token)
    with pytest.raises(OverflowCheckpointConflictError, match="checkpoint|receipt"):
        OverflowCoordinator(
            storage,
            window,
            token,
            snapshot["overflow"],
            source_digest="captured-source",
            context_limits={},
        )
    with pytest.raises(OverflowCheckpointConflictError, match="checkpoint|receipt"):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            reserve_attempt_id="must-not-reserve",
        )
    after = storage.get_agent_run(parent.id)
    assert after is not None and after.generation_request_snapshot == snapshot
    assert storage.count_all_profiles() == 0
    dispatch.assert_not_called()


def test_checkpoint_writer_cannot_drop_required_progress_fields(journal_pipeline):
    storage, window, token = _overflow_coordinator(journal_pipeline, attempts=[UNKNOWN])
    before = storage.get_agent_run(f"window:{window.window_id}")
    assert before is not None
    with pytest.raises(OverflowCheckpointConflictError, match="checkpoint"):
        storage.checkpoint_extraction_overflow(
            window,
            token,
            expected_revision=0,
            source_digest="captured-source",
            checkpoint={"stage": "reconcile"},
        )
    after = storage.get_agent_run(before.id)
    assert after is not None
    assert after.generation_request_snapshot == before.generation_request_snapshot


@pytest.mark.parametrize("output", [False, [], {}])
def test_false_and_empty_receipts_remain_completed_after_reload(
    journal_pipeline, output
):
    storage, window, token = _overflow_coordinator(journal_pipeline, attempts=[UNKNOWN])
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )
    coordinator.complete_step("saved-step", output, ModelProvenance())
    saved = storage.get_agent_run(parent.id)
    assert saved is not None
    reloaded = OverflowCoordinator(
        storage,
        window,
        token,
        saved.generation_request_snapshot["overflow"],
        source_digest="captured-source",
        context_limits={},
    )
    assert reloaded.completed_step("saved-step") == {
        "output": output,
        "model_provenance": {"model_name": None, "provider": None},
    }
    assert saved.generation_request_snapshot["overflow"]["attempts"] == [UNKNOWN]


def test_gate_cache_miss_cannot_dispatch_after_locked_receipt_becomes_null(
    journal_pipeline, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-no-provider")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    storage, window, token = _overflow_coordinator(
        journal_pipeline, attempts=[UNKNOWN], actual_source=True
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    assert parent is not None
    prepared = prepare_extraction_input(
        load_window_inputs(storage, window), user_id=window.user_id
    )
    coordinator = OverflowCoordinator(
        storage,
        window,
        token,
        parent.generation_request_snapshot["overflow"],
        source_digest=prepared.source_digest,
        context_limits={"gpt-4o": 128000},
        prepared=prepared,
    )
    with present_extraction_chunk(prepared, prepared.chunks[0], coordinator.record):
        prompt = _prepared_presentation_text()
    key = coordinator.step_key(
        "gate-v1",
        {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]},
    )
    assert coordinator.completed_step(key) is None
    snapshot = parent.generation_request_snapshot
    snapshot["overflow"]["checkpoint"]["completed_steps"][key] = None
    with storage._stream_sql() as db:
        db.query(
            "UPDATE _agent_runs SET generation_request_snapshot=? WHERE id=?",
            (json.dumps(snapshot), parent.id),
        )
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", max_tokens=1024, fallback_models=[])
    )
    wire = Mock(side_effect=AssertionError("Null receipt cannot replay paid work"))
    monkeypatch.setattr(client, "_completion_with_deadline", wire)
    with pytest.raises(ProviderRequestGuardError):
        judge_extraction_chunks(
            coordinator=coordinator,
            prepared=prepared,
            record=coordinator.record,
            client=client,
            model="gpt-4o",
            build_prompt=_prepared_presentation_text,
            minimum_after_positive=1,
        )
    after = storage.get_agent_run(parent.id)
    assert after is not None and after.generation_request_snapshot == snapshot
    assert storage.count_all_profiles() == 0
    wire.assert_not_called()
