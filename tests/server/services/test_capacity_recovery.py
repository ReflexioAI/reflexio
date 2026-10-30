"""Capacity opt-in cannot leak or erase durable recovery accounting."""

from contextvars import copy_context
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from reflexio.server.llm.input_budget import (
    InputBudgetExceededError,
    extraction_request_guard,
    extraction_use_model_capacity,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.components.reconciliation import (
    Decisions,
    Reconciler,
)
from reflexio.server.services.storage.storage_base._extraction_stream import (
    LeaseLostError,
    OverflowAttemptBudgetExceededError,
    OverflowCheckpointConflictError,
    WindowInputsDeletedError,
)
from tests.server.services.test_durable_window_pipeline import _overflow_coordinator
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline


def test_audited_capacity_retains_real_output_margin_fallback_and_context_isolation(
    monkeypatch,
):
    counter = Mock(return_value=90000)
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", counter
    )
    reserve = Mock()
    guard = extraction_request_guard(
        context_limits={"primary": 512000, "fallback": 110000}, reserve_attempt=reserve
    )
    request = {
        "model": "primary",
        "max_tokens": 128000,
        "messages": [{"role": "user", "content": "source"}],
    }
    ordinary = copy_context()
    with pytest.raises(InputBudgetExceededError):
        guard(request, 120, (), False)
    token = extraction_use_model_capacity.set(True)
    try:
        guard(request, 120, (), False)
        with pytest.raises(InputBudgetExceededError):
            ordinary.run(guard, request, 120, (), False)
        with pytest.raises(InputBudgetExceededError):
            guard({**request, "model": "fallback", "max_tokens": 8192}, 120, (), False)
        counter.return_value = 321000  # 385200 with margin > 384000 native headroom.
        with pytest.raises(InputBudgetExceededError):
            guard(request, 120, (), False)
    finally:
        extraction_use_model_capacity.reset(token)
    assert reserve.call_count == 1
    assert not extraction_use_model_capacity.get()


def test_reconciler_scopes_capacity_even_when_the_provider_raises():
    def fail(**kwargs):
        assert extraction_use_model_capacity.get()
        raise RuntimeError("provider failed")

    engine = Reconciler(
        cast(
            LiteLLMClient, SimpleNamespace(generate_chat_response_with_provenance=fail)
        ),
        "model",
    )
    with pytest.raises(RuntimeError, match="provider failed"):
        engine.ask("reconcile-method", {}, Decisions, lambda _: None, {})
    assert not extraction_use_model_capacity.get()


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "source",
        "window",
        "revision",
        "lease",
        "missing",
        "parent_running",
        "quota",
    ],
)
def test_fenced_recovery_preserves_original_journal_and_rejects_changed_authority(
    pipeline, failure, kind
):
    if kind == "playbook":
        from reflexio.models.config_schema import UserPlaybookExtractorConfig

        configurator = pipeline[0].request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Extract procedural advice"
                    ),
                }
            )
        )
    attempts = [
        {"id": f"paid-{i}", "usage": None, "model_provenance": None}
        for i in range(128 if failure == "quota" else 5)
    ]
    storage, window, token = _overflow_coordinator(
        pipeline, actual_source=True, attempts=attempts
    )
    parent = storage.get_agent_run(f"window:{window.window_id}")
    original = parent.generation_request_snapshot["overflow"]
    storage.fail_extraction_input_budget(window, token, "input_budget_exceeded")
    token = storage.claim_user_extraction(window.user_id, "operator-recovery-test", 300)
    assert token is not None
    expected = None
    supplied_window = window
    supplied_token = token
    revision = original["revision"]
    if failure == "source":
        storage.conn.execute(
            "UPDATE interactions SET content='changed' WHERE interaction_id=?",
            (window.manifest[0]["interaction_id"],),
        )
        storage.conn.commit()
        expected = WindowInputsDeletedError
    elif failure == "window":
        supplied_window = replace(window, policy={**window.policy, "changed": True})
        expected = OverflowCheckpointConflictError
    elif failure == "revision":
        revision += 1
        expected = OverflowCheckpointConflictError
    elif failure == "lease":
        supplied_token = "stale-lease"
        expected = LeaseLostError
    elif failure == "missing":
        storage.conn.execute("DELETE FROM _agent_runs WHERE id=?", (parent.id,))
        storage.conn.commit()
        expected = OverflowCheckpointConflictError
    elif failure == "parent_running":
        storage.conn.execute(
            "UPDATE _agent_runs SET status='running' WHERE id=?", (parent.id,)
        )
        storage.conn.commit()
        expected = OverflowCheckpointConflictError
    elif failure == "quota":
        expected = OverflowAttemptBudgetExceededError

    def recover():
        storage.recover_extraction_input_budget(
            supplied_window,
            supplied_token,
            expected_revision=revision,
            source_digest=original["source_digest"],
            expected_attempts=1,
        )

    if expected:
        with pytest.raises(expected):
            recover()
    else:
        recover()
        assert (
            storage.get_agent_run(parent.id).generation_request_snapshot["overflow"]
            == original
        )
        with pytest.raises(OverflowCheckpointConflictError, match="already re-armed"):
            recover()
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT attempts,last_error,retry_at,completed_seq FROM extraction_cursors WHERE user_id=? AND kind=?",
            (window.user_id, window.kind),
        )[0]
    assert cursor["attempts"] == 1
    assert cursor["last_error"] == "input_budget_exceeded"
    assert cursor["completed_seq"] == window.predecessor
    assert (cursor["retry_at"] == float("inf")) is bool(failure)


@pytest.mark.integration
@pytest.mark.parametrize("change", [None, "receipt", "missing_parent"])
def test_read_only_plan_then_explicit_recovery_is_once_only(pipeline, change):
    from reflexio.server.services.extraction.input_recovery import (
        apply_input_budget_recovery,
        plan_input_budget_recovery,
    )

    storage, window, token = _overflow_coordinator(pipeline, actual_source=True)
    storage.fail_extraction_input_budget(window, token, "input_budget_exceeded")
    if change == "missing_parent":
        storage.conn.execute(
            "DELETE FROM _agent_runs WHERE id=?", (f"window:{window.window_id}",)
        )
        storage.conn.commit()
    before = storage.conn.total_changes
    receipt = plan_input_budget_recovery(storage, window.window_id)
    assert storage.conn.total_changes == before
    if change == "missing_parent":
        assert receipt["reason"] == "missing_parent_unknown_history"
    else:
        assert receipt["recoverable"]
    if change == "receipt":
        receipt["state_sha256"] = "wrong"
    if change:
        with pytest.raises(OverflowCheckpointConflictError):
            apply_input_budget_recovery(storage, receipt)
        assert storage.conn.total_changes == before
    else:
        apply_input_budget_recovery(storage, receipt)
        with pytest.raises(OverflowCheckpointConflictError):
            apply_input_budget_recovery(storage, receipt)


@pytest.mark.parametrize("path", ["fallback", "repair"])
def test_actual_client_checks_audited_fallback_and_repair_before_dispatch(
    monkeypatch, path
):
    from reflexio.server.llm.input_budget import extraction_provider_guard
    from reflexio.server.llm.litellm_client import (
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
    )

    def count(**kwargs):
        if "text" in kwargs:
            return 300
        return 90000 if len(kwargs["messages"]) == 2 else 321000

    monkeypatch.setattr("reflexio.server.llm.input_budget.litellm.token_counter", count)
    client = LiteLLMClient(
        LiteLLMConfig(
            model="gpt-4o",
            max_tokens=128000,
            max_retries=0,
            fallback_models=["gpt-4o-mini"] if path == "fallback" else [],
        )
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="invalid JSON", tool_calls=None, refusal=None
                ),
                finish_reason="stop",
            )
        ],
        usage=None,
        model="gpt-4o",
        _hidden_params={"custom_llm_provider": "openai"},
    )
    dispatch = Mock(
        side_effect=LiteLLMClientError("transport failure")
        if path == "fallback"
        else None,
        return_value=response,
    )
    monkeypatch.setattr(client, "_completion_with_deadline", dispatch)
    reserve = Mock()
    token = extraction_provider_guard.set(
        extraction_request_guard(
            context_limits={"gpt-4o": 512000, "gpt-4o-mini": 180000},
            reserve_attempt=reserve,
        )
    )
    try:
        engine = Reconciler(client, "gpt-4o")
        with pytest.raises(InputBudgetExceededError):
            engine.ask(
                "reconcile-method", {}, Decisions, lambda _: None, {"memory": []}
            )
    finally:
        extraction_provider_guard.reset(token)
    assert dispatch.call_count == reserve.call_count == engine.calls == 1
    assert not extraction_use_model_capacity.get()


@pytest.mark.integration
def test_twenty_k_initial_geometry_runs_complete_http_window_under_original_quota(
    pipeline, monkeypatch
):
    import json

    from reflexio.models.config_schema import UserPlaybookExtractorConfig
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    engine, worker, http = pipeline
    configurator = engine.request_context.configurator
    configurator.set_config(
        configurator.get_config().model_copy(
            update={
                "profile_extractor_config": None,
                "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                    extraction_definition_prompt="Learn explicit user instructions"
                ),
            }
        )
    )
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON", json.dumps({"minimax/MiniMax-M3": 512000})
    )
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client",
        lambda _: LiteLLMClient(
            LiteLLMConfig(
                model="minimax/MiniMax-M3",
                max_tokens=128000,
                timeout=5,
                fallback_models=[],
            )
        ),
    )
    storage = engine.get_storage()
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(
        storage,
        "_expand_document",
        lambda *_args, **_kwargs: pytest.fail(
            "Source-quoted publication cannot expand source instructions"
        ),
    )
    instruction = "Never deploy without explicit confirmation."
    large_original = "0123456789abcdef " * 100000
    requests = []

    def respond(params, hard_timeout):
        assert "Select original evidence needed" in params["messages"][0]["content"]
        body = json.loads(params["messages"][1]["content"])
        selected = [
            {"kind": "instruction", "evidence_refs": [item["evidence_ref"]]}
            for item in body["original_chunk"]["evidence"]
            if item["text"] == instruction
        ]
        requests.append(params)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {"evidence": selected, "candidate_evidence": []}
                        ),
                        tool_calls=None,
                        refusal=None,
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
        LiteLLMClient, "_completion_with_deadline", staticmethod(respond)
    )
    result = http.post(
        "/api/publish_interaction",
        json={
            "user_id": "u",
            "request_id": "large-initial-capacity",
            "session_id": "large",
            "source": "test",
            "force_extraction": True,
            "interaction_data_list": [
                {"role": "User", "content": instruction, "created_at": 100},
                {
                    "role": "Assistant",
                    "content": "Inspection output",
                    "created_at": 101,
                    "tools_used": [
                        {
                            "tool_name": "inspection",
                            "status": "success",
                            "tool_data": {"output": large_original},
                        }
                    ],
                },
            ],
        },
    )
    assert result.status_code == 200, result.json()
    assert result.json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 1
    assert storage.extraction_status("u", "large-initial-capacity")["status"] == "done"
    with storage._stream_sql(read_only=True) as db:
        window = db.query(
            "SELECT window_id,manifest FROM extraction_windows WHERE user_id=? AND kind=?",
            ("u", "playbook"),
        )[0]
        cursor = db.query(
            "SELECT completed_seq,attempts,last_error FROM extraction_cursors WHERE user_id=? AND kind=?",
            ("u", "playbook"),
        )[0]
        saved = db.query(
            "SELECT content,source_span FROM user_playbooks WHERE user_id=?", ("u",)
        )
    parent = storage.get_agent_run(f"window:{window['window_id']}")
    manifest = parent.generation_request_snapshot["overflow"]
    assert manifest["input_chunk_tokens"] == 20000
    assert 1 < len(manifest["chunk_digests"]) <= 48
    assert len(requests) == len(manifest["attempts"]) <= 128
    assert (
        cursor["completed_seq"] == 2
        and cursor["attempts"] == 0
        and cursor["last_error"] is None
    )
    assert saved == [{"content": instruction, "source_span": instruction}]
    original_rows = storage.get_interactions_by_ids(
        [item["interaction_id"] for item in json.loads(window["manifest"])]
    )
    assert original_rows[0].content == instruction
    assert original_rows[1].tools_used[0].tool_data["output"] == large_original
