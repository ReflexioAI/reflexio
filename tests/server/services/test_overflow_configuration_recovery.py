"""Corrected declarations resume frozen extraction without resetting its quota."""

import json
import math
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import HttpUrl

from reflexio.models.config_schema import APIKeyConfig, CustomEndpointConfig
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from tests.server.services.test_durable_window_pipeline import payload
from tests.server.services.test_durable_window_pipeline import pipeline as pipeline


@pytest.mark.parametrize("mode", ["recover", "policy_changed", "capacity"])
def test_corrected_context_declaration_resumes_only_the_original_contract(
    pipeline, monkeypatch, mode
):
    engine, worker, http = pipeline
    storage = engine.get_storage()
    with storage._stream_sql(read_only=True) as db:
        started = db.now()
    elapsed = [0.0]
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: started + elapsed[0])
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    endpoint = "https://custom.invalid/v1"
    model = "openai/gpt-4o"
    declaration_key = f"{model}@{endpoint}"
    # This test stubs every provider dispatch. Keep endpoint safety validation
    # active while making its public DNS answer deterministic and network-free.
    monkeypatch.setattr(
        "reflexio.models.api_schema.validators.socket.getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))
        ],
    )
    keys = APIKeyConfig(
        custom_endpoint=CustomEndpointConfig(
            model=model,
            api_key="test-key-no-provider",
            api_base=HttpUrl(endpoint),
        )
    )
    configurator = engine.request_context.configurator
    configurator.set_config(
        configurator.get_config().model_copy(update={"api_key_config": keys})
    )
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        json.dumps({declaration_key: 512} if mode == "capacity" else {}),
    )
    # A custom endpoint must supply its output cap explicitly; this test isolates
    # correction of context declarations, not the separate missing-output gate.
    monkeypatch.setattr(
        "reflexio.lib._base.create_generation_litellm_client",
        lambda _: LiteLLMClient(
            LiteLLMConfig(
                model=model,
                max_tokens=1024,
                timeout=5,
                fallback_models=[],
                api_key_config=keys,
            )
        ),
    )
    monkeypatch.setattr(
        storage, "_get_embedding", lambda *_args, **_kwargs: [0.1] * 512
    )
    monkeypatch.setattr(storage, "_expand_document", lambda *_args, **_kwargs: None)
    dispatched = []

    def respond(params, hard_timeout):
        dispatched.append(params)
        name = getattr(params.get("response_format"), "__name__", "") or json.dumps(
            params.get("response_format"), default=str
        )
        if "ProfileDeduplicationOutput" in name or any(
            "unique_ids" in str(message.get("content", ""))
            for message in params["messages"]
        ):
            result = {"duplicate_groups": [], "unique_ids": ["NEW-0"], "deletions": []}
        else:
            result = {
                "profiles": [
                    {
                        "content": "The user prefers concise answers.",
                        "time_to_live": "infinity",
                        "source_span": "I prefer concise answers with details 0",
                    }
                ]
            }
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
            model=model,
            _hidden_params={"custom_llm_provider": "openai"},
        )

    monkeypatch.setattr(
        LiteLLMClient, "_completion_with_hard_timeout", staticmethod(respond)
    )
    complete = Mock(wraps=storage.complete_extraction)
    monkeypatch.setattr(storage, "complete_extraction", complete)
    assert http.post(
        "/api/publish_interaction", json=payload("declaration-recovery", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert dispatched == [] and storage.count_all_profiles() == 0
    reason = (
        "input_budget_exceeded"
        if mode == "capacity"
        else "input_budget_configuration_error"
    )
    assert storage.extraction_status("u", "declaration-recovery") == {
        "status": "failed",
        "reason": reason,
    }
    with storage._stream_sql(read_only=True) as db:
        cursor = db.query(
            "SELECT window_id,retry_at,attempts FROM extraction_cursors WHERE user_id=? AND kind=?",
            ("u", "profile"),
        )[0]
    assert cursor["attempts"] == 1
    assert (
        math.isinf(cursor["retry_at"])
        if mode == "capacity"
        else cursor["retry_at"] == started + 1
    )
    run_id = f"window:{cursor['window_id']}"
    parent = storage.get_agent_run(run_id)
    manifest = parent.generation_request_snapshot["overflow"]
    assert parent.last_error == reason and manifest["attempts"] == []
    # Fixing an operator declaration never rewrites source/prompt/quota state.
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON", json.dumps({declaration_key: 65536})
    )
    if mode == "policy_changed":
        render = engine.request_context.prompt_manager.render_prompt
        monkeypatch.setattr(
            engine.request_context.prompt_manager,
            "render_prompt",
            lambda *args, **kwargs: (
                render(*args, **kwargs) + "\nChanged extraction policy"
            ),
        )
    assert http.post(
        "/api/publish_interaction", json=payload("new-tail", count=5)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert dispatched == []
    assert (
        storage.get_agent_run(run_id).generation_request_snapshot["overflow"]
        == manifest
    )
    elapsed[0] = 2.0
    result = worker.drain_org(storage.org_id, 1, 300)
    after = storage.get_agent_run(run_id)
    frozen = after.generation_request_snapshot["overflow"]
    assert frozen["source_digest"] == manifest["source_digest"]
    assert frozen["chunk_digests"] == manifest["chunk_digests"]
    assert frozen["attempt_limit"] == manifest["attempt_limit"] == 128
    if mode == "recover":
        assert result == 1 and storage.count_all_profiles() == 1
        assert complete.call_count == 1
        assert len(dispatched) == len(frozen["attempts"]) == 2
        assert (
            storage.extraction_status("u", "declaration-recovery")["status"] == "done"
        )
        with storage._stream_sql(read_only=True) as db:
            windows = db.query(
                "SELECT window_id,completed FROM extraction_windows WHERE user_id=? AND kind=?",
                ("u", "profile"),
            )
            progress = db.query(
                "SELECT completed_seq,attempts FROM extraction_cursors WHERE user_id=? AND kind=?",
                ("u", "profile"),
            )[0]
        assert windows == [{"window_id": cursor["window_id"], "completed": 1}]
        assert progress["completed_seq"] == 10 and progress["attempts"] == 0
    else:
        assert result == 0 and dispatched == [] and frozen["attempts"] == []
        assert storage.count_all_profiles() == 0
        complete.assert_not_called()
        assert (
            storage.extraction_status("u", "declaration-recovery")["status"] == "failed"
        )
        with storage._stream_sql(read_only=True) as db:
            held = db.query(
                "SELECT retry_at,attempts FROM extraction_cursors WHERE user_id=? AND kind=?",
                ("u", "profile"),
            )[0]
        assert (
            math.isinf(held["retry_at"])
            if mode == "capacity"
            else held["retry_at"] == started + 4
        )
        assert held["attempts"] == (1 if mode == "capacity" else 2)
