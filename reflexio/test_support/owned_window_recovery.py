# ruff: noqa: S101,S608 -- Test-only assertions; SQL names are fixed storage-adapter registry entries.
"""Shared cross-backend crash/ownership regression, with physical inference stubbed."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.user_lease import (
    adopt_agent_run_for_execution,
    owned_user_work_context,
)
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.storage.storage_base import AgentRunStatus, BaseStorage


def recovery_publish_request() -> dict[str, Any]:
    """The identical admitted source for SQLite and native recovery cells."""
    return {
        "user_id": "u",
        "request_id": "crashed-window",
        "session_id": "s",
        "source": "test",
        "force_extraction": True,
        "interaction_data_list": [
            {
                "role": "User",
                "content": f"I prefer concise answers with details {index}",
                "created_at": 100 - index,
            }
            for index in range(10)
        ],
    }


class SimulatedProcessDeath(BaseException):
    """Bypass application Exception handling, while unwinding owned test resources."""


def install_crashing_model(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    monkeypatch.setenv(
        "REFLEXIO_LLM_CONTEXT_LIMITS_JSON",
        '{"minimax/MiniMax-M3":1000000,"gpt-4o":65536}',
    )
    monkeypatch.setattr(
        "reflexio.server.llm.input_budget.litellm.token_counter", lambda **_: 100
    )
    calls: list[int] = []

    def respond(params: dict[str, Any], hard_timeout: float) -> SimpleNamespace:
        del hard_timeout
        calls.append(1)
        if len(calls) == 1:
            raise SimulatedProcessDeath()
        schema = params.get("response_format")
        schema_name = getattr(schema, "__name__", "") or json.dumps(schema, default=str)
        content = {
            "profiles": [
                {
                    "content": "The user prefers concise answers.",
                    "time_to_live": "infinity",
                    "source_span": "I prefer concise answers with details 0",
                }
            ]
        }
        if "OperationOutput" in schema_name:
            # This provider authors the current wire from the actual visible
            # source book; crash recovery must exercise the real compiler.
            for message in params["messages"]:
                text = message.get("content", "")
                marker = '\n\n{"original_window":'
                start = text.find(marker) if isinstance(text, str) else -1
                if start < 0:
                    continue
                payload, _ = json.JSONDecoder().raw_decode(text[start + 2 :])
                groups = []
                excerpts = [
                    excerpt
                    for session in payload["original_window"]
                    for interaction in session["interactions"]
                    if interaction["role"] == "user"
                    for excerpt in interaction["content"]["excerpts"]
                ]
                for proposal in payload["NEW"]:
                    quote = proposal["source_span"]
                    evidence = next(e for e in excerpts if quote in e["text"])
                    groups.append(
                        {
                            "new_ids": [proposal["id"]],
                            "writes": [
                                {
                                    "clauses": [
                                        {
                                            "action": "assert",
                                            "source_new_id": proposal["id"],
                                            "content": quote,
                                            "evidence": [
                                                {
                                                    "excerpt_id": evidence[
                                                        "excerpt_id"
                                                    ],
                                                    "stance": "assertion",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            ],
                            "forget": [],
                        }
                    )
                content = {"snapshot": payload["snapshot"], "groups": groups}
                break
            else:
                raise AssertionError(
                    "Current operation prompt must expose its source book"
                )
        elif "ProfileDeduplicationOutput" in schema_name:
            raise AssertionError("Retired profile actor must not be requested")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(content), tool_calls=None, refusal=None
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
        LiteLLMClient, "_completion_with_hard_timeout", Mock(side_effect=respond)
    )
    return calls


def assert_running_window_recovery(
    storage: BaseStorage,
    worker: DurableLearningWorker,
    calls: list[int],
    monkeypatch: pytest.MonkeyPatch,
    parent_state: str,
    expire_parent: Callable[[BaseStorage, pytest.MonkeyPatch], None],
) -> None:
    with pytest.raises(SimulatedProcessDeath):
        worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    with storage._stream_sql(read_only=True) as db:
        rows = db.query(
            f"SELECT id FROM {db.table('_agent_runs')} WHERE org_id=? AND user_id=? LIMIT 2",
            (storage.org_id, "u"),
        )
    assert len(rows) == 1
    original = storage.get_agent_run(rows[0]["id"])
    assert original is not None
    assert original.status == AgentRunStatus.RUNNING
    assert original.committed_output is None and calls == [1]
    attempts = copy.deepcopy(
        original.generation_request_snapshot["overflow"]["attempts"]
    )
    assert len(attempts) == 1 and attempts[0]["usage"] is None
    if parent_state != "released":
        lease_seconds = 1 if parent_state == "expired" else 300
        token = storage.claim_user_extraction("u", "test-real-parent", lease_seconds)
        assert token is not None
        with owned_user_work_context(storage, "u", token, lease_seconds):
            adopt_agent_run_for_execution(storage, original)
        if parent_state == "expired":
            expire_parent(storage, monkeypatch)
        else:
            if parent_state == "replacement":
                storage.release_user_extraction("u", token)
                token = storage.claim_user_extraction("u", "test-replacement", 300)
                assert token is not None
            before = storage.get_agent_run(original.id)
            worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
            assert calls == [1] and storage.get_agent_run(original.id) == before
            storage.fence_user_extraction("u", token)
            storage.release_user_extraction("u", token)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    recovered = storage.get_agent_run(original.id)
    assert recovered is not None
    assert recovered.status == AgentRunStatus.FINALIZED
    assert (
        recovered.generation_request_snapshot["overflow"]["attempts"][0] == attempts[0]
    )
    assert storage.extraction_status("u", "crashed-window")["status"] == "done"
    assert storage.count_all_profiles() == 1
    manifest = recovered.generation_request_snapshot["overflow"]
    prior = original.generation_request_snapshot["overflow"]
    assert manifest["attempt_limit"] == prior["attempt_limit"]
    assert manifest["source_digest"] == prior["source_digest"]
    assert manifest["chunk_digests"] == prior["chunk_digests"]
    assert len(manifest["attempts"]) > len(attempts)
    completed_calls = list(calls)
    worker.drain_org(storage.org_id, batch_size=1, lease_seconds=300)
    assert calls == completed_calls and storage.count_all_profiles() == 1
