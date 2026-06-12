from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from dotenv import dotenv_values

from reflexio import InteractionData, ReflexioClient
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.models.config_schema import (
    Config,
    PendingToolCallConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.extraction.resumable_agent import (
    FINISH_EXTRACTION_TOOL_NAME,
)
from reflexio.server.services.extraction.resume_worker import ExtractionResumeWorker
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
    PendingToolCallRecord,
    PendingToolCallStatus,
    RunToolDependencyRecord,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
    human_feedback_scope,
)

pytestmark = pytest.mark.e2e

_LIVE_E2E_TIMEOUT_SECONDS = 240
_LIVE_E2E_POLL_SECONDS = 5
_LIVE_E2E_USER_AGENT = "reflexio-live-resumable-e2e"


def _request_context(storage: SQLiteStorage) -> RequestContext:
    ctx = RequestContext.__new__(RequestContext)
    ctx.org_id = "e2e_resumable_org"
    ctx.storage = storage
    ctx.storage_base_dir = None
    ctx.configurator = MagicMock()
    ctx.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite(),
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="default_profile_extractor",
            extraction_definition_prompt="Extract durable deployment preferences.",
        ),
        pending_tool_call_config=PendingToolCallConfig(
            enabled=True,
        ),
    )
    ctx.configurator.get_agent_context.return_value = "Test agent context"
    ctx.prompt_manager = MagicMock()
    ctx.prompt_manager.render_prompt.side_effect = lambda prompt_id, variables: (
        f"{prompt_id}: {variables}"
    )
    return ctx


def _seed_interactions(storage: SQLiteStorage) -> None:
    storage.add_request(
        Request(
            request_id="request_1",
            user_id="user_1",
            created_at=1_000,
            source="api",
            agent_version="v1",
            session_id="request_1",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=1,
            user_id="user_1",
            request_id="request_1",
            created_at=1_000,
            role="user",
            content="Please remember our deployment target once confirmed.",
        )
    )
    storage._insert_interaction(
        Interaction(
            interaction_id=2,
            user_id="user_1",
            request_id="request_1",
            created_at=1_001,
            role="assistant",
            content="I will ask for the deployment target and continue.",
        )
    )


def _seed_followup_ready_run(storage: SQLiteStorage) -> None:
    storage.create_agent_run(
        AgentRunRecord(
            id="run_1",
            binding=AgentBinding(
                org_id="e2e_resumable_org",
                extractor_kind="profile",
                user_id="user_1",
                request_id="request_1",
                agent_version="v1",
                source="api",
                source_interaction_ids=[1, 2],
                window_start_interaction_id=1,
                window_end_interaction_id=2,
                extractor_config_hash="old_hash",
            ),
            status=AgentRunStatus.FINALIZED_PENDING_TOOL,
            generation_request_snapshot={"request_id": "request_1"},
            max_steps_remaining=7,
        )
    )
    now = datetime(2026, 5, 28, tzinfo=UTC)
    question = "What deployment target should be treated as canonical?"
    scope = human_feedback_scope("e2e_resumable_org")
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id="ptc_1",
            org_id="e2e_resumable_org",
            user_id="user_1",
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human",
                question_text=question,
            ),
            status=PendingToolCallStatus.PENDING,
            question_text=question,
            args={"question": question},
            expires_at=now + timedelta(hours=1),
            cache_until=now + timedelta(minutes=5),
        )
    )
    storage.attach_run_tool_dependency(
        RunToolDependencyRecord(run_id="run_1", pending_tool_call_id="ptc_1")
    )
    storage.resolve_pending_tool_call(
        "ptc_1",
        result={"answer": "Use AWS ECS."},
        resolved_at=now,
        valid_for_seconds=3600,
    )


def _load_live_e2e_settings() -> tuple[str, str]:
    if os.environ.get("RUN_LIVE_RESUMABLE_E2E") != "true":
        pytest.skip("Set RUN_LIVE_RESUMABLE_E2E=true to run live resumable E2E")
    if os.environ.get("MOCK_LLM_RESPONSE", "").lower() == "true":
        pytest.skip("Live resumable E2E requires MOCK_LLM_RESPONSE=false")

    dotenv_values_from_file: dict[str, str | None] = {}
    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in (start, *start.parents):
            env_path = parent / ".env"
            if not env_path.exists():
                continue
            candidate = dotenv_values(env_path)
            if candidate.get("REFLEXIO_API_KEY"):
                dotenv_values_from_file = dict(candidate)
                break
        if dotenv_values_from_file:
            break

    api_key = os.environ.get("REFLEXIO_API_KEY") or dotenv_values_from_file.get(
        "REFLEXIO_API_KEY"
    )
    base_url = (
        os.environ.get("REFLEXIO_URL")
        or dotenv_values_from_file.get("REFLEXIO_URL")
        or "http://localhost:8081"
    )

    if not api_key:
        pytest.skip("Live resumable E2E requires REFLEXIO_API_KEY")
    return str(base_url).rstrip("/"), str(api_key)


def _live_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": _LIVE_E2E_USER_AGENT,
    }


def _api_request(
    method: str,
    base_url: str,
    path: str,
    headers: dict[str, str],
    **kwargs: Any,
) -> Any:
    response = requests.request(
        method,
        f"{base_url}{path}",
        headers=headers,
        timeout=60,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def _poll_until(
    label: str,
    predicate: Callable[[], Any],
    *,
    timeout_seconds: int = _LIVE_E2E_TIMEOUT_SECONDS,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    last_value: Any = None
    while time.monotonic() < deadline:
        last_value = predicate()
        if last_value:
            return last_value
        time.sleep(_LIVE_E2E_POLL_SECONDS)
    raise AssertionError(f"Timed out waiting for {label}; last value: {last_value!r}")


def _config_restore_patch(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "profile_extractor_config",
            "user_playbook_extractor_config",
            "pending_tool_call_config",
            "skip_should_run_check",
            "window_size",
            "stride_size",
            "extraction_backend",
        )
        if key in config
    }


def _apply_live_config_patch(
    base_url: str,
    headers: dict[str, str],
    patch_payload: dict[str, Any],
    *,
    expected_extractor_name: str | None = None,
) -> dict[str, Any]:
    _api_request("POST", base_url, "/api/update_config", headers, json=patch_payload)
    _api_request("POST", base_url, "/api/admin/cache/invalidate", headers, json={})
    config = _api_request("GET", base_url, "/api/get_config", headers)
    if expected_extractor_name is not None:
        profile_config = config.get("profile_extractor_config") or {}
        assert profile_config.get("extractor_name") == expected_extractor_name
    return config


def _live_resumable_config(marker: str) -> tuple[dict[str, Any], str, str]:
    extractor_name = "resumable_human_question_live_e2e"
    question_text = (
        f"For live resumable test {marker}, what deployment target should be "
        "treated as canonical?"
    )
    profile_prefix = f"Live resumable test {marker} canonical deployment target:"
    return (
        {
            "profile_extractor_config": {
                "extractor_name": extractor_name,
                "extraction_definition_prompt": (
                    f"This is an end-to-end test for marker {marker}. "
                    "If the session mentions this marker but does not explicitly "
                    "provide the canonical deployment target, call ask_human before "
                    "extracting the deployment-target profile. Use this exact "
                    f"question text: {question_text!r}. Use answer_format "
                    f"'short text' and include the tag {marker!r}. After a human "
                    "answer is available, you must call finish_extraction with "
                    "exactly one profile. Never call finish_extraction with "
                    "profiles null after a resolved answer is present. The profile "
                    f"content must be exactly {profile_prefix!r} followed by one "
                    "space and the human-provided answer. Use time_to_live "
                    "'one_year'. Your first action for this marker must be the "
                    "ask_human tool call. Do not infer or invent the target."
                ),
                "context_prompt": "Live E2E test for resumable extraction.",
            },
            "user_playbook_extractor_config": None,
            "pending_tool_call_config": {
                "enabled": True,
                "pending_ttl_seconds": 3600,
                "dedup_cache_seconds": 60,
                "prior_answer_valid_seconds": 3600,
                "resume_poll_interval_seconds": 1.0,
                "tool_overrides": {
                    "ask_human": {
                        "pending_ttl_seconds": 3600,
                        "dedup_cache_seconds": 60,
                        "prior_answer_valid_seconds": 3600,
                    }
                },
            },
            "skip_should_run_check": True,
            "window_size": 4,
            "stride_size": 4,
            "extraction_backend": "classic",
        },
        question_text,
        profile_prefix,
    )


def _find_pending_question(
    base_url: str,
    headers: dict[str, str],
    *,
    marker: str,
    status: str,
) -> dict[str, Any] | None:
    payload = _api_request(
        "GET",
        base_url,
        f"/api/pending_tool_calls?status={status}&limit=100",
        headers,
    )
    for pending_call in payload["pending_tool_calls"]:
        question = pending_call.get("question_text") or ""
        tags = pending_call.get("tags") or []
        result = pending_call.get("result") or {}
        answer = result.get("answer") or ""
        if marker in question or marker in tags or marker in answer:
            return pending_call
    return None


def _profile_content_with(
    client: ReflexioClient,
    *,
    user_id: str,
    required_parts: tuple[str, ...],
) -> str | None:
    profiles = client.get_profiles(user_id=user_id, force_refresh=True).user_profiles
    for profile in profiles:
        content = profile.content
        if all(part in content for part in required_parts):
            return content
    return None


def test_resumable_extraction_resumes_after_human_answer(
    monkeypatch,
    tool_call_completion,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        storage = SQLiteStorage(
            org_id="e2e_resumable_org",
            db_path=f"{temp_dir}/reflexio.db",
        )
        _seed_interactions(storage)
        _seed_followup_ready_run(storage)
        worker = ExtractionResumeWorker(
            request_context=_request_context(storage),
            llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
        )
        make_tc, _ = tool_call_completion
        response = make_tc(
            FINISH_EXTRACTION_TOOL_NAME,
            {
                "profiles": [
                    {
                        "content": "User deployment target is AWS ECS.",
                        "time_to_live": "infinity",
                    }
                ]
            },
        )

        with (
            patch("litellm.completion", side_effect=[response]),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=False,
            ),
        ):
            resumed = worker.drain(max_runs=1)

        run = storage.get_agent_run("run_1")
        assert resumed == 1
        assert run is not None
        assert run.status == AgentRunStatus.FINALIZED
        assert run.max_steps_remaining == 6
        assert storage.list_run_tool_dependencies("run_1")[0].consumed_at is not None
        assert [profile.content for profile in storage.get_user_profile("user_1")] == [
            "User deployment target is AWS ECS."
        ]


@pytest.mark.requires_credentials
@pytest.mark.timeout(360)
def test_live_resumable_question_resolve_and_edit_roundtrip():
    """Exercise the resumable agent through the live API with real LLM calls.

    This test intentionally avoids patching litellm. It requires a running
    Reflexio backend, a valid API key, and live provider credentials in that
    backend process.
    """
    base_url, api_key = _load_live_e2e_settings()
    headers = _live_headers(api_key)
    client = ReflexioClient(api_key=api_key, url_endpoint=base_url, timeout=300)
    client.session.headers.update({"User-Agent": _LIVE_E2E_USER_AGENT})

    marker = f"RESUMABLE_LIVE_E2E_{uuid.uuid4().hex[:8]}"
    user_id = f"resumable_live_e2e_user_{uuid.uuid4().hex[:8]}"
    session_id = f"resumable-live-e2e-{uuid.uuid4().hex[:8]}"
    first_answer = "AWS ECS"
    edited_answer = "Google Cloud Run"
    config_patch, question_text, profile_prefix = _live_resumable_config(marker)

    original_config = _api_request("GET", base_url, "/api/get_config", headers)
    restore_patch = _config_restore_patch(original_config)

    try:
        _apply_live_config_patch(
            base_url,
            headers,
            config_patch,
            expected_extractor_name="resumable_human_question_live_e2e",
        )

        client.publish_interaction(
            user_id=user_id,
            session_id=session_id,
            source="pytest-live-resumable-e2e",
            wait_for_response=True,
            force_extraction=True,
            interactions=[
                InteractionData(
                    role="user",
                    content=(
                        f"Remember live resumable marker {marker}. "
                        "The canonical deployment target is unknown and not "
                        "available anywhere in this transcript. The only valid "
                        "next step is to ask the configured human follow-up "
                        "question before extracting any profile. Once a human "
                        "answer exists, the durable profile should contain exactly "
                        f"this prefix: {profile_prefix}"
                    ),
                ),
                InteractionData(
                    role="assistant",
                    content=(
                        "I cannot infer the deployment target. I need a human "
                        "answer before storing any deployment-target profile."
                    ),
                ),
            ],
        )

        pending_call = _poll_until(
            "live ask_human pending question",
            lambda: _find_pending_question(
                base_url,
                headers,
                marker=marker,
                status="pending",
            ),
        )
        assert pending_call["tool_name"] == "ask_human"
        assert pending_call["status"] == "pending"
        assert pending_call["question_text"] == question_text
        assert pending_call["user_id"] == user_id

        pending_call_id = pending_call["id"]
        resolved_call = _api_request(
            "POST",
            base_url,
            f"/api/pending_tool_calls/{pending_call_id}/resolve",
            headers,
            json={"result": {"answer": first_answer}, "valid_for_seconds": 3600},
        )
        assert resolved_call["status"] == "resolved"
        assert resolved_call["result"]["answer"] == first_answer

        first_profile = _poll_until(
            "live profile generated from resolved human answer",
            lambda: _profile_content_with(
                client,
                user_id=user_id,
                required_parts=(profile_prefix, first_answer),
            ),
        )
        assert marker in first_profile

        edited_call = _api_request(
            "PATCH",
            base_url,
            f"/api/pending_tool_calls/{pending_call_id}/answer",
            headers,
            json={"answer": edited_answer, "valid_for_seconds": 3600},
        )
        assert edited_call["status"] == "resolved"
        assert edited_call["result"]["answer"] == edited_answer
        assert edited_call["result"].get("not_applicable") is not True

        edited_profile = _poll_until(
            "live profile regenerated from edited human answer",
            lambda: _profile_content_with(
                client,
                user_id=user_id,
                required_parts=(profile_prefix, edited_answer),
            ),
        )
        assert marker in edited_profile

        latest_resolved_call = _find_pending_question(
            base_url,
            headers,
            marker=marker,
            status="resolved",
        )
        assert latest_resolved_call is not None
        assert latest_resolved_call["result"]["answer"] == edited_answer
        assert latest_resolved_call["result"].get("not_applicable") is not True
    finally:
        _apply_live_config_patch(base_url, headers, restore_patch)
