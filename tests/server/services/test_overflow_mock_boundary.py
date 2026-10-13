"""Mock simulations cannot acknowledge or replace durable real extraction work."""

from unittest.mock import Mock

import pytest

from reflexio.lib._base import create_generation_litellm_client
from reflexio.models.config_schema import UserPlaybookExtractorConfig
from reflexio.server.llm.input_budget import InputBudgetConfigurationError
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.window_executor import WindowExecutor
from tests.server.services.test_durable_window_pipeline import (
    _overflow_coordinator,
    payload,
)
from tests.server.services.test_durable_window_pipeline import (
    pipeline as imported_pipeline,
)

pytestmark = pytest.mark.integration
mock_pipeline = imported_pipeline


def configure_kind(engine, kind):
    if kind == "playbook":
        configurator = engine.request_context.configurator
        configurator.set_config(
            configurator.get_config().model_copy(
                update={
                    "profile_extractor_config": None,
                    "user_playbook_extractor_config": UserPlaybookExtractorConfig(
                        extraction_definition_prompt="Extract agent improvements"
                    ),
                }
            )
        )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_new_mock_windows_complete_without_admitting_overflow(
    mock_pipeline, monkeypatch, kind
):
    engine, worker, client = mock_pipeline
    configure_kind(engine, kind)
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    dispatch = Mock(side_effect=AssertionError("mock simulation dispatched a provider"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", dispatch)
    request_id = f"mock-{kind}"
    response = client.post(
        "/api/publish_interaction", json=payload(request_id, force=True)
    )
    assert response.status_code == 200 and response.json()["success"]
    worker.drain_org("stream-test", 3, 300)
    assert (
        client.get("/api/learning_status", params={"request_id": request_id}).json()[
            "status"
        ]
        == "done"
    )
    storage = engine.get_storage()
    with storage._stream_sql() as db:
        query = (
            "SELECT COUNT(*) AS n FROM profiles"
            if kind == "profile"
            else "SELECT COUNT(*) AS n FROM user_playbooks"
        )
        assert db.query(query)[0]["n"] > 0
        assert db.query("SELECT id FROM _agent_runs WHERE user_id='u'") == []
    dispatch.assert_not_called()


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("legacy", [False, True])
def test_mock_toggle_refuses_existing_real_work_without_writes_or_quota_changes(
    mock_pipeline, monkeypatch, kind, legacy
):
    engine, _, _ = mock_pipeline
    configure_kind(engine, kind)
    monkeypatch.setenv("REFLEXIO_EXTRACTION_OVERFLOW_ENABLED", "true")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "true")
    storage, window, token = _overflow_coordinator(
        mock_pipeline,
        legacy=legacy,
        attempts=[{"id": "already-consumed", "usage": None, "model_provenance": None}],
    )
    parent_id = f"window:{window.window_id}"
    before = storage.get_agent_run(parent_id)
    assert before is not None
    executor = WindowExecutor(
        engine.request_context, create_generation_litellm_client(engine.request_context)
    )
    with pytest.raises(InputBudgetConfigurationError, match="Mock mode"):
        executor.execute(window, token)
    assert storage.get_agent_run(parent_id) == before
    with storage._stream_sql() as db:
        assert db.query("SELECT COUNT(*) AS n FROM profiles")[0]["n"] == 0
        assert db.query("SELECT COUNT(*) AS n FROM user_playbooks")[0]["n"] == 0
        row = db.query(
            "SELECT completed,outcome,effects FROM extraction_windows WHERE window_id=?",
            (window.window_id,),
        )[0]
        assert (
            not row["completed"] and row["outcome"] is None and row["effects"] is None
        )
