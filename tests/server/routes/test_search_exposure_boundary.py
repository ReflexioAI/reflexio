"""Search-route contract for synchronous user-playbook exposure recording."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.domain import UserPlaybook
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api import create_app
from reflexio.server.extensions import register_service
from reflexio.server.services.search_exposure import SEARCH_EXPOSURE_RECORDER


def _playbook(playbook_id: int, content: str) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=playbook_id,
        user_id="user-1",
        agent_version="agent-v1",
        request_id=f"source-{playbook_id}",
        playbook_name=f"Playbook {playbook_id}",
        created_at=1_700_000_000 + playbook_id,
        content=content,
        trigger=f"Trigger {playbook_id}",
        tags=["support"],
    )


@contextmanager
def _search_results(playbooks: list[UserPlaybook]) -> Iterator[MagicMock]:
    reflexio = MagicMock()
    reflexio.request_context.configurator.get_config.return_value = Config(
        storage_config=StorageConfigSQLite()
    )
    result = MagicMock(
        success=True,
        profiles=[],
        agent_playbooks=[],
        user_playbooks=playbooks,
        reformulated_query=None,
        msg="OK",
        agent_trace=None,
        rehydrated_text=None,
    )
    reflexio.unified_search.return_value = result
    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=reflexio,
    ):
        yield reflexio


def _client() -> TestClient:
    return TestClient(
        create_app(
            get_org_id=lambda: "org-1",
            get_caller_type=lambda: "production_agent",
        ),
        raise_server_exceptions=False,
    )


@dataclass
class _Recorder:
    batches: list[Any] = field(default_factory=list)
    completed: bool = False

    def record(self, batch: Any) -> None:
        self.batches.append(batch)
        self.completed = True


def test_unified_search_records_the_final_user_playbook_set_before_return() -> None:
    playbooks = [_playbook(11, "First"), _playbook(12, "Second")]
    recorder = _Recorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    with _search_results(playbooks):
        response = _client().post(
            "/api/search",
            json={
                "query": "answer",
                "user_id": "user-1",
                "request_id": "request-1",
                "session_id": "session-1",
                "interaction_id": 41,
            },
        )

    assert response.status_code == 200, response.text
    assert recorder.completed is True
    assert len(recorder.batches) == 1
    batch = recorder.batches[0]
    assert batch.org_id == "org-1"
    assert batch.request_id == "request-1"
    assert batch.session_id == "session-1"
    assert batch.interaction_id == 41
    assert batch.user_id == "user-1"
    assert batch.user_playbooks == tuple(playbooks)


def test_recorder_failure_prevents_a_successful_search_response() -> None:
    class _FailingRecorder:
        def record(self, _batch: Any) -> None:
            raise RuntimeError("ledger unavailable")

    register_service(SEARCH_EXPOSURE_RECORDER, _FailingRecorder())

    with _search_results([_playbook(11, "First")]):
        response = _client().post(
            "/api/search",
            json={
                "query": "answer",
                "user_id": "user-1",
                "request_id": "request-1",
            },
        )

    assert response.status_code == 500


def test_no_user_playbook_results_record_one_empty_synchronous_batch() -> None:
    recorder = _Recorder()
    register_service(SEARCH_EXPOSURE_RECORDER, recorder)

    with _search_results([]):
        response = _client().post(
            "/api/search",
            json={"query": "answer", "user_id": "user-1"},
        )

    assert response.status_code == 200, response.text
    assert recorder.completed is True
    assert len(recorder.batches) == 1
    assert recorder.batches[0].user_playbooks == ()


def test_oss_search_succeeds_when_no_recorder_is_registered() -> None:
    with _search_results([_playbook(11, "First")]):
        response = _client().post(
            "/api/search",
            json={"query": "answer", "user_id": "user-1"},
        )

    assert response.status_code == 200, response.text
    assert [item["user_playbook_id"] for item in response.json()["user_playbooks"]] == [
        11
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 101),
        ("request_id", "r" * 256),
        ("session_id", "s" * 256),
        ("user_id", "u" * 256),
    ],
)
def test_unified_search_rejects_oversized_work_before_search_execution(
    field: str,
    value: object,
) -> None:
    with _search_results([]) as reflexio:
        response = _client().post(
            "/api/search",
            json={"query": "answer", field: value},
        )

    assert response.status_code == 422
    reflexio.unified_search.assert_not_called()


def test_unified_search_accepts_exact_workload_and_identifier_limits() -> None:
    with _search_results([]) as reflexio:
        response = _client().post(
            "/api/search",
            json={
                "query": "answer",
                "top_k": 100,
                "request_id": "r" * 255,
                "session_id": "s" * 255,
                "user_id": "u" * 255,
            },
        )

    assert response.status_code == 200, response.text
    reflexio.unified_search.assert_called_once()
