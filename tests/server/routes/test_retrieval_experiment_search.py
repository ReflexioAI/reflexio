from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.retriever_schema import UnifiedSearchResponse
from reflexio.models.config_schema import (
    Config,
    RetrievalExperimentConfig,
    RetrievalExperimentRecord,
    StorageConfigSQLite,
)
from reflexio.server.api import create_app
from reflexio.server.services.retrieval_experiment import (
    assign_retrieval_experiment_arm,
)


def _holdout_user() -> str:
    for index in range(1_000):
        user_id = f"holdout-{index}"
        if (
            assign_retrieval_experiment_arm(
                org_id="org-1",
                experiment_id="exp-1",
                user_id=user_id,
                holdout_percentage=10,
            )
            == "holdout"
        ):
            return user_id
    raise AssertionError("could not find a deterministic holdout user")


def _treatment_user() -> str:
    for index in range(1_000):
        user_id = f"treatment-{index}"
        if (
            assign_retrieval_experiment_arm(
                org_id="org-1",
                experiment_id="exp-1",
                user_id=user_id,
                holdout_percentage=10,
            )
            == "treatment"
        ):
            return user_id
    raise AssertionError("could not find a deterministic treatment user")


@pytest.mark.parametrize(
    ("path", "payload", "result_field"),
    [
        ("/api/search", {"query": "answer", "user_id": "USER"}, "profiles"),
        (
            "/api/search_profiles",
            {"query": "answer", "user_id": "USER"},
            "user_profiles",
        ),
        (
            "/api/search_user_playbooks",
            {"query": "answer", "user_id": "USER"},
            "user_playbooks",
        ),
        (
            "/api/search_agent_playbooks",
            {"query": "answer", "user_id": "USER"},
            "agent_playbooks",
        ),
    ],
)
def test_holdout_returns_empty_contract_without_running_retrieval(
    path: str, payload: dict[str, str], result_field: str
) -> None:
    record = RetrievalExperimentRecord(
        experiment_id="exp-1", holdout_percentage=10, started_at=100
    )
    config = Config(
        storage_config=StorageConfigSQLite(),
        retrieval_experiment_config=RetrievalExperimentConfig(
            experiment_id="exp-1", holdout_percentage=10
        ),
        retrieval_experiment_history=[record],
    )
    fake_reflexio = MagicMock()
    fake_reflexio.request_context.configurator.get_config.return_value = config
    user_id = _holdout_user()
    request_payload = {
        key: (user_id if value == "USER" else value) for key, value in payload.items()
    }
    app = create_app(
        get_org_id=lambda: "org-1",
        get_caller_type=lambda: "internal",
    )

    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=fake_reflexio,
    ):
        response = TestClient(app).post(path, json=request_payload)

    assert response.status_code == 200
    body = response.json()
    assert body[result_field] == []
    assert body["experiment"] == {"experiment_id": "exp-1", "arm": "holdout"}
    fake_reflexio.unified_search.assert_not_called()
    fake_reflexio.search_user_profiles.assert_not_called()
    fake_reflexio.search_user_playbooks.assert_not_called()
    fake_reflexio.search_agent_playbooks.assert_not_called()


def test_active_experiment_rejects_missing_user_id() -> None:
    config = Config(
        storage_config=StorageConfigSQLite(),
        retrieval_experiment_config=RetrievalExperimentConfig(
            experiment_id="exp-1", holdout_percentage=10
        ),
    )
    fake_reflexio = SimpleNamespace(
        request_context=SimpleNamespace(
            configurator=SimpleNamespace(get_config=lambda: config)
        )
    )
    app = create_app(
        get_org_id=lambda: "org-1",
        get_caller_type=lambda: "internal",
    )

    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=fake_reflexio,
    ):
        response = TestClient(app).post(
            "/api/search_agent_playbooks", json={"query": "answer"}
        )

    assert response.status_code == 422
    assert "user_id is required" in response.json()["detail"]


@pytest.mark.parametrize(
    ("caller_type", "payload", "expected_experiment"),
    [
        (
            "internal",
            {"query": "answer", "user_id": _treatment_user()},
            {"experiment_id": "exp-1", "arm": "treatment"},
        ),
        ("dashboard", {"query": "answer"}, None),
    ],
)
def test_treatment_retrieves_and_dashboard_bypasses_assignment(
    caller_type: str,
    payload: dict[str, str],
    expected_experiment: dict[str, str] | None,
) -> None:
    config = Config(
        storage_config=StorageConfigSQLite(),
        retrieval_experiment_config=RetrievalExperimentConfig(
            experiment_id="exp-1", holdout_percentage=10
        ),
    )
    fake_reflexio = MagicMock()
    fake_reflexio.request_context.configurator.get_config.return_value = config
    fake_reflexio.unified_search.return_value = UnifiedSearchResponse(success=True)
    app = create_app(
        get_org_id=lambda: "org-1",
        get_caller_type=lambda: caller_type,
    )

    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=fake_reflexio,
    ):
        response = TestClient(app).post("/api/search", json=payload)

    assert response.status_code == 200, response.text
    assert response.json().get("experiment") == expected_experiment
    fake_reflexio.unified_search.assert_called_once()
