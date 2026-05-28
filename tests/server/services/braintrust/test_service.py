"""Service-level tests for the Braintrust connector orchestrator.

Uses an in-memory fake storage and a stub client to exercise the full
connect → select_projects → sync → status → disconnect flow without HTTP.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    ConnectBraintrustRequest,
    ImportedScore,
    SelectProjectsRequest,
)
from reflexio.server.services.braintrust import _encryption
from reflexio.server.services.braintrust.client import (
    BraintrustAuthError,
    BraintrustHTTPError,
)
from reflexio.server.services.braintrust.service import (
    BraintrustConnectorService,
    _scores_from_spans,
)


class _InMemoryStorage:
    """Tiny in-memory replacement for BaseStorage with just the methods we use."""

    def __init__(self) -> None:
        self.connections: dict[str, BraintrustConnection] = {}
        self.scores: list[ImportedScore] = []

    def save_braintrust_connection(self, connection: BraintrustConnection) -> None:
        self.connections[connection.org_id] = connection

    def get_braintrust_connection(self, org_id: str) -> BraintrustConnection | None:
        return self.connections.get(org_id)

    def delete_braintrust_connection(self, org_id: str) -> None:
        self.connections.pop(org_id, None)

    def save_imported_scores(self, scores: list[ImportedScore]) -> None:
        self.scores.extend(scores)


def _stub_client_factory(*, valid: bool = True, payload: dict[str, Any] | None = None):
    """Build a `client_factory` callable returning a configured MagicMock client.

    `payload` lets a test override per-method results. Defaults model a tiny
    workspace with one project containing one experiment + two spans.
    """
    payload = payload or {}
    orgs = payload.get("organizations", [{"id": "ws_1", "name": "My Workspace"}])
    projects = payload.get("projects", [{"id": "p_1", "name": "Production"}])
    experiments = payload.get(
        "experiments", [{"id": "exp_1", "name": "v1", "created_at": 1700000000}]
    )
    spans = payload.get(
        "spans",
        [
            {
                "id": "span_a",
                "created_at": 1700000010,
                "metadata": {"reflexio_session_id": "sess_42"},
                "scores": {"hallucination": 0.05, "factuality": 0.95},
            },
            {
                "id": "span_b",
                "created_at": 1700000020,
                "metadata": {},
                "scores": {"hallucination": 0.30},
            },
        ],
    )

    def make_client(api_key: str) -> MagicMock:  # noqa: ARG001
        client = MagicMock()
        if valid:
            client.validate_key.return_value = True
        else:
            client.validate_key.return_value = False
        client.list_organizations.return_value = orgs
        client.list_projects.return_value = projects
        client.list_experiments.return_value = experiments
        client.list_spans.return_value = spans
        return client

    return make_client


def test_connect_returns_workspaces_for_valid_key() -> None:
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    response = svc.connect(ConnectBraintrustRequest(api_key="sk-valid"))
    assert response.success is True
    assert len(response.workspaces) == 1
    assert response.workspaces[0].workspace_id == "ws_1"
    assert response.workspaces[0].projects[0].project_id == "p_1"
    # Nothing persisted yet
    assert storage.connections == {}


def test_connect_closes_client() -> None:
    storage = _InMemoryStorage()
    client = _stub_client_factory()("sk-valid")
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=lambda _key: client
    )
    response = svc.connect(ConnectBraintrustRequest(api_key="sk-valid"))
    assert response.success is True
    client.close.assert_called_once()


def test_connect_returns_failure_for_invalid_key() -> None:
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage,
        org_id="org_t",
        client_factory=_stub_client_factory(valid=False),
    )
    response = svc.connect(ConnectBraintrustRequest(api_key="sk-bad"))
    assert response.success is False
    assert "rejected" in response.msg.lower()


def test_connect_returns_failure_on_auth_error_during_org_listing() -> None:
    """If list_organizations raises BraintrustAuthError, fail gracefully."""
    storage = _InMemoryStorage()

    def factory(_key: str) -> MagicMock:
        c = MagicMock()
        c.validate_key.return_value = True
        c.list_organizations.side_effect = BraintrustAuthError("nope")
        return c

    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=factory
    )
    response = svc.connect(ConnectBraintrustRequest(api_key="sk"))
    assert response.success is False


def test_select_projects_persists_encrypted_connection(monkeypatch) -> None:
    """select_projects writes a BraintrustConnection; API key is encrypted."""
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()

    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    response = svc.select_projects(
        SelectProjectsRequest(
            api_key="sk-customer",
            workspace_id="ws_1",
            workspace_name="My Workspace",
            project_ids=["p_1"],
        )
    )
    assert response.success is True
    stored = storage.connections["org_t"]
    # No Fernet key configured → passthrough; the stored value equals the input
    assert stored.api_key_enc == "sk-customer"
    assert stored.project_ids == ["p_1"]
    assert stored.workspace_name == "My Workspace"


def test_status_reports_disconnected_when_no_row() -> None:
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    s = svc.status()
    assert s.connected is False
    assert s.project_count == 0


def test_status_reflects_persisted_connection(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    svc.select_projects(
        SelectProjectsRequest(
            api_key="sk",
            workspace_id="ws_1",
            workspace_name="WS",
            project_ids=["p_1", "p_2"],
        )
    )
    s = svc.status()
    assert s.connected is True
    assert s.workspace_id == "ws_1"
    assert s.project_count == 2


def test_disconnect_removes_the_row(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    svc.select_projects(SelectProjectsRequest(api_key="sk", workspace_id="ws_1"))
    svc.disconnect()
    assert svc.status().connected is False


def test_sync_once_writes_imported_scores(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    storage = _InMemoryStorage()
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=_stub_client_factory()
    )
    svc.select_projects(
        SelectProjectsRequest(api_key="sk", workspace_id="ws_1", project_ids=["p_1"])
    )

    response = svc.sync_once()

    # 2 scores from span_a (hallucination, factuality) + 1 from span_b
    assert response.success is True
    assert response.scored_count == 3
    assert len(storage.scores) == 3
    # Matched session_id for span_a (metadata contains reflexio_session_id)
    matched = [s for s in storage.scores if s.session_id == "sess_42"]
    assert len(matched) == 2
    # Unmatched span_b → session_id is None
    unmatched = [s for s in storage.scores if s.session_id is None]
    assert len(unmatched) == 1
    # last_sync_ts persisted
    assert storage.connections["org_t"].last_sync_ts is not None


def test_sync_once_uses_last_successful_sync_ts(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    storage = _InMemoryStorage()
    client = MagicMock()
    client.list_experiments.return_value = []
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=lambda _key: client
    )
    storage.save_braintrust_connection(
        BraintrustConnection(
            org_id="org_t",
            api_key_enc="sk",
            workspace_id="ws_1",
            project_ids=["p_1"],
            last_sync_ts=12345,
        )
    )

    response = svc.sync_once()

    assert response.success is True
    client.list_experiments.assert_called_once_with("p_1", since_ts=12345)
    client.close.assert_called_once()


def test_sync_failure_does_not_advance_last_sync_ts(monkeypatch) -> None:
    monkeypatch.delenv("REFLEXIO_FERNET_KEYS", raising=False)
    _encryption._reset_for_test()
    storage = _InMemoryStorage()
    client = MagicMock()
    client.list_experiments.side_effect = BraintrustHTTPError(500, "boom")
    svc = BraintrustConnectorService(
        storage=storage, org_id="org_t", client_factory=lambda _key: client
    )
    storage.save_braintrust_connection(
        BraintrustConnection(
            org_id="org_t",
            api_key_enc="sk",
            workspace_id="ws_1",
            project_ids=["p_1"],
            last_sync_ts=12345,
        )
    )

    response = svc.sync_once()

    assert response.success is False
    assert storage.connections["org_t"].last_sync_ts == 12345
    assert "boom" in (storage.connections["org_t"].last_error or "")
    client.close.assert_called_once()


def test_sync_once_returns_failure_when_not_connected() -> None:
    svc = BraintrustConnectorService(
        storage=_InMemoryStorage(),
        org_id="org_t",
        client_factory=_stub_client_factory(),
    )
    response = svc.sync_once()
    assert response.success is False


def test_scores_from_spans_extracts_session_id_only_when_metadata_string() -> None:
    """The reflexio_session_id metadata key must be a non-empty string."""
    spans = [
        {
            "id": "s1",
            "created_at": 1,
            "metadata": {"reflexio_session_id": "sess_ok"},
            "scores": {"a": 0.1},
        },
        {
            "id": "s2",
            "created_at": 2,
            "metadata": {"reflexio_session_id": ""},
            "scores": {"a": 0.2},
        },
        {
            "id": "s3",
            "created_at": 3,
            "metadata": {},
            "scores": {"a": 0.3},
        },
        {
            "id": "s4",
            "created_at": 4,
            "scores": {"a": 0.4},  # no metadata at all
        },
    ]
    out = _scores_from_spans(spans, org_id="org_t")
    assert len(out) == 4
    by_id = {s.source_run_id: s for s in out}
    assert by_id["s1"].session_id == "sess_ok"
    assert by_id["s2"].session_id is None
    assert by_id["s3"].session_id is None
    assert by_id["s4"].session_id is None
