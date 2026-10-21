"""Receipt support is distinct from incremental aggregation availability."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.aggregation_operations import (
    PlaybookAggregationOperation,
)
from reflexio.server.api import create_app


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
def test_receipt_lookup_checks_dedicated_capability(available, existing, incremental):
    receipt = PlaybookAggregationOperation(
        operation_id="operation",
        request_id="request",
        agent_version="v1",
        status="succeeded",
        created_at=1,
        updated_at=2,
        completed_at=2,
    )
    reflexio = MagicMock()
    storage = reflexio.get_storage.return_value
    storage.supports_incremental_playbook_aggregation = incremental
    storage.supports_durable_playbook_aggregation_operations = available
    storage.get_playbook_aggregation_operation.return_value = (
        receipt if existing else None
    )
    app = create_app(get_org_id=lambda: "test-org")
    app.state.limiter.enabled = False
    with patch(
        "reflexio.server.routes.playbooks.publisher_api.get_reflexio",
        return_value=reflexio,
    ) as resolve:
        response = TestClient(app, headers={"User-Agent": "Mozilla/5.0"}).get(
            "/api/playbook_aggregation_operations/operation"
        )
    resolve.assert_called_once_with(org_id="test-org")
    storage.get_playbook_aggregation_operation.assert_called_once_with("operation")
    if existing:
        assert response.status_code == 200
        assert response.json() == receipt.model_dump()
    else:
        assert response.status_code == (404 if available else 503)


@pytest.mark.parametrize("deployment_mode", [None, "platform", "self_host"])
def test_request_org_admission_and_restart_poll_register_only_local_worker(
    tmp_path, monkeypatch, deployment_mode
):
    from types import SimpleNamespace

    from fastapi import Request

    from reflexio.models.config_schema import PlaybookAggregatorConfig, PlaybookConfig
    from reflexio.server.api_endpoints.request_context import RequestContext
    from reflexio.server.services.configurator.base_configurator import BaseConfigurator
    from reflexio.server.services.playbook import (
        aggregation_operations,
        aggregation_scheduler,
    )
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    if deployment_mode is None:
        monkeypatch.delenv("DEPLOYMENT_MODE", raising=False)
    else:
        monkeypatch.setenv("DEPLOYMENT_MODE", deployment_mode)
    monkeypatch.setattr(aggregation_scheduler, "_LOCAL_SCHEDULERS", {})
    starts = MagicMock()
    monkeypatch.setattr(
        aggregation_scheduler.PlaybookAggregationScheduler, "start", starts
    )
    monkeypatch.setattr(
        aggregation_operations,
        "create_generation_litellm_client",
        lambda _context: MagicMock(),
    )
    storage = SQLiteStorage(org_id="request-org", db_path=str(tmp_path / "request.db"))
    storage.supports_incremental_playbook_aggregation = False
    configurator = MagicMock(spec=BaseConfigurator)
    configurator.get_prompt_bank_paths.return_value = []
    configurator.create_storage.return_value = storage
    configurator.get_config.return_value.user_playbook_extractor_config = (
        PlaybookConfig(
            extraction_definition_prompt="stub",
            aggregation_config=PlaybookAggregatorConfig(
                min_cluster_size=2, reaggregation_trigger_count=2
            ),
        )
    )
    context = RequestContext(org_id="request-org", configurator=configurator)
    facade = SimpleNamespace(get_storage=lambda: storage, request_context=context)

    def requesting_org(request: Request):
        return request.headers["X-Organization"]

    monkeypatch.setattr(
        "reflexio.server.routes.playbooks.publisher_api.get_reflexio",
        lambda *, org_id: facade if org_id == context.org_id else None,
    )
    app = create_app(get_org_id=requesting_org)
    app.state.limiter.enabled = False
    client = TestClient(
        app, headers={"User-Agent": "Mozilla/5.0", "X-Organization": context.org_id}
    )
    try:
        response = client.post(
            "/api/playbook_aggregation_operations",
            json={"request_id": "admitted", "agent_version": "v1"},
        )
        assert response.status_code == 202 and response.json()["status"] == "queued"
        operation_id = response.json()["operation_id"]
        expected_starts = 1 if deployment_mode is None else 0
        assert starts.call_count == expected_starts
        conflict = client.post(
            "/api/playbook_aggregation_operations",
            json={"request_id": "conflicting", "agent_version": "v1"},
        )
        assert conflict.status_code == 409 and starts.call_count == expected_starts
        # Simulate lost process registration while durable work survives.
        aggregation_scheduler._LOCAL_SCHEDULERS.clear()
        polled = client.get(f"/api/playbook_aggregation_operations/{operation_id}")
        assert polled.status_code == 200 and polled.json()["status"] == "queued"
        if deployment_mode is None:
            assert set(aggregation_scheduler._LOCAL_SCHEDULERS) == {context.org_id}
            scheduler = aggregation_scheduler._LOCAL_SCHEDULERS[context.org_id]
            assert list(scheduler._context_provider()) == [context]
            scheduler._run_once()
            aggregation_scheduler._LOCAL_SCHEDULERS.clear()
            terminal = client.get(
                f"/api/playbook_aggregation_operations/{operation_id}"
            )
            assert terminal.json()["status"] == "succeeded"
            assert not aggregation_scheduler._LOCAL_SCHEDULERS
            assert starts.call_count == 2
        else:
            # Managed deployments retain exclusive scheduling ownership.
            assert (
                not aggregation_scheduler._LOCAL_SCHEDULERS and starts.call_count == 0
            )
    finally:
        client.close()
        storage.conn.close()


def test_missing_sqlite_vec_blocks_admission_without_hiding_receipts(tmp_path):
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    with patch.object(SQLiteStorage, "_try_load_sqlite_vec", return_value=False):
        storage = SQLiteStorage(
            org_id="test-org", db_path=str(tmp_path / "no-vector.db")
        )
    try:
        assert storage.supports_durable_playbook_aggregation_operations is False
        assert (
            storage.conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name='playbook_aggregation_clusters_vec'"
            ).fetchone()[0]
            == 0
        )
        # Simulate a receipt admitted by a previously qualified process.
        receipt = storage.submit_playbook_aggregation_operation("existing", "v1")
        reflexio = MagicMock()
        reflexio.get_storage.return_value = storage
        app = create_app(get_org_id=lambda: "test-org")
        app.state.limiter.enabled = False
        client = TestClient(app, headers={"User-Agent": "Mozilla/5.0"})
        with patch(
            "reflexio.server.routes.playbooks.publisher_api.get_reflexio",
            return_value=reflexio,
        ):
            response = client.post(
                "/api/playbook_aggregation_operations",
                json={"request_id": "new", "agent_version": "v2"},
            )
            assert response.status_code == 503
            assert (
                storage.conn.execute(
                    "SELECT count(*) FROM playbook_aggregation_operations"
                ).fetchone()[0]
                == 1
            )
            assert (
                client.get(
                    f"/api/playbook_aggregation_operations/{receipt.operation_id}"
                ).status_code
                == 200
            )
            assert (
                client.get("/api/playbook_aggregation_operations/missing").status_code
                == 503
            )
    finally:
        storage.conn.close()


def test_legacy_adapter_receipt_lookup_is_unavailable():
    reflexio = MagicMock()
    reflexio.get_storage.return_value.get_playbook_aggregation_operation.side_effect = (
        NotImplementedError
    )
    app = create_app(get_org_id=lambda: "test-org")
    app.state.limiter.enabled = False
    with patch(
        "reflexio.server.routes.playbooks.publisher_api.get_reflexio",
        return_value=reflexio,
    ):
        response = TestClient(app, headers={"User-Agent": "Mozilla/5.0"}).get(
            "/api/playbook_aggregation_operations/operation"
        )
    assert response.status_code == 503


@pytest.mark.parametrize("method", ["post", "get"])
@pytest.mark.parametrize("factory_failure", [False, True])
def test_unconfigured_storage_is_503_without_masking_factory_errors(
    monkeypatch, method, factory_failure
):
    from reflexio.lib._base import ReflexioBase
    from reflexio.server.api_endpoints.request_context import RequestContext

    context = object.__new__(RequestContext)
    context.storage = None
    facade = object.__new__(ReflexioBase)
    facade.request_context = context

    def resolve(**_kwargs):
        if factory_failure:
            raise RuntimeError("configuration failed")
        return facade

    monkeypatch.setattr(
        "reflexio.server.routes.playbooks.publisher_api.get_reflexio", resolve
    )
    start = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.routes.playbooks.ensure_local_playbook_aggregation_scheduler",
        start,
    )
    app = create_app(get_org_id=lambda: "unconfigured")
    app.state.limiter.enabled = False
    client = TestClient(app, raise_server_exceptions=False)
    if method == "post":
        response = client.post(
            "/api/playbook_aggregation_operations",
            json={"request_id": "request", "agent_version": "v1"},
        )
    else:
        response = client.get("/api/playbook_aggregation_operations/operation")
    assert response.status_code == (500 if factory_failure else 503)
    if not factory_failure:
        assert (
            response.json()["detail"]
            == "Durable aggregation is unavailable on this storage"
        )
    start.assert_not_called()
