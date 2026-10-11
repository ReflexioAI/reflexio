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
