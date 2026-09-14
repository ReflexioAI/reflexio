"""Capability discovery must not hide previously committed receipts."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from reflexio.models.api_schema.aggregation_operations import (
    PlaybookAggregationOperation,
)
from reflexio.server.api import create_app


@pytest.mark.parametrize("available", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_receipt_lookup_reports_capability_only_when_missing(available, existing):
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
    storage.supports_incremental_playbook_aggregation = available
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
        assert response.json()["detail"] == (
            "Aggregation operation not found"
            if available
            else "Durable aggregation is unavailable on this storage"
        )
