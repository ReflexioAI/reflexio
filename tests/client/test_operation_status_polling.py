"""Unit tests for ReflexioClient operation-status polling."""

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.domain.enums import OperationStatus


def test_poll_operation_status_accepts_one_second_started_at_skew():
    client = ReflexioClient.__new__(ReflexioClient)

    def fake_make_request(method: str, path: str, **kwargs):
        assert method == "GET"
        assert path == "/api/get_operation_status"
        assert kwargs["params"] == {"service_name": "profile_generation"}
        return {
            "success": True,
            "operation_status": {
                "service_name": "profile_generation",
                "status": "completed",
                "started_at": 99,
                "completed_at": 100,
                "total_users": 1,
                "processed_users": 1,
                "failed_users": 0,
                "progress_percentage": 100,
            },
        }

    client._make_request = fake_make_request

    response = client._poll_operation_status(
        "profile_generation",
        poll_interval=0,
        max_wait=0.1,
        min_started_at=100,
    )

    assert response.operation_status is not None
    assert response.operation_status.status == OperationStatus.COMPLETED
