"""Unit tests for ReflexioClient operation-status polling."""

import pytest

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.domain.enums import OperationStatus


def test_poll_operation_status_accepts_one_second_started_at_skew(
    monkeypatch: pytest.MonkeyPatch,
):
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

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    response = client._poll_operation_status(
        "profile_generation",
        poll_interval=0,
        max_wait=0.1,
        min_started_at=100,
    )

    assert response.operation_status is not None
    assert response.operation_status.status == OperationStatus.COMPLETED


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
def test_waiting_generation_preserves_actual_generated_count(monkeypatch, kind, status):
    from reflexio.models.api_schema.service_schemas import (
        GetOperationStatusResponse,
        OperationStatusInfo,
        RerunPlaybookGenerationRequest,
        RerunProfileGenerationRequest,
    )

    client = ReflexioClient.__new__(ReflexioClient)
    monkeypatch.setattr(
        client, "_make_request", lambda *_args, **_kwargs: {"success": True}
    )
    result = GetOperationStatusResponse(
        success=True,
        operation_status=OperationStatusInfo(
            service_name=f"{kind}_generation",
            status=OperationStatus(status),
            started_at=1,
            processed_users=1,
            stats={"total_generated": 3},
        ),
    )
    monkeypatch.setattr(
        client, "_poll_operation_status", lambda *_args, **_kwargs: result
    )
    request = (
        RerunProfileGenerationRequest()
        if kind == "profile"
        else RerunPlaybookGenerationRequest(agent_version="v1")
    )
    response = getattr(client, f"_rerun_{kind}_generation_sync")(request)
    assert response.success is (status == "completed")
    assert (
        getattr(
            response,
            "profiles_generated" if kind == "profile" else "playbooks_generated",
        )
        == 3
    )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize(
    "error", [TimeoutError("deadline"), RuntimeError("unavailable")]
)
def test_polling_error_cannot_be_mistaken_for_completion(monkeypatch, kind, error):
    from reflexio.models.api_schema.service_schemas import (
        RerunPlaybookGenerationRequest,
        RerunProfileGenerationRequest,
    )

    client = ReflexioClient.__new__(ReflexioClient)
    monkeypatch.setattr(
        client, "_make_request", lambda *_args, **_kwargs: {"success": True}
    )

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(client, "_poll_operation_status", fail)
    request = (
        RerunProfileGenerationRequest()
        if kind == "profile"
        else RerunPlaybookGenerationRequest(agent_version="v1")
    )
    response = getattr(client, f"_rerun_{kind}_generation_sync")(request)
    assert response.success is False
    assert "check generation status before retrying" in response.msg
