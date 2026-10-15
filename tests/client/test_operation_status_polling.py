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
        client,
        "_make_request",
        lambda *_args, **_kwargs: {"success": True, "operation_id": "attempt-test"},
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

    def poll_expected_attempt(_service_name, **kwargs):
        assert kwargs["operation_id"] == "attempt-test"
        return result

    monkeypatch.setattr(client, "_poll_operation_status", poll_expected_attempt)
    request = (
        RerunProfileGenerationRequest()
        if kind == "profile"
        else RerunPlaybookGenerationRequest(agent_version="v1")
    )
    response = getattr(client, f"_rerun_{kind}_generation_sync")(request)
    assert response.success is (status == "completed")
    assert response.operation_id == "attempt-test"
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
        client,
        "_make_request",
        lambda *_args, **_kwargs: {"success": True, "operation_id": "attempt-test"},
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
    assert response.operation_id == "attempt-test"


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
def test_exact_attempt_ignores_prior_completion_and_clock_skew(monkeypatch, terminal):
    client = ReflexioClient.__new__(ReflexioClient)
    responses = iter(
        [
            {
                "success": True,
                "operation_status": {
                    "service_name": "playbook_generation",
                    "status": "completed",
                    "started_at": 100,
                    "request_params": {"operation_id": "old"},
                    "stats": {"total_generated": 99},
                },
            },
            {
                "success": True,
                "operation_status": {
                    "service_name": "playbook_generation",
                    "status": terminal,
                    "started_at": 1,
                    "request_params": {"operation_id": "mine"},
                    "stats": {"total_generated": 3},
                },
            },
        ]
    )
    monkeypatch.setattr(
        client, "_make_request", lambda *_args, **_kwargs: next(responses)
    )
    response = client._poll_operation_status(
        "playbook_generation",
        poll_interval=0,
        max_wait=0.1,
        min_started_at=100,
        operation_id="mine",
    )
    assert response.operation_status is not None
    assert response.operation_status.status == OperationStatus(terminal)
    assert response.operation_status.stats["total_generated"] == 3


def test_mismatched_attempt_never_confirms_completion(monkeypatch):
    client = ReflexioClient.__new__(ReflexioClient)
    monkeypatch.setattr(
        client,
        "_make_request",
        lambda *_args, **_kwargs: {
            "success": True,
            "operation_status": {
                "service_name": "playbook_generation",
                "status": "completed",
                "started_at": 100,
                "request_params": {"operation_id": "other"},
            },
        },
    )
    with pytest.raises(TimeoutError):
        client._poll_operation_status(
            "playbook_generation", poll_interval=0.01, max_wait=0, operation_id="mine"
        )


@pytest.mark.parametrize("kind", ["profile", "playbook"])
def test_legacy_server_cannot_confirm_an_attempt(monkeypatch, kind):
    from reflexio.models.api_schema.service_schemas import (
        RerunPlaybookGenerationRequest,
        RerunProfileGenerationRequest,
    )

    client = ReflexioClient.__new__(ReflexioClient)
    monkeypatch.setattr(
        client, "_make_request", lambda *_args, **_kwargs: {"success": True}
    )

    def unexpected_poll(*_args, **_kwargs):
        pytest.fail("Legacy service label is not an attempt ID")

    monkeypatch.setattr(client, "_poll_operation_status", unexpected_poll)
    request = (
        RerunProfileGenerationRequest()
        if kind == "profile"
        else RerunPlaybookGenerationRequest(agent_version="v1")
    )
    response = getattr(client, f"_rerun_{kind}_generation_sync")(request)
    assert response.success is False
    assert "upgrade the server" in response.msg


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["profile", "playbook"])
@pytest.mark.parametrize("count", [None, 0, 3])
def test_completed_attempt_distinguishes_unknown_from_zero_count(
    monkeypatch, kind, count
):
    from reflexio.models.api_schema.service_schemas import (
        RerunPlaybookGenerationRequest,
        RerunProfileGenerationRequest,
    )

    client = ReflexioClient.__new__(ReflexioClient)
    calls = []

    def respond(method, path, **_kwargs):
        calls.append((method, path))
        if method == "POST":
            return {"success": True, "operation_id": "completed-attempt"}
        return {
            "success": True,
            "operation_status": {
                "service_name": f"{kind}_generation",
                "status": "completed",
                "started_at": 1,
                "request_params": {"operation_id": "completed-attempt"},
                "processed_users": 1,
                "stats": {} if count is None else {"total_generated": count},
            },
        }

    monkeypatch.setattr(client, "_make_request", respond)
    request = (
        RerunProfileGenerationRequest()
        if kind == "profile"
        else RerunPlaybookGenerationRequest(agent_version="v1")
    )
    result = getattr(client, f"_rerun_{kind}_generation_sync")(request)
    assert result.success is True
    assert result.operation_id == "completed-attempt"
    assert (
        getattr(
            result, "profiles_generated" if kind == "profile" else "playbooks_generated"
        )
        == count
    )
    assert calls == [
        ("POST", f"/api/rerun_{kind}_generation"),
        ("GET", "/api/get_operation_status"),
    ]
