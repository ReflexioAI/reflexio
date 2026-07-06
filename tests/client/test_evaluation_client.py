"""Unit tests for ``ReflexioClient`` evaluation management methods."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from reflexio import (
    GradeOnDemandRequest,
    GradeOnDemandResponse,
    RegenerateRequest,
    RegenerateStartResponse,
    RegenerateStatusResponse,
)
from reflexio.client import ReflexioClient


def _json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b"{}"
    response.headers = {"Content-Type": "application/json"}
    return response


@patch("reflexio.client.client.requests.Session")
def test_regenerate_evaluations_posts_kwargs(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {"job_id": "regen_123", "total": 7}
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.regenerate_evaluations(from_ts=100, to_ts=200)

    assert isinstance(result, RegenerateStartResponse)
    assert result.job_id == "regen_123"
    assert result.total == 7
    args, kwargs = mock_session.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/evaluations/regenerate")
    assert kwargs["json"] == {"evaluation_name": None, "from_ts": 100, "to_ts": 200}


@patch("reflexio.client.client.requests.Session")
def test_regenerate_evaluations_accepts_request_model(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {"job_id": "regen_model", "total": 1}
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    client.regenerate_evaluations(RegenerateRequest(from_ts=10, to_ts=20))

    assert mock_session.request.call_args.kwargs["json"] == {
        "evaluation_name": None,
        "from_ts": 10,
        "to_ts": 20,
    }


@patch("reflexio.client.client.requests.Session")
def test_get_evaluation_regeneration_status(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {
            "job_id": "regen_123",
            "status": "running",
            "total": 7,
            "completed": 3,
            "failed": 1,
            "failures": [{"session_id": "s1", "reason": "timeout"}],
            "started_at": 1000.0,
            "finished_at": None,
            "total_candidates": 9,
            "sampled_count": 7,
            "concurrency_limit": 2,
        }
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.get_evaluation_regeneration_status("regen_123")

    assert isinstance(result, RegenerateStatusResponse)
    assert result.completed == 3
    assert result.failures[0].session_id == "s1"
    args, _kwargs = mock_session.request.call_args
    assert args[0] == "GET"
    assert args[1].endswith("/api/evaluations/regenerate/regen_123")


@patch("reflexio.client.client.requests.Session")
def test_cancel_evaluation_regeneration(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response({"status": "cancelled"})
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.cancel_evaluation_regeneration("regen_123")

    assert result == {"status": "cancelled"}
    args, _kwargs = mock_session.request.call_args
    assert args[0] == "DELETE"
    assert args[1].endswith("/api/evaluations/regenerate/regen_123")


@patch("reflexio.client.client.requests.Session")
def test_grade_on_demand_posts_kwargs(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {
            "session_id": "session_001",
            "result_id": 42,
            "cached": False,
            "skipped_reason": None,
        }
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.grade_on_demand(
        session_id="session_001",
        agent_version="v2.1.0",
    )

    assert isinstance(result, GradeOnDemandResponse)
    assert result.result_id == 42
    args, kwargs = mock_session.request.call_args
    assert args[0] == "POST"
    assert args[1].endswith("/api/evaluations/grade_on_demand")
    assert kwargs["json"] == {
        "session_id": "session_001",
        "agent_version": "v2.1.0",
        "evaluation_name": None,
    }


@patch("reflexio.client.client.requests.Session")
def test_grade_on_demand_accepts_request_dict(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {
            "session_id": "session_001",
            "result_id": None,
            "cached": False,
            "skipped_reason": "NO_REQUESTS",
        }
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    client.grade_on_demand(
        {
            "session_id": "session_001",
            "agent_version": "v2.1.0",
        }
    )

    assert mock_session.request.call_args.kwargs["json"] == {
        "session_id": "session_001",
        "agent_version": "v2.1.0",
        "evaluation_name": None,
    }


def test_evaluation_models_export_from_client_package() -> None:
    from reflexio.client import (  # noqa: PLC0415
        GradeOnDemandRequest as ClientGradeOnDemandRequest,
    )
    from reflexio.client import (
        RegenerateRequest as ClientRegenerateRequest,
    )

    assert ClientGradeOnDemandRequest is GradeOnDemandRequest
    assert ClientRegenerateRequest is RegenerateRequest
