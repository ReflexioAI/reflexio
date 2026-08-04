"""Python client contracts for session outcomes."""

import inspect
from typing import Any

from reflexio import ReflexioClient


def test_get_session_outcomes_posts_typed_filters(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {"success": True, "session_outcomes": [], "message": "Found 0"}

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    response = client.get_session_outcomes(
        session_ids=["s1", "s1"],
        user_id="u1",
        source="agent",
        outcome="success",
        label="resolved",
        start_time=100,
        end_time=200,
        top_k=25,
        offset=5,
    )

    assert response.success is True
    assert captured == {
        "method": "POST",
        "path": "/api/get_session_outcomes",
        "json": {
            "session_ids": ["s1"],
            "user_id": "u1",
            "source": "agent",
            "outcome": "success",
            "label": "resolved",
            "start_time": 100,
            "end_time": 200,
            "top_k": 25,
            "offset": 5,
        },
    }


def test_get_session_outcomes_has_no_untyped_filter_kwargs() -> None:
    parameters = inspect.signature(ReflexioClient.get_session_outcomes).parameters

    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
