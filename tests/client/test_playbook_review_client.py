from datetime import UTC, datetime
from typing import Any

from reflexio import ReflexioClient


def test_review_user_playbooks_posts_typed_bulk_request(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {
            "success": True,
            "report_only": False,
            "selected_count": 0,
            "results": [],
        }

    monkeypatch.setattr(client, "_make_request", fake_make_request)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 2, tzinfo=UTC)

    response = client.review_user_playbooks(
        start_time=start,
        end_time=end,
        top_k=25,
        report_only=False,
    )

    assert response.success is True
    assert response.report_only is False
    assert captured == {
        "method": "POST",
        "path": "/api/review_user_playbooks",
        "json": {
            "start_time": "2026-01-01T00:00:00Z",
            "end_time": "2026-01-02T00:00:00Z",
            "top_k": 25,
            "report_only": False,
        },
        "timeout": 300,
    }


def test_report_only_review_uses_extended_client_timeout(monkeypatch) -> None:
    client = ReflexioClient(
        api_key="test-key",
        url_endpoint="http://localhost:8000",
        timeout=450,
    )
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {
            "success": True,
            "report_only": True,
            "selected_count": 0,
            "results": [],
        }

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.review_user_playbooks(
        start_time=datetime(2026, 1, 1, tzinfo=UTC),
        end_time=datetime(2026, 1, 2, tzinfo=UTC),
    )

    assert captured["timeout"] == 600
