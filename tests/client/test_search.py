from typing import Any

from reflexio import ReflexioClient


def test_unified_search_serializes_tag_filter(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {
            "success": True,
            "profiles": [],
            "agent_playbooks": [],
            "user_playbooks": [],
        }

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.search(query="billing", tags=["billing", "support"])

    assert captured["method"] == "POST"
    assert captured["path"] == "/api/search"
    assert captured["json"]["tags"] == ["billing", "support"]


def test_agent_playbook_search_serializes_assignment_user(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {"success": True, "agent_playbooks": []}

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.search_agent_playbooks(query="billing", user_id="user-1")

    assert captured["json"]["user_id"] == "user-1"
