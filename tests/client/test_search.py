import inspect
from pathlib import Path
from typing import Any

import pytest

from reflexio import ReflexioClient
from reflexio.models.api_schema.retriever_schema import (
    SearchUserPlaybookRequest,
    UnifiedSearchRequest,
)


def _non_null_schema(
    field_name: str, *, request_model: type[Any] = UnifiedSearchRequest
) -> dict[str, Any]:
    field_schema = request_model.model_json_schema()["properties"][field_name]
    return next(
        option
        for option in field_schema.get("anyOf", [field_schema])
        if option.get("type") != "null"
    )


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


def test_user_playbook_search_serializes_correlation_ids(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {"success": True, "user_playbooks": []}

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.search_user_playbooks(
        query="billing", request_id="request-1", session_id="session-1"
    )

    assert captured["json"]["request_id"] == "request-1"
    assert captured["json"]["session_id"] == "session-1"


@pytest.mark.asyncio
async def test_unified_search_async_uses_native_transport(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    async def fake_make_async_request(
        method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        captured.update(method=method, path=path, **kwargs)
        return {
            "success": True,
            "profiles": [],
            "agent_playbooks": [],
            "user_playbooks": [],
        }

    monkeypatch.setattr(client, "_make_async_request", fake_make_async_request)
    result = await client.search_async(
        query="billing", user_id="u1", agent_version="a1", top_k=4
    )

    assert result.success is True
    assert captured["method"] == "POST"
    assert captured["path"] == "/api/search"
    assert captured["json"]["user_id"] == "u1"
    assert captured["json"]["agent_version"] == "a1"
    assert captured["json"]["top_k"] == 4


def test_playbook_and_unified_searches_serialize_source(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    payloads: dict[str, dict[str, Any]] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        payloads[path] = kwargs["json"]
        if path == "/api/search_user_playbooks":
            return {"success": True, "user_playbooks": []}
        if path == "/api/search_agent_playbooks":
            return {"success": True, "agent_playbooks": []}
        return {
            "success": True,
            "profiles": [],
            "agent_playbooks": [],
            "user_playbooks": [],
        }

    monkeypatch.setattr(client, "_make_request", fake_make_request)

    client.search_user_playbooks(query="billing", source="api")
    client.search_agent_playbooks(query="billing", source="api")
    client.search(query="billing", source="api")

    assert payloads["/api/search_user_playbooks"]["source"] == "api"
    assert payloads["/api/search_agent_playbooks"]["source"] == "api"
    assert payloads["/api/search"]["source"] == "api"


@pytest.mark.asyncio
async def test_unified_search_async_serializes_source(monkeypatch) -> None:
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")
    captured: dict[str, Any] = {}

    async def fake_make_async_request(
        method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "success": True,
            "profiles": [],
            "agent_playbooks": [],
            "user_playbooks": [],
        }

    monkeypatch.setattr(client, "_make_async_request", fake_make_async_request)

    await client.search_async(query="billing", source="webhook")

    assert captured["json"]["source"] == "webhook"


def test_unified_search_docs_track_schema_contract() -> None:
    top_k_schema = _non_null_schema("top_k")
    identifier_limit = _non_null_schema("request_id")["maxLength"]
    interaction_minimum = _non_null_schema("interaction_id")["exclusiveMinimum"] + 1
    client_docs = inspect.getdoc(ReflexioClient.search) or ""
    registry_docs = (
        Path(__file__).parents[2] / "docs/lib/methods/unified-search.ts"
    ).read_text(encoding="utf-8")

    assert f"1 to {top_k_schema['maximum']}" in client_docs
    assert client_docs.count(f"at most {identifier_limit} characters") >= 3
    assert f"positive integer (minimum {interaction_minimum})" in client_docs

    assert f"1 to {top_k_schema['maximum']}" in registry_docs
    for field_name in ("user_id", "request_id", "session_id"):
        assert f'name: "{field_name}"' in registry_docs
    assert registry_docs.count(f"at most {identifier_limit} characters") >= 3
    assert 'name: "interaction_id"' in registry_docs
    assert f"positive integer (minimum {interaction_minimum})" in registry_docs


def test_user_playbook_search_docs_track_top_k_schema_contract() -> None:
    top_k_schema = _non_null_schema("top_k", request_model=SearchUserPlaybookRequest)
    identifier_limit = _non_null_schema(
        "request_id", request_model=SearchUserPlaybookRequest
    )["maxLength"]
    client_docs = inspect.getdoc(ReflexioClient.search_user_playbooks) or ""
    registry_docs = (
        Path(__file__).parents[2] / "docs/lib/methods/user-playbooks.ts"
    ).read_text(encoding="utf-8")

    assert f"1 to {top_k_schema['maximum']}" in client_docs
    assert client_docs.count(f"at most {identifier_limit} characters") >= 2
    assert f"1 to {top_k_schema['maximum']}" in registry_docs
    assert registry_docs.count(f"at most {identifier_limit} characters") >= 2
