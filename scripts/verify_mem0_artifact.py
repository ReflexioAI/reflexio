"""Verify an installed Reflexio wheel against the real mem0 package."""

# ruff: noqa: S101 -- this executable is a compact CI assertion harness.

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import httpx
import mem0
from mem0.client.types import AddMemoryOptions, SearchMemoryOptions

from reflexio import ReflexioClient


class _Record:
    def __init__(self, **values: object) -> None:
        self._values = values

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return self._values


class _ReflexioStub:
    timeout = 0.5

    def __init__(self) -> None:
        self.sync_publishes: list[dict[str, object]] = []
        self.sync_searches: list[dict[str, object]] = []
        self.async_publishes: list[dict[str, object]] = []
        self.async_searches: list[dict[str, object]] = []

    @staticmethod
    def _search_response() -> SimpleNamespace:
        return SimpleNamespace(
            success=True,
            profiles=[_Record(id="profile-1")],
            user_playbooks=[_Record(id=1)],
            agent_playbooks=[_Record(id=2)],
        )

    def publish_interaction(self, **kwargs: object) -> SimpleNamespace:
        self.sync_publishes.append(kwargs)
        return SimpleNamespace(success=True)

    def search(self, **kwargs: object) -> SimpleNamespace:
        self.sync_searches.append(kwargs)
        return self._search_response()

    async def publish_interaction_async(self, **kwargs: object) -> SimpleNamespace:
        self.async_publishes.append(kwargs)
        return SimpleNamespace(success=True)

    async def search_async(self, **kwargs: object) -> SimpleNamespace:
        self.async_searches.append(kwargs)
        return self._search_response()


def _response(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/add/"):
        return httpx.Response(200, json={"results": [{"id": "memory-1"}]})
    if request.url.path.endswith("/search/"):
        return httpx.Response(
            200, json={"results": [{"id": "memory-1", "memory": "prefers tea"}]}
        )
    raise AssertionError(f"Unexpected mem0 request: {request.method} {request.url}")


def _validate_api_key(client: object) -> str:
    client.org_id = "org-ci"  # type: ignore[attr-defined]
    client.project_id = "project-ci"  # type: ignore[attr-defined]
    return "ci@example.com"


def _assert_signatures() -> None:
    from reflexio.mem0 import AsyncMemoryClient, MemoryClient

    assert issubclass(MemoryClient, mem0.MemoryClient)
    assert issubclass(AsyncMemoryClient, mem0.AsyncMemoryClient)
    for wrapper, base in (
        (MemoryClient, mem0.MemoryClient),
        (AsyncMemoryClient, mem0.AsyncMemoryClient),
    ):
        wrapper_init = inspect.signature(wrapper)
        base_init = inspect.signature(base)
        for name in base_init.parameters:
            assert name in wrapper_init.parameters
            assert wrapper_init.parameters[name].kind == base_init.parameters[name].kind
        for name in (
            "reflexio_api_key",
            "reflexio_url_endpoint",
            "reflexio_client",
            "reflexio_timeout",
        ):
            assert wrapper_init.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
            assert wrapper_init.parameters[name].default is None
        signature = inspect.signature(wrapper.search)
        assert (
            signature.parameters["include_reflexio"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert signature.parameters["include_reflexio"].default is False


def _assert_sync_contract() -> None:
    from reflexio.mem0 import MemoryClient

    reflexio = _ReflexioStub()
    transport = httpx.MockTransport(_response)
    http_client = httpx.Client(transport=transport)
    with patch.object(mem0.MemoryClient, "_validate_api_key", _validate_api_key):
        client = MemoryClient(
            api_key="mem0-test-key",
            host="https://mem0.invalid",
            client=http_client,
            reflexio_client=cast(ReflexioClient, reflexio),
        )

    add_options = AddMemoryOptions(
        filters={
            "app_id": "app/雪",
            "user_id": "user|1",
            "agent_id": "agent:1",
            "run_id": "run/1",
        }
    )
    added = client.add("I prefer tea", add_options)
    assert added == {"results": [{"id": "memory-1"}]}
    assert len(reflexio.sync_publishes) == 1

    search_options = SearchMemoryOptions(
        filters={"app_id": "app/雪", "user_id": "user|1", "agent_id": "agent:1"}
    )
    base = client.search("tea", search_options)
    assert base == {"results": [{"id": "memory-1", "memory": "prefers tea"}]}
    assert reflexio.sync_searches == []

    enriched = client.search("tea", search_options, include_reflexio=True)
    assert enriched is not base
    assert enriched["reflexio"]["status"] == "ok"
    assert enriched["reflexio"]["profiles"] == [{"id": "profile-1"}]
    assert len(reflexio.sync_searches) == 1


async def _assert_async_contract() -> None:
    from reflexio.mem0 import AsyncMemoryClient

    reflexio = _ReflexioStub()
    transport = httpx.MockTransport(_response)
    http_client = httpx.AsyncClient(transport=transport)
    with patch.object(mem0.AsyncMemoryClient, "_validate_api_key", _validate_api_key):
        client = AsyncMemoryClient(
            api_key="mem0-test-key",
            host="https://mem0.invalid",
            client=http_client,
            reflexio_client=cast(ReflexioClient, reflexio),
        )

    options = AddMemoryOptions(filters={"user_id": "user-1", "agent_id": "agent-1"})
    added = await client.add("I prefer tea", options)
    assert added == {"results": [{"id": "memory-1"}]}
    assert len(reflexio.async_publishes) == 1

    search_options = SearchMemoryOptions(
        filters={"user_id": "user-1", "agent_id": "agent-1"}
    )
    base = await client.search("tea", search_options)
    assert reflexio.async_searches == []
    enriched = await client.search("tea", search_options, include_reflexio=True)
    assert enriched is not base
    assert enriched["reflexio"]["status"] == "ok"
    assert len(reflexio.async_searches) == 1
    await http_client.aclose()


def main() -> None:
    from reflexio.mem0 import AsyncMemory, Memory

    assert Memory is mem0.Memory
    assert AsyncMemory is mem0.AsyncMemory
    _assert_signatures()
    _assert_sync_contract()
    asyncio.run(_assert_async_contract())
    print("installed mem0 artifact contract passed")


if __name__ == "__main__":
    main()
