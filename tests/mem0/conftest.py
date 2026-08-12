"""Fixtures for reflexio.mem0 tests.

mem0ai is an optional dependency and is not installed in CI, so these tests
inject a stub ``mem0`` module into ``sys.modules`` before importing
``reflexio.mem0``, and remove the stub-bound ``reflexio.mem0`` modules
afterwards so other tests never see them.
"""

import sys
import types
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


class StubMemoryClient:
    """Minimal stand-in for mem0.MemoryClient that records calls."""

    def __init__(self, api_key=None, host=None, client=None):
        self.api_key = api_key
        self.host = host
        self.client = client
        self.calls = []
        self.add_result = {"results": [{"id": "m1", "event": "ADD"}]}
        self.search_result = {
            "results": [{"id": "m1", "memory": "likes jazz", "score": 0.9}]
        }
        self.get_all_result = {
            "count": 0,
            "next": None,
            "previous": None,
            "results": [],
        }
        self.raise_on_add = None
        self.raise_on_search = None

    def add(self, messages, options=None, **kwargs):
        self.calls.append(("add", messages, options, kwargs))
        if self.raise_on_add is not None:
            raise self.raise_on_add
        return self.add_result

    def search(self, query, options=None, **kwargs):
        self.calls.append(("search", query, options, kwargs))
        if self.raise_on_search is not None:
            raise self.raise_on_search
        return self.search_result

    def get_all(self, options=None, **kwargs):
        self.calls.append(("get_all", options, kwargs))
        return self.get_all_result

    def get(self, memory_id):
        self.calls.append(("get", memory_id))
        return {"id": memory_id}

    def delete_all(self, options=None, **kwargs):
        self.calls.append(("delete_all", options, kwargs))
        return {"message": "ok"}

    def delete(self, memory_id):
        self.calls.append(("delete", memory_id))
        return {"id": memory_id}

    def delete_users(self, user_id=None, agent_id=None, app_id=None, run_id=None):
        self.calls.append(("delete_users", user_id, agent_id, app_id, run_id))
        return {"message": "ok"}

    def reset(self):
        self.calls.append(("reset",))
        return {"message": "reset"}


class StubAsyncMemoryClient(StubMemoryClient):
    """Async hosted-client stand-in with the same observable test state."""

    async def add(self, messages, options=None, **kwargs):
        return super().add(messages, options, **kwargs)

    async def search(self, query, options=None, **kwargs):
        return super().search(query, options, **kwargs)

    async def get_all(self, options=None, **kwargs):
        return super().get_all(options, **kwargs)

    async def delete_all(self, options=None, **kwargs):
        return super().delete_all(options, **kwargs)

    async def delete(self, memory_id):
        return super().delete(memory_id)

    async def delete_users(self, user_id=None, agent_id=None, app_id=None, run_id=None):
        return super().delete_users(user_id, agent_id, app_id, run_id)

    async def reset(self):
        return super().reset()


class AddMemoryOptions:  # pragma: no cover - import/type surface only
    pass


class SearchMemoryOptions:  # pragma: no cover - import/type surface only
    pass


def _purge_reflexio_mem0_modules():
    for name in [n for n in list(sys.modules) if n.startswith("reflexio.mem0")]:
        sys.modules.pop(name, None)
    reflexio_pkg = sys.modules.get("reflexio")
    if reflexio_pkg is not None and hasattr(reflexio_pkg, "mem0"):
        delattr(reflexio_pkg, "mem0")


@pytest.fixture
def mem0_stub(monkeypatch):
    """Install a stub ``mem0`` module and yield it; clean up bound imports."""
    module: Any = types.ModuleType("mem0")
    module.__path__ = []
    module.MemoryClient = StubMemoryClient
    module.Memory = type("Memory", (), {})
    module.AsyncMemory = type("AsyncMemory", (), {})
    module.AsyncMemoryClient = StubAsyncMemoryClient
    client_module: Any = types.ModuleType("mem0.client")
    client_module.__path__ = []
    types_module: Any = types.ModuleType("mem0.client.types")
    types_module.AddMemoryOptions = AddMemoryOptions
    types_module.SearchMemoryOptions = SearchMemoryOptions
    for name in [n for n in list(sys.modules) if n == "mem0" or n.startswith("mem0.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mem0", module)
    monkeypatch.setitem(sys.modules, "mem0.client", client_module)
    monkeypatch.setitem(sys.modules, "mem0.client.types", types_module)
    _purge_reflexio_mem0_modules()
    yield module
    _purge_reflexio_mem0_modules()


@pytest.fixture
def wrapped_cls(mem0_stub):
    """The reflexio.mem0.MemoryClient class, imported against the stub."""
    from reflexio.mem0 import MemoryClient

    return MemoryClient


@pytest.fixture
def async_wrapped_cls(mem0_stub):
    """The async wrapper class, imported against the async hosted stub."""
    from reflexio.mem0 import AsyncMemoryClient

    return AsyncMemoryClient


class FakeView:
    """Stands in for a pydantic view model returned by ReflexioClient.search."""

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, mode="python"):
        return dict(self._payload)


@pytest.fixture
def reflexio_mock():
    """MagicMock ReflexioClient with an empty-but-valid unified search result."""
    mock = MagicMock()
    mock.timeout = 5.0
    mock.publish_interaction.return_value = types.SimpleNamespace(success=True)
    mock.search.return_value = types.SimpleNamespace(
        success=True,
        profiles=[FakeView({"profile_id": "p1", "content": "prefers jazz"})],
        user_playbooks=[FakeView({"user_playbook_id": 7, "content": "greet by name"})],
        agent_playbooks=[
            FakeView({"agent_playbook_id": 3, "content": "confirm order"})
        ],
    )
    mock.publish_interaction_async = AsyncMock(
        return_value=types.SimpleNamespace(success=True)
    )
    mock.search_async = AsyncMock(return_value=mock.search.return_value)
    mock._make_async_request = AsyncMock(return_value={"success": True})
    return mock
