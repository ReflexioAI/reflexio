"""Fixtures for reflexio.mem0 tests.

mem0ai is an optional dependency and is not installed in CI, so these tests
inject a stub ``mem0`` module into ``sys.modules`` before importing
``reflexio.mem0``, and remove the stub-bound ``reflexio.mem0`` modules
afterwards so other tests never see them.
"""

import sys
import types
from typing import Any
from unittest.mock import MagicMock

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
    module.MemoryClient = StubMemoryClient
    module.Memory = type("Memory", (), {})
    module.AsyncMemory = type("AsyncMemory", (), {})
    module.AsyncMemoryClient = type("AsyncMemoryClient", (), {})
    for name in [n for n in list(sys.modules) if n == "mem0" or n.startswith("mem0.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "mem0", module)
    _purge_reflexio_mem0_modules()
    yield module
    _purge_reflexio_mem0_modules()


@pytest.fixture
def wrapped_cls(mem0_stub):
    """The reflexio.mem0.MemoryClient class, imported against the stub."""
    from reflexio.mem0 import MemoryClient

    return MemoryClient


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
    mock.search.return_value = types.SimpleNamespace(
        profiles=[FakeView({"profile_id": "p1", "content": "prefers jazz"})],
        user_playbooks=[FakeView({"user_playbook_id": 7, "content": "greet by name"})],
        agent_playbooks=[
            FakeView({"agent_playbook_id": 3, "content": "confirm order"})
        ],
    )
    return mock
