"""Smoke check against the real mem0ai package (skipped when not installed).

Run locally with `uv pip install mem0ai` to verify the subclass binds against
the real base class; CI skips this module.
"""

import sys

import pytest

mem0 = pytest.importorskip("mem0")


@pytest.fixture(autouse=True)
def _fresh_reflexio_mem0():
    for name in [n for n in list(sys.modules) if n.startswith("reflexio.mem0")]:
        sys.modules.pop(name, None)
    yield
    for name in [n for n in list(sys.modules) if n.startswith("reflexio.mem0")]:
        sys.modules.pop(name, None)


def test_wrapper_binds_against_real_mem0():
    from reflexio.mem0 import AsyncMemoryClient, MemoryClient

    assert issubclass(MemoryClient, mem0.MemoryClient)
    assert issubclass(AsyncMemoryClient, mem0.AsyncMemoryClient)
    # The methods the wrapper overrides must exist on the real base.
    for method in ("add", "search", "get_all"):
        assert callable(getattr(mem0.MemoryClient, method))


def test_local_reexports_match_real_mem0():
    import reflexio.mem0 as wrapper_module

    assert wrapper_module.Memory is mem0.Memory
    assert wrapper_module.AsyncMemory is mem0.AsyncMemory
