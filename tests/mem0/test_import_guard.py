"""Import behavior of reflexio.mem0 with and without mem0ai installed."""

import builtins
import subprocess
import sys

import pytest

from tests.mem0.conftest import _purge_reflexio_mem0_modules


def test_missing_mem0_raises_helpful_import_error(monkeypatch):
    _purge_reflexio_mem0_modules()
    # A None entry makes `import mem0` raise ImportError deterministically.
    monkeypatch.setitem(sys.modules, "mem0", None)
    with pytest.raises(ImportError, match=r"reflexio-ai\[mem0\]"):
        import reflexio.mem0  # noqa: F401
    _purge_reflexio_mem0_modules()


def test_installed_mem0_import_failure_is_preserved(monkeypatch):
    _purge_reflexio_mem0_modules()
    original_import = builtins.__import__

    def fail_inside_mem0(name, *args, **kwargs):
        if name == "mem0":
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_inside_mem0)
    with pytest.raises(ModuleNotFoundError, match="httpx") as exc_info:
        import reflexio.mem0  # noqa: F401
    assert exc_info.value.name == "httpx"
    _purge_reflexio_mem0_modules()


def test_import_reflexio_does_not_import_mem0_module():
    code = (
        "import sys; import reflexio; "
        "assert 'reflexio.mem0' not in sys.modules, 'reflexio.mem0 imported eagerly'; "
        "assert 'mem0' not in sys.modules, 'mem0 imported eagerly'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603 — fixed argv.


def test_local_classes_reexported_and_hosted_async_wrapped(mem0_stub):
    import reflexio.mem0 as wrapper_module

    assert wrapper_module.Memory is mem0_stub.Memory
    assert wrapper_module.AsyncMemory is mem0_stub.AsyncMemory
    assert issubclass(wrapper_module.AsyncMemoryClient, mem0_stub.AsyncMemoryClient)
    assert wrapper_module.MemoryClient is not mem0_stub.MemoryClient
    assert issubclass(wrapper_module.MemoryClient, mem0_stub.MemoryClient)
