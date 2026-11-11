"""Provider imports belong to LLM entry points and retain their ordering guard."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_ORDER_GUARD = """
import importlib.abc
import sys
class OrderGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "litellm":
            assert "openai.resources" in sys.modules, "provider resources were not preloaded"
sys.meta_path.insert(0, OrderGuard())
"""


def run_probe(program: str, tmp_path: Path):
    return subprocess.run(  # noqa: S603 - fixed interpreter and repository-owned programs
        [sys.executable, "-c", program],
        env=dict(
            os.environ,
            PYTHON_DOTENV_DISABLED="1",
            LITELLM_LOCAL_MODEL_COST_MAP="true",
            REFLEXIO_LOG_DIR=str(tmp_path / "logs"),
            LOCAL_STORAGE_PATH=str(tmp_path / "storage"),
        ),
        text=True,
        capture_output=True,
        timeout=35,
        check=False,
    )


def test_server_environment_import_does_not_load_provider_clients(tmp_path):
    result = run_probe(
        "import sys\n"
        "import reflexio.server.env_utils\n"
        "assert 'openai' not in sys.modules\n"
        "assert 'litellm' not in sys.modules\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "module",
    [
        "reflexio.server.llm.litellm_client",
        "reflexio.server.llm.providers.claude_code_provider",
        "reflexio.server.services.playbook_optimizer.judge",
        "reflexio.server.services.offline_tuner.evidence.native_capacity",
    ],
)
def test_cold_provider_import_preloads_resources(module, tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + f"import importlib\nimportlib.import_module({module!r})\n"
        + "from litellm.exceptions import ContextWindowExceededError as expected\n"
        + "from reflexio.server.llm._litellm_subprocess import ContextWindowExceededError\n"
        + "assert ContextWindowExceededError is expected\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_simultaneous_first_provider_imports_complete(tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + """
import importlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import reflexio.server.env_utils
assert "openai" not in sys.modules
modules = (
    "reflexio.server.llm.litellm_client",
    "reflexio.server.llm.providers.claude_code_provider",
    "reflexio.server.services.playbook_optimizer.judge",
    "reflexio.server.services.offline_tuner.evidence.native_capacity",
)
barrier = Barrier(len(modules))
def load(name):
    barrier.wait(timeout=5)
    return importlib.import_module(name).__name__
with ThreadPoolExecutor(max_workers=4) as executor:
    assert tuple(executor.map(load, modules, timeout=25)) == modules
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_eager_mock_preloads_resources_and_restores_completion(tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + """
from types import SimpleNamespace
from unittest.mock import Mock
from reflexio.test_support.llm_mock import configure_llm_mock, cleanup_llm_mock
config = SimpleNamespace(args=[], option=SimpleNamespace(collectonly=False))
configure_llm_mock(config)
import litellm
assert isinstance(litellm.completion, Mock)
cleanup_llm_mock(config)
assert not isinstance(litellm.completion, Mock)
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_preload_oracle_refuses_a_missing_bootstrap(tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + """
from types import ModuleType
sys.modules["reflexio.provider_bootstrap"] = ModuleType("reflexio.provider_bootstrap")
import reflexio.server.llm.litellm_client
""",
        tmp_path,
    )
    assert result.returncode != 0
    assert "provider resources were not preloaded" in result.stderr


def test_direct_module_litellm_imports_follow_bootstrap():
    import ast

    source_root = Path(__file__).resolve().parents[3] / "reflexio"
    found = []
    missing = []
    for path in sorted(source_root.rglob("*.py")):
        guarded = False
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if "reflexio.provider_bootstrap" in names:
                guarded = True
            if any(name == "litellm" or name.startswith("litellm.") for name in names):
                found.append(path.relative_to(source_root).as_posix())
                if not guarded:
                    missing.append(found[-1])
    assert found, "the source scan must reach real provider imports"
    assert not missing, (
        f"preload provider resources before importing LiteLLM: {missing}"
    )


@pytest.mark.parametrize(
    "statement",
    [
        "mock.litellm_is_patched()",
        "mock.inline_test_completion({}, 0)",
        "mock.configure_llm_mock(SimpleNamespace(args=[], option=SimpleNamespace(collectonly=True))); mock.unpatched_litellm().__enter__()",
    ],
)
def test_mock_helpers_preload_on_first_provider_touch(statement, tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + """
from types import SimpleNamespace
from reflexio.test_support import llm_mock as mock
assert "openai" not in sys.modules
class FirstImportReached(Exception):
    pass
class StopBeforeProviderCall(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "litellm":
            assert "openai.resources" in sys.modules, "provider resources were not preloaded"
            raise FirstImportReached()
sys.meta_path.insert(0, StopBeforeProviderCall())
try:
    exec(STATEMENT)
except FirstImportReached:
    pass
else:
    raise AssertionError("the helper did not reach its first provider import")
""".replace("STATEMENT", repr(statement)),
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "role", ["controller", "worker", "scheduler-only", "transport-controller"]
)
def test_distributed_controller_defers_mock_but_executors_remain_eager(role, tmp_path):
    result = run_probe(
        _ORDER_GUARD
        + """
from types import SimpleNamespace
from unittest.mock import Mock
from reflexio.test_support.llm_mock import configure_llm_mock, cleanup_llm_mock
role = ROLE
controller = role in ("controller", "transport-controller")
config = SimpleNamespace(args=[], option=SimpleNamespace(
    collectonly=False,
    dist="load",
    numprocesses=2 if role in ("controller", "worker") else 0,
    tx=["popen"] if role != "scheduler-only" else [],
))
if role == "worker":
    config.workerinput = {}
configure_llm_mock(config)
assert ("litellm" in sys.modules) == (not controller)
from reflexio.server.llm import litellm_client
import litellm
assert isinstance(litellm.completion, Mock)
cleanup_llm_mock(config)
assert not isinstance(litellm.completion, Mock)
""".replace("ROLE", repr(role)),
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_keyless_provider_floor_does_not_import_disabled_cli_adapters(tmp_path):
    result = run_probe(
        """
import os
import sys
from reflexio.server.llm.model_defaults import provider_env_vars
from reflexio.test_support.llm_credentials import ensure_provider_credential
for name in provider_env_vars():
    os.environ.pop(name, None)
os.environ.pop("CLAUDE_SMART_USE_LOCAL_CLI", None)
os.environ.pop("OPENCLAW_SMART_USE_LOCAL_CLI", None)
ensure_provider_credential()
assert os.environ["OPENAI_API_KEY"] == "sk-placeholder-mock"
assert "reflexio.server.llm.providers.claude_code_provider" not in sys.modules
assert "reflexio.server.llm.providers.openclaw_provider" not in sys.modules
assert "litellm" not in sys.modules
assert "openai" not in sys.modules
""",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_llm_fixture_definitions_do_not_load_provider_runtime(tmp_path):
    fixture_path = Path(__file__).with_name("conftest.py").resolve()
    result = run_probe(
        "import runpy, sys\n"
        + f"runpy.run_path({str(fixture_path)!r})\n"
        + "assert 'litellm' not in sys.modules\nassert 'openai' not in sys.modules\n",
        tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
