"""Fresh interpreters exercise the session mock's import and cleanup boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    [
        "alias-and-restore",
        "e2e",
        "runtime",
        "already-imported",
        "cleanup-before-import",
        "temporary-before-import",
        "bounded-pending",
        "import-cleanup-order",
        "immediate-cleanup-order",
        "installed",
        pytest.param(
            "fork-lock-reset",
            marks=pytest.mark.skipif(
                not hasattr(os, "fork"), reason="fork is POSIX-only"
            ),
        ),
    ],
)
def test_mock_import_lifecycle(mode: str, tmp_path: Path) -> None:
    env = dict(
        os.environ,
        PYTHON_DOTENV_DISABLED="1",
        LITELLM_MODE="PRODUCTION",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        REFLEXIO_LOG_DIR=str(tmp_path / "logs"),
        LOCAL_STORAGE_PATH=str(tmp_path / "storage"),
    )
    result = subprocess.run(  # noqa: S603 — repository-owned script and fixed modes
        [
            sys.executable,
            str(Path(__file__).with_name("_llm_mock_import_probe.py")),
            mode,
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.unit
def test_credential_floor_does_not_import_execution_adapters() -> None:
    result = subprocess.run(  # noqa: S603 — repository-owned script and fixed modes
        [
            sys.executable,
            "-c",
            """
import os
import sys
from reflexio.server.llm.model_defaults import GENERATION_CAPABLE_PROVIDERS, provider_env_vars
from reflexio.test_support.llm_credentials import ensure_provider_credential
for name in provider_env_vars():
    os.environ.pop(name, None)
name = next(name for name, provider in provider_env_vars().items()
            if provider in GENERATION_CAPABLE_PROVIDERS)
os.environ[name] = 'sk-placeholder-mock'
assert 'litellm' not in sys.modules
ensure_provider_credential()
assert os.environ[name] == 'sk-placeholder-mock'
assert 'litellm' not in sys.modules
""",
        ],
        env=dict(os.environ, PYTHON_DOTENV_DISABLED="1"),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_live_scope_before_transport_import_keeps_real_dispatch(tmp_path: Path) -> None:
    result = subprocess.run(  # noqa: S603 — fixed interpreter and repository-owned program
        [
            sys.executable,
            "-c",
            """
import sys
from types import SimpleNamespace
from reflexio.test_support import llm_mock
config = SimpleNamespace(args=['tests/server/llm'], option=SimpleNamespace(collectonly=False))
llm_mock.configure_llm_mock(config)
assert 'reflexio.server.llm._litellm_text_generation' not in sys.modules
with llm_mock.unpatched_litellm():
    from reflexio.server.llm import _litellm_text_generation as generation
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    client = LiteLLMClient(LiteLLMConfig(model='gpt-4o'))
    assert client._run_completion is generation._run_isolated_completion
    with llm_mock.unpatched_litellm():
        assert client._run_completion is generation._run_isolated_completion
    assert client._run_completion is generation._run_isolated_completion
assert client._run_completion is llm_mock.inline_test_completion
llm_mock.cleanup_llm_mock(config)
assert client._run_completion is generation._run_isolated_completion
""",
        ],
        env=dict(
            os.environ,
            PYTHON_DOTENV_DISABLED="1",
            REFLEXIO_LOG_DIR=str(tmp_path / "logs"),
            LOCAL_STORAGE_PATH=str(tmp_path / "storage"),
        ),
        text=True,
        capture_output=True,
        timeout=25,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
