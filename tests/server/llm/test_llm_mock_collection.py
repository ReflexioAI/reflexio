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
        SENTRY_DSN="",
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
        env=dict(os.environ, PYTHON_DOTENV_DISABLED="1", SENTRY_DSN=""),
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
