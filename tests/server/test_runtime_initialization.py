"""Real entrypoints share startup; metadata imports have no startup side effects."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "mode",
    [
        "passive",
        "api",
        "inference",
        "server_entry",
        "library",
        "storage",
        "failure",
        "concurrent",
        "reentrant",
        "logging_retry",
        "legacy_attributes",
        "reload",
    ],
)
def test_runtime_startup_in_fresh_owned_process(tmp_path, mode):
    source = Path(__file__).resolve().parents[2]
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=sk-owned-startup\n"
        f"LOCAL_STORAGE_PATH={tmp_path / 'data'}\n"
        "REFLEXIO_LOG_PROFILE=production\n"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "TMPDIR", "LANG", "SystemRoot"}
    }
    environment.update(
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        PYTHONPATH=str(source),
        REFLEXIO_ENV_FILE=str(env_file),
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        REFLEXIO_NATIVE_COMPUTE_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    if mode == "passive":
        environment["REFLEXIO_LOG_PROFILE"] = "invalid-serving-policy"
    elif mode == "logging_retry":
        environment["REFLEXIO_LOG_PROFILE"] = "verbose"
    result = subprocess.run(  # noqa: S603 -- owned source and fixture only.
        [
            sys.executable,
            str(source / "tests/test_scripts/runtime_startup_probe.py"),
            mode,
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"PASS {mode}" in result.stdout


def test_integrity_exceptions_keep_one_identity_and_failure_family():
    from reflexio.server.llm import _litellm_types, input_budget

    for name in (
        "InputBudgetConfigurationError",
        "ExtractionInputIntegrityError",
        "SavedOutputIncompatibleError",
        "ComputationEvidenceChangedError",
    ):
        assert getattr(input_budget, name) is getattr(_litellm_types, name)
    assert issubclass(
        input_budget.ExtractionInputIntegrityError,
        input_budget.InputBudgetConfigurationError,
    )
