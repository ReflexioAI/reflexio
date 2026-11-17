"""Actual early CLI dispatch upgrades only the explicit owned retained file."""

import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "test_data/sqlite_70daf32c"


@pytest.mark.parametrize("forbid_loaders", [False, True])
def test_real_offline_cli_ignores_operator_env_and_reports_honest_retry(
    tmp_path, forbid_loaders
):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "operator.env"
    original = b"REFLEXIO_DURABLE_LEARNING_POLL_SECONDS=private-invalid-input\nOPERATOR_SECRET=owned-marker\n"
    config.write_bytes(original)
    target = tmp_path / "retained.sqlite"
    target.write_bytes(
        gzip.decompress((FIXTURES / "eligible_queued.sqlite.gz").read_bytes())
    )
    source = Path(__file__).resolve().parents[2]
    environment = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "TMPDIR", "LANG", "SystemRoot"}
    }
    environment.update(
        HOME=str(home),
        USERPROFILE=str(home),
        PYTHONPATH=str(source),
        REFLEXIO_ENV_FILE=str(config),
        REFLEXIO_DURABLE_LEARNING_POLL_SECONDS="private-invalid-input",
    )
    launch = [sys.executable, "-m", "reflexio.cli"]
    if forbid_loaders:
        launch = [
            sys.executable,
            "-c",
            """import runpy
from reflexio.cli import env_loader
def forbidden(*args, **kwargs):
    raise AssertionError("operator loader called")
env_loader.load_existing_reflexio_env = forbidden
env_loader.load_reflexio_env = forbidden
runpy.run_module("reflexio.cli", run_name="__main__")
""",
        ]
    outputs = []
    for _ in range(2):
        result = subprocess.run(  # noqa: S603 -- Fixed CLI entrypoint and owned fixture path; no shell.
            [
                *launch,
                "--json",
                "data",
                "upgrade-local",
                str(target),
            ],
            env=environment,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "owned-marker" not in result.stdout + result.stderr
        outputs.append(json.loads(result.stdout)["data"])
    assert outputs[0]["established"] is True
    assert outputs[0]["original_provenance_available"] is True
    assert outputs[0]["provenance"]["closure_counts"]["queued_admissions"] > 0
    assert outputs[1]["established"] is False
    assert outputs[1]["original_provenance_available"] is False
    assert outputs[1]["provenance"] is None
    assert outputs[0]["authority"] == outputs[1]["authority"]
    assert config.read_bytes() == original
    assert sorted(p.name for p in home.iterdir()) == ["operator.env"]


def test_ordinary_help_tree_registers_same_command_and_refuses_missing_file(tmp_path):
    app = create_app()
    runner = CliRunner()
    help_result = runner.invoke(app, ["data", "upgrade-local", "--help"])
    assert help_result.exit_code == 0
    assert "stopping affected" in help_result.stdout
    target = tmp_path / "missing.sqlite"
    result = runner.invoke(app, ["--json", "data", "upgrade-local", str(target)])
    assert result.exit_code != 0
    assert not result.stdout
    assert json.loads(result.stderr)["ok"] is False
    assert not target.exists()
