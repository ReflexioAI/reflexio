"""The shared loader refuses obsolete alarm timing before blank pruning."""

import os
from pathlib import Path

import pytest

from reflexio.cli import env_loader


@pytest.mark.parametrize("mode", ["platform", "self_host"])
@pytest.mark.parametrize("value", ["", "0", "300", "600", "private-obsolete-value"])
def test_raw_file_refuses_alarm_timer_before_export(
    monkeypatch, tmp_path: Path, mode: str, value: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", tmp_path / "unused-home")
    monkeypatch.setattr(env_loader, "_loaded_env_path", None)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS", raising=False)
    monkeypatch.delenv("ALARM_UPGRADE_MARKER", raising=False)
    (tmp_path / f".env.{mode}").write_text(
        f"REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS={value}\nALARM_UPGRADE_MARKER=exported\n"
    )
    with pytest.raises(
        ValueError, match="REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS"
    ) as error:
        env_loader.load_existing_reflexio_env_for_mode(cli_mode=mode)
    assert "private-obsolete-value" not in str(error.value)
    assert "do not copy its duration" in str(error.value)
    assert "REFLEXIO_LEARNING_ALARM_SUPPRESS_SECONDS" not in os.environ
    assert "ALARM_UPGRADE_MARKER" not in os.environ
