"""The shared loader refuses obsolete WAL identity input before blank pruning."""

import os
from pathlib import Path

import pytest

from reflexio.cli import env_loader


@pytest.mark.parametrize("mode", ["platform", "self_host"])
@pytest.mark.parametrize("value", ["", "0", "300", "600", "private-obsolete-value"])
def test_raw_file_refuses_wal_instance_override_before_export(
    monkeypatch, tmp_path: Path, mode: str, value: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", tmp_path / "unused-home")
    monkeypatch.setattr(env_loader, "_loaded_env_path", None)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("REFLEXIO_INSTANCE_ID", raising=False)
    monkeypatch.delenv("WAL_UPGRADE_MARKER", raising=False)
    (tmp_path / f".env.{mode}").write_text(
        f"REFLEXIO_INSTANCE_ID={value}\nWAL_UPGRADE_MARKER=exported\n"
    )
    with pytest.raises(ValueError, match="REFLEXIO_INSTANCE_ID") as error:
        env_loader.load_existing_reflexio_env_for_mode(cli_mode=mode)
    assert "private-obsolete-value" not in str(error.value)
    assert "existing resolved WAL leaf" in str(error.value)
    assert "REFLEXIO_INSTANCE_ID" not in os.environ
    assert "WAL_UPGRADE_MARKER" not in os.environ
