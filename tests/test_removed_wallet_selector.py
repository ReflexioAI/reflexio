"""The shared loader refuses obsolete wallet policy before blank pruning."""

import os
from pathlib import Path

import pytest

from reflexio.cli import env_loader


@pytest.mark.parametrize("mode", ["platform", "self_host"])
@pytest.mark.parametrize("value", ["", "false", "true", "private-obsolete-value"])
def test_raw_file_refuses_wallet_selector_before_export(
    monkeypatch, tmp_path: Path, mode: str, value: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", tmp_path / "unused-home")
    monkeypatch.setattr(env_loader, "_loaded_env_path", None)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("POOLED_CREDITS_ENABLED", raising=False)
    monkeypatch.delenv("WALLET_UPGRADE_MARKER", raising=False)
    (tmp_path / f".env.{mode}").write_text(
        f"POOLED_CREDITS_ENABLED={value}\nWALLET_UPGRADE_MARKER=exported\n"
    )
    with pytest.raises(ValueError, match="POOLED_CREDITS_ENABLED") as error:
        env_loader.load_existing_reflexio_env_for_mode(cli_mode=mode)
    assert "private-obsolete-value" not in str(error.value)
    assert "do not copy its boolean value" in str(error.value)
    assert "POOLED_CREDITS_ENABLED" not in os.environ
    assert "WALLET_UPGRADE_MARKER" not in os.environ
