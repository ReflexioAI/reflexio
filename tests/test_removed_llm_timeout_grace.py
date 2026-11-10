"""The shared loader refuses obsolete LLM timeout padding before blank pruning."""

import os
from pathlib import Path

import pytest

from reflexio.cli import env_loader


@pytest.mark.parametrize("mode", ["platform", "self_host"])
@pytest.mark.parametrize("value", ["", "0", "300", "600", "private-obsolete-value"])
def test_raw_file_refuses_timeout_grace_before_export(
    monkeypatch, tmp_path: Path, mode: str, value: str
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", tmp_path / "unused-home")
    monkeypatch.setattr(env_loader, "_loaded_env_path", None)
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", raising=False)
    monkeypatch.delenv("LLM_UPGRADE_MARKER", raising=False)
    (tmp_path / f".env.{mode}").write_text(
        f"REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS={value}\nLLM_UPGRADE_MARKER=exported\n"
    )
    with pytest.raises(
        ValueError, match="REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS"
    ) as error:
        env_loader.load_existing_reflexio_env_for_mode(cli_mode=mode)
    assert "private-obsolete-value" not in str(error.value)
    assert "one accepted-work deadline" in str(error.value)
    assert "REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS" not in os.environ
    assert "LLM_UPGRADE_MARKER" not in os.environ


@pytest.mark.parametrize("value", ["", "0", "5", "private-obsolete-value"])
def test_direct_client_refuses_obsolete_timeout_padding(monkeypatch, value):
    from unittest.mock import Mock

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    execute = Mock()
    monkeypatch.setattr(client, "_run_completion", execute)
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", value)
    with pytest.raises(
        ValueError, match="REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS"
    ) as error:
        client._completion_with_hard_timeout({"model": "gpt-4o", "timeout": 1}, 1)
    assert "private-obsolete-value" not in str(error.value)
    execute.assert_not_called()


@pytest.mark.parametrize("value", ["", "0", "5"])
def test_public_request_rejects_padding_before_reservation(monkeypatch, value):
    from unittest.mock import Mock

    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    client = LiteLLMClient(LiteLLMConfig(model="gpt-4o"))
    guard = Mock()
    physical = Mock()
    monkeypatch.setattr(client, "_completion_with_hard_timeout", physical)
    monkeypatch.setenv("REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS", value)
    with pytest.raises(ValueError, match="REFLEXIO_LLM_HARD_TIMEOUT_GRACE_SECONDS"):
        client.generate_response("synthetic", provider_request_guard=guard)
    guard.assert_not_called()
    physical.assert_not_called()
