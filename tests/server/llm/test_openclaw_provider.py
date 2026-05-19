"""Tests for the openclaw LiteLLM custom provider."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm.providers import openclaw_provider as ocp


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Each test starts with fresh registration state."""
    ocp._REGISTERED = False
    ocp._HANDLER = None


def _fake_completed_process(
    stdout: str, stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["openclaw"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestEnvFlags:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.ENV_ENABLE, raising=False)
        assert ocp._env_enabled() is False

    def test_enabled_with_truthy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ocp.ENV_ENABLE, "1")
        assert ocp._env_enabled() is True
        monkeypatch.setenv(ocp.ENV_ENABLE, "true")
        assert ocp._env_enabled() is True
        monkeypatch.setenv(ocp.ENV_ENABLE, "YES")
        assert ocp._env_enabled() is True

    def test_disabled_with_falsy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ocp.ENV_ENABLE, "0")
        assert ocp._env_enabled() is False
        monkeypatch.setenv(ocp.ENV_ENABLE, "")
        assert ocp._env_enabled() is False


class TestResolveCliPath:
    def test_env_override_honored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))
        assert ocp._resolve_cli_path() == str(fake)

    def test_falls_back_to_which(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.ENV_CLI_PATH, raising=False)
        with patch("shutil.which", return_value="/usr/local/bin/openclaw"):
            assert ocp._resolve_cli_path() == "/usr/local/bin/openclaw"

    def test_none_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.ENV_CLI_PATH, raising=False)
        with patch("shutil.which", return_value=None):
            assert ocp._resolve_cli_path() is None


class TestExtractText:
    def test_text_key(self) -> None:
        assert ocp._extract_text({"text": "A"}) == "A"

    def test_content_key(self) -> None:
        assert ocp._extract_text({"content": "B"}) == "B"

    def test_output_key(self) -> None:
        assert ocp._extract_text({"output": "C"}) == "C"

    def test_message_content_nested(self) -> None:
        assert ocp._extract_text({"message": {"content": "D"}}) == "D"

    def test_unknown_shape_returns_empty(self) -> None:
        assert ocp._extract_text({"unknown": "X"}) == ""

    def test_non_dict_returns_empty(self) -> None:
        assert ocp._extract_text("plain string") == ""
        assert ocp._extract_text([]) == ""
        assert ocp._extract_text(None) == ""


class TestCallCli:
    def test_invokes_subprocess_with_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process(json.dumps({"text": "PONG"}))
        with patch("subprocess.run", return_value=proc) as run:
            result = ocp._call_cli(prompt="ping", model=None, timeout_s=10)
        assert result == "PONG"
        argv = run.call_args[0][0]
        assert str(fake) in argv
        assert "--prompt" in argv
        assert "ping" in argv
        assert "--json" in argv
        assert "infer" in argv and "model" in argv and "run" in argv

    def test_propagates_recursion_guard(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process(json.dumps({"text": "ok"}))
        with patch("subprocess.run", return_value=proc) as run:
            ocp._call_cli(prompt="p", model=None, timeout_s=10)
        env = run.call_args[1]["env"]
        assert env.get("OPENCLAW_SMART_INTERNAL") == "1"

    def test_passes_model_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process(json.dumps({"text": "ok"}))
        with patch("subprocess.run", return_value=proc) as run:
            ocp._call_cli(prompt="p", model="anthropic/claude-sonnet-4-6", timeout_s=10)
        argv = run.call_args[0][0]
        assert "--model" in argv
        assert "anthropic/claude-sonnet-4-6" in argv

    def test_raises_on_nonzero_exit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process("", stderr="bad model", returncode=2)
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(ocp.OpenClawCLIError, match="bad model"):
                ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_missing_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.ENV_CLI_PATH, raising=False)
        with patch("shutil.which", return_value=None):
            with pytest.raises(ocp.OpenClawCLIError, match="not found"):
                ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process("not json")
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(ocp.OpenClawCLIError, match="non-JSON"):
                ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_empty_completion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))

        proc = _fake_completed_process(json.dumps({"unrelated": "x"}))
        with patch("subprocess.run", return_value=proc):
            with pytest.raises(ocp.OpenClawCLIError, match="no completion text"):
                ocp._call_cli(prompt="p", model=None, timeout_s=10)


class TestRegisterIfEnabled:
    def test_skips_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.ENV_ENABLE, raising=False)
        assert ocp.register_if_enabled() is False

    def test_skips_when_cli_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ocp.ENV_ENABLE, "1")
        monkeypatch.delenv(ocp.ENV_CLI_PATH, raising=False)
        with patch("shutil.which", return_value=None):
            assert ocp.register_if_enabled() is False

    def test_registers_when_enabled_and_cli_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_ENABLE, "1")
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))
        assert ocp.register_if_enabled() is True
        assert ocp._REGISTERED is True

    def test_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.ENV_ENABLE, "1")
        monkeypatch.setenv(ocp.ENV_CLI_PATH, str(fake))
        assert ocp.register_if_enabled() is True
        # Second call must not fail
        assert ocp.register_if_enabled() is True
