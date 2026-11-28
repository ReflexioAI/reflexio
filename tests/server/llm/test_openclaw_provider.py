"""Tests for the openclaw LiteLLM custom provider."""

from __future__ import annotations

import json
import os
import subprocess
from unittest.mock import patch

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


@pytest.mark.parametrize("legacy", ["OPENCLAW_BIN", "OPENCLAW_SMART_USE_LOCAL_CLI"])
@pytest.mark.parametrize("value", ["", "synthetic-private-value"])
def test_legacy_bindings_require_explicit_adoption(monkeypatch, legacy, value):
    monkeypatch.setenv(legacy, value)
    with pytest.raises(ValueError, match=legacy) as exc:
        ocp.openclaw_executable()
    if value:
        assert value not in str(exc.value)


def test_missing_selected_executable_refuses_instead_of_searching_path(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(tmp_path / "missing"))
    with (
        patch("shutil.which", return_value="/unselected/openclaw") as which,
        pytest.raises(RuntimeError, match="absolute executable file") as exc,
    ):
        ocp.openclaw_executable()
    which.assert_not_called()
    assert str(tmp_path) not in str(exc.value)


def test_executable_validation_uses_exact_launch_path(monkeypatch, tmp_path):
    chosen = tmp_path / "openclaw"
    chosen.write_text("#!/bin/sh")
    chosen.chmod(0o755)
    monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(chosen) + "/")

    with pytest.raises(RuntimeError, match="absolute executable file"):
        ocp.openclaw_executable()


@pytest.mark.skipif(os.name == "nt", reason="POSIX supports trailing spaces in names")
def test_nonblank_path_keeps_trailing_space(monkeypatch, tmp_path):
    chosen = tmp_path / "openclaw "
    chosen.write_text("#!/bin/sh")
    chosen.chmod(0o755)
    monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(chosen))

    assert ocp.openclaw_executable() == str(chosen)


def test_explicit_executable_is_the_only_launch_authority(monkeypatch, tmp_path):
    chosen = tmp_path / "selected-openclaw"
    chosen.write_text("#!/bin/sh")
    chosen.chmod(0o755)
    monkeypatch.setenv("REFLEXIO_OPENCLAW_EXECUTABLE", str(chosen))
    with patch("shutil.which", return_value="/unselected/openclaw") as which:
        assert ocp.register_if_configured()
        assert ocp.openclaw_executable() == str(chosen)
        which.assert_not_called()


def test_path_binary_cannot_dispatch_without_explicit_authority(monkeypatch):
    monkeypatch.delenv("REFLEXIO_OPENCLAW_EXECUTABLE", raising=False)
    with (
        patch("shutil.which", return_value="/unselected/openclaw"),
        patch.object(ocp, "run_native_cli") as dispatch,
        pytest.raises(ocp.OpenClawCLIError, match="REFLEXIO_OPENCLAW_EXECUTABLE"),
    ):
        ocp._call_cli(prompt="synthetic", model=None, timeout_s=1)
    dispatch.assert_not_called()


class TestExtractText:
    def test_outputs_text(self) -> None:
        # Shape emitted by `openclaw infer model run --json` (2026.10.6).
        payload = {
            "ok": True,
            "capability": "model.run",
            "outputs": [{"text": "Four.", "mediaUrl": None}],
        }
        assert ocp._extract_text(payload) == "Four."

    def test_joins_multiple_outputs(self) -> None:
        assert ocp._extract_text({"outputs": [{"text": "A"}, {"text": "B"}]}) == "AB"

    def test_top_level_text_is_not_the_envelope(self) -> None:
        assert ocp._extract_text({"text": "A"}) == ""

    def test_non_dict_returns_empty(self) -> None:
        assert ocp._extract_text("plain string") == ""
        assert ocp._extract_text([]) == ""
        assert ocp._extract_text(None) == ""
        assert ocp._extract_text({"outputs": "x"}) == ""


class TestCallCli:
    def test_invokes_subprocess_with_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process(
            json.dumps({"ok": True, "outputs": [{"text": "PONG"}]})
        )
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
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process(
            json.dumps({"ok": True, "outputs": [{"text": "ok"}]})
        )
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
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process(
            json.dumps({"ok": True, "outputs": [{"text": "ok"}]})
        )
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
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process("", stderr="bad model", returncode=2)
        with (
            patch("subprocess.run", return_value=proc),
            pytest.raises(ocp.OpenClawCLIError, match="bad model"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_missing_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.OPENCLAW_EXECUTABLE_ENV, raising=False)
        with (
            patch("shutil.which", return_value=None),
            pytest.raises(ocp.OpenClawCLIError, match="REFLEXIO_OPENCLAW_EXECUTABLE"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_invalid_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process("not json")
        with (
            patch("subprocess.run", return_value=proc),
            pytest.raises(ocp.OpenClawCLIError, match="non-JSON"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_empty_completion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process(json.dumps({"unrelated": "x"}))
        with (
            patch("subprocess.run", return_value=proc),
            pytest.raises(ocp.OpenClawCLIError, match="no completion text"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_raises_on_reported_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        proc = _fake_completed_process(json.dumps({"ok": False, "error": "no auth"}))
        with (
            patch("subprocess.run", return_value=proc),
            pytest.raises(ocp.OpenClawCLIError, match="no auth"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_refuses_prompt_over_argv_limit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

        with (
            patch("subprocess.run") as run,
            pytest.raises(ocp.OpenClawCLIError, match="at most"),
        ):
            ocp._call_cli(prompt="x" * 130_000, model=None, timeout_s=10)
        run.assert_not_called()


class TestRegisterIfEnabled:
    def test_skips_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ocp.OPENCLAW_EXECUTABLE_ENV, raising=False)
        assert ocp.register_if_configured() is False

    def test_registers_when_enabled_and_cli_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))
        assert ocp.register_if_configured() is True
        assert ocp._REGISTERED is True

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))
        assert ocp.register_if_configured() is True
        # Second call must not fail
        assert ocp.register_if_configured() is True


class TestAutoDetection:
    """The explicit binding also owns model auto-detection."""

    def test_opt_in_with_cli_resolves_every_generation_role(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        from reflexio.server.llm import model_defaults as md

        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))
        for env_var in md._ENV_TO_PROVIDER:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.delenv("REFLEXIO_NATIVE_CLI_MODEL", raising=False)

        providers = md.detect_available_providers()
        assert "openclaw" in providers
        assert "openclaw" in md.GENERATION_CAPABLE_PROVIDERS
        for role in (md.ModelRole.GENERATION, md.ModelRole.EXTRACTION_AGENT):
            assert md.resolve_model_name(role) == "openclaw/default"

    def test_not_detected_without_opt_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from reflexio.server.llm import model_defaults as md

        monkeypatch.delenv(ocp.OPENCLAW_EXECUTABLE_ENV, raising=False)
        assert "openclaw" not in md.detect_available_providers()

    def test_default_model_uses_openclaws_configured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ocp.ENV_DEFAULT_MODEL, raising=False)
        with patch.object(ocp, "_call_cli", return_value="ok") as call:
            ocp.OpenClawLLM().completion(
                timeout=5,
                model="openclaw/default",
                messages=[{"role": "user", "content": "hi"}],
            )
        assert call.call_args.kwargs["model"] is None


class TestCliFailureReporting:
    @staticmethod
    def _fake(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        fake = tmp_path / "openclaw"
        fake.write_text("#!/bin/sh")
        fake.chmod(0o755)
        monkeypatch.setenv(ocp.OPENCLAW_EXECUTABLE_ENV, str(fake))

    def test_reports_the_json_error_not_the_generic_stderr(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._fake(monkeypatch, tmp_path)
        failed = _fake_completed_process(
            json.dumps({"ok": False, "error": {"message": "Unknown model: x"}}),
            stderr="[openclaw] The CLI command failed.",
            returncode=1,
        )
        with (
            patch("subprocess.run", return_value=failed),
            pytest.raises(ocp.OpenClawCLIError, match="Unknown model: x"),
        ):
            ocp._call_cli(prompt="p", model=None, timeout_s=10)

    def test_retries_state_maintenance_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._fake(monkeypatch, tmp_path)
        monkeypatch.setattr(ocp, "_BUSY_BACKOFF_SECONDS", 0)
        busy = _fake_completed_process(
            json.dumps(
                {
                    "ok": False,
                    "error": {
                        "message": "OpenClaw state at x is undergoing offline "
                        "maintenance; retry when it finishes."
                    },
                }
            ),
            returncode=1,
        )
        ok = _fake_completed_process(
            json.dumps({"ok": True, "outputs": [{"text": "hi"}]})
        )
        with patch("subprocess.run", side_effect=[busy, busy, ok]) as run:
            assert ocp._call_cli(prompt="p", model=None, timeout_s=10) == "hi"
        assert run.call_count == 3


class TestToolCalling:
    """The extraction agent is a tool loop; the CLI only returns text."""

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Return the extracted items",
                "parameters": {
                    "type": "object",
                    "properties": {"items": {"type": "array"}},
                },
            },
        }
    ]

    def test_reports_openclaw_models_as_tool_capable(self) -> None:
        from reflexio.server.llm.tools import supports_tool_calling

        assert supports_tool_calling("openclaw/default")
        assert supports_tool_calling("openclaw/openai/gpt-5.6-luna")

    def test_tool_json_reply_becomes_a_tool_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ocp.ENV_DEFAULT_MODEL, raising=False)
        reply = 'Sure.\n{"tool": "finish", "args": {"items": ["x"]}}'
        with patch.object(ocp, "_call_cli", return_value=reply) as call:
            response = ocp.OpenClawLLM().completion(
                timeout=5,
                model="openclaw/default",
                messages=[
                    {"role": "system", "content": "Extract."},
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "a"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "no results"},
                ],
                optional_params={"tools": self.TOOLS},
            )
        prompt = call.call_args.kwargs["prompt"]
        assert "EXTERNAL TOOL-CALLING MODE" in prompt and "- finish:" in prompt
        assert 'called search with {"q": "a"}' in prompt
        assert "tool[c1]: no results" in prompt
        message = response.choices[0].message
        assert message.content is None
        assert message.tool_calls
        function = message.tool_calls[0].function  # pyright: ignore[reportAttributeAccessIssue]
        assert function.name == "finish"
        assert json.loads(function.arguments) == {"items": ["x"]}
        assert response.choices[0].finish_reason == "tool_calls"

    def test_unparsable_reply_stays_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with patch.object(ocp, "_call_cli", return_value="I cannot help"):
            response = ocp.OpenClawLLM().completion(
                timeout=5,
                model="openclaw/default",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"tools": self.TOOLS},
            )
        assert response.choices[0].message.content == "I cannot help"

    def test_response_format_is_written_into_the_prompt(self) -> None:
        from pydantic import BaseModel

        class Out(BaseModel):
            answer: str

        with patch.object(ocp, "_call_cli", return_value='{"answer": "4"}') as call:
            ocp.OpenClawLLM().completion(
                timeout=5,
                model="openclaw/default",
                messages=[{"role": "user", "content": "2+2"}],
                optional_params={"response_format": Out},
            )
        assert '"answer"' in call.call_args.kwargs["prompt"]
