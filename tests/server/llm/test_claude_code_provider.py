"""Tests for the claude-code LiteLLM custom provider."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from reflexio.server.llm.providers import claude_code_provider as ccp
from reflexio.server.llm.providers.claude_code_provider import (
    ClaudeCodeCLIError,
    ClaudeCodeLLM,
    _split_system_and_dialogue,
    is_claude_code_available,
    register_if_enabled,
)


def _stream_json(result_text: str) -> str:
    """Build a minimal stream-json NDJSON body with one terminal ``result`` event."""
    return json.dumps({"type": "result", "result": result_text, "session_id": "s"}) + "\n"


@pytest.fixture(autouse=True)
def _reset_module_state() -> None:
    """Each test starts with fresh registration and warn-once flags."""
    ccp._REGISTERED = False
    ccp._HANDLER = None
    ccp._IMAGE_WARNED = False
    ccp._MULTITURN_WARNED = False
    ccp._UNSUPPORTED_PARAMS_WARNED.clear()


def _fake_completed_process(
    stdout: str, stderr: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class _Person(BaseModel):
    name: str
    age: int


class TestSplitSystemAndDialogue:
    def test_system_message_separated(self) -> None:
        msgs = [
            {"role": "system", "content": "You are a helper."},
            {"role": "user", "content": "Hi"},
        ]
        sys_prompt, dialogue = _split_system_and_dialogue(msgs)
        assert sys_prompt == "You are a helper."
        assert dialogue == "User: Hi"

    def test_multiple_system_messages_joined(self) -> None:
        msgs = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Go"},
        ]
        sys_prompt, _ = _split_system_and_dialogue(msgs)
        assert sys_prompt == "Rule 1\n\nRule 2"

    def test_assistant_and_user_alternation(self) -> None:
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        _, dialogue = _split_system_and_dialogue(msgs)
        assert dialogue == "User: q1\n\nAssistant: a1\n\nUser: q2"

    def test_content_block_list_flattened(self) -> None:
        msgs = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "cached rule",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": "hi"},
        ]
        sys_prompt, dialogue = _split_system_and_dialogue(msgs)
        assert sys_prompt == "cached rule"
        assert dialogue == "User: hi"

    def test_tool_role_prefixed(self) -> None:
        msgs = [
            {"role": "user", "content": "fetch"},
            {"role": "tool", "content": "result: 42"},
            {"role": "assistant", "content": "done"},
        ]
        _, dialogue = _split_system_and_dialogue(msgs)
        assert "Tool: result: 42" in dialogue

    def test_image_blocks_dropped_with_single_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"data": "xyz"}},
                ],
            },
        ]
        with caplog.at_level(
            logging.WARNING, logger="reflexio.server.llm.providers.claude_code_provider"
        ):
            _, dialogue = _split_system_and_dialogue(msgs)
        assert "User: describe" in dialogue
        assert "data:image" not in dialogue
        # Second image block in the same split must not produce a second warning.
        image_warns = [r for r in caplog.records if "image content" in r.message]
        assert len(image_warns) == 1

    def test_multiturn_emits_single_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        msgs = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        with caplog.at_level(
            logging.WARNING, logger="reflexio.server.llm.providers.claude_code_provider"
        ):
            _split_system_and_dialogue(msgs)
            _split_system_and_dialogue(msgs)
        multiturn_warns = [r for r in caplog.records if "multi-turn" in r.message]
        assert len(multiturn_warns) == 1


class TestClaudeCodeLLMCompletion:
    def _mock_cli(
        self, monkeypatch: pytest.MonkeyPatch, result_text: str = "ok"
    ) -> MagicMock:
        """Mock subprocess.run to return a stream-json NDJSON body with one result event."""
        mock_run = MagicMock(return_value=_fake_completed_process(_stream_json(result_text)))
        monkeypatch.setattr(ccp.subprocess, "run", mock_run)
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        return mock_run

    def test_basic_completion_shapes_model_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._mock_cli(monkeypatch, result_text="hello world")
        llm = ClaudeCodeLLM()

        response = llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "ping"}],
        )

        assert response.choices[0].message.content == "hello world"  # type: ignore[union-attr]
        assert response.model == "claude-code/default"
        # stream-json does not surface usage tokens at terminal event.
        assert response.usage.prompt_tokens == 0  # type: ignore[attr-defined]
        assert response.usage.completion_tokens == 0  # type: ignore[attr-defined]
        assert response.usage.total_tokens == 0  # type: ignore[attr-defined]

    def test_uses_stream_json_output_format(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The provider must invoke the CLI with stream-json, not the legacy json envelope."""
        mock_run = self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )

        cmd = mock_run.call_args.args[0]
        fmt_idx = cmd.index("--output-format")
        assert cmd[fmt_idx + 1] == "stream-json"
        # stream-json requires --verbose to emit events.
        assert "--verbose" in cmd

    def test_sets_max_retries_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLAUDE_CODE_MAX_RETRIES=3 must be passed in the subprocess env."""
        mock_run = self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )

        env = mock_run.call_args.kwargs["env"]
        assert env["CLAUDE_CODE_MAX_RETRIES"] == "3"
        assert env["CLAUDE_SMART_INTERNAL"] == "1"

    def test_system_message_goes_to_append_system_prompt_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hello"},
            ],
        )

        cmd = mock_run.call_args.args[0]
        assert "--append-system-prompt" in cmd
        flag_idx = cmd.index("--append-system-prompt")
        assert cmd[flag_idx + 1] == "Be terse."
        # User turn goes through stdin, not argv.
        assert mock_run.call_args.kwargs["input"] == "User: hello"

    def test_no_system_message_omits_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "hello"}],
        )

        cmd = mock_run.call_args.args[0]
        assert "--append-system-prompt" not in cmd

    def test_response_format_appends_schema_to_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = self._mock_cli(monkeypatch, result_text='{"name":"Yi","age":31}')
        llm = ClaudeCodeLLM()

        response = llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "Extract"}],
            optional_params={"response_format": _Person},
        )

        cmd = mock_run.call_args.args[0]
        assert "--append-system-prompt" in cmd
        flag_idx = cmd.index("--append-system-prompt")
        injected_system = cmd[flag_idx + 1]
        assert "JSON" in injected_system
        assert '"name"' in injected_system
        assert '"age"' in injected_system
        # Raw JSON text passes through; LiteLLMClient parses it downstream.
        assert response.choices[0].message.content == '{"name":"Yi","age":31}'  # type: ignore[union-attr]

    def test_response_format_merges_with_existing_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = self._mock_cli(monkeypatch, result_text="{}")
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "go"},
            ],
            optional_params={"response_format": _Person},
        )

        cmd = mock_run.call_args.args[0]
        flag_idx = cmd.index("--append-system-prompt")
        injected_system = cmd[flag_idx + 1]
        assert injected_system.startswith("Be terse.")
        assert "JSON" in injected_system

    def test_response_format_dict_schema_also_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mock_run = self._mock_cli(monkeypatch, result_text="{}")
        llm = ClaudeCodeLLM()

        llm.completion(
            model="claude-code/default",
            messages=[{"role": "user", "content": "go"}],
            optional_params={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "schema": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                        }
                    },
                }
            },
        )

        cmd = mock_run.call_args.args[0]
        flag_idx = cmd.index("--append-system-prompt")
        assert '"x"' in cmd[flag_idx + 1]

    def test_unsupported_params_warn_once(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        with caplog.at_level(
            logging.WARNING,
            logger="reflexio.server.llm.providers.claude_code_provider",
        ):
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"temperature": 0.0, "max_tokens": 512},
            )
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
                optional_params={"temperature": 0.0, "max_tokens": 512},
            )

        temp_warns = [r for r in caplog.records if "temperature" in r.message]
        max_warns = [r for r in caplog.records if "max_tokens" in r.message]
        assert len(temp_warns) == 1
        assert len(max_warns) == 1

    def test_non_zero_exit_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ccp.subprocess,
            "run",
            MagicMock(
                return_value=_fake_completed_process(
                    stdout="", stderr="auth failed", returncode=2
                )
            ),
        )
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        llm = ClaudeCodeLLM()

        with pytest.raises(ClaudeCodeCLIError, match="stream failed"):
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_timeout_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ccp.subprocess,
            "run",
            MagicMock(side_effect=subprocess.TimeoutExpired(cmd=["claude"], timeout=1)),
        )
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        llm = ClaudeCodeLLM(timeout_seconds=1)

        with pytest.raises(ClaudeCodeCLIError, match="timed out"):
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_malformed_stream_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-NDJSON garbage on stdout (exit 0) is treated as a failed call —
        the stream parser sees no terminal event and marks success=False."""
        monkeypatch.setattr(
            ccp.subprocess,
            "run",
            MagicMock(return_value=_fake_completed_process(stdout="not json at all")),
        )
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        llm = ClaudeCodeLLM()

        with pytest.raises(ClaudeCodeCLIError, match="stream failed"):
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_cli_missing_raises_on_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: None)
        llm = ClaudeCodeLLM()

        with pytest.raises(ClaudeCodeCLIError, match="not found"):
            llm.completion(
                model="claude-code/default",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_positional_model_response_arg_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LiteLLM sometimes passes extra positional args; they must be tolerated."""
        self._mock_cli(monkeypatch)
        llm = ClaudeCodeLLM()

        response = llm.completion(
            "some-positional-model-response-arg",
            model="claude-code/default",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert response.choices[0].message.content == "ok"  # type: ignore[union-attr]

    def test_codex_host_uses_codex_exec(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Codex runs generation through ``codex exec`` instead of Claude flags."""

        def fake_run(cmd, **kwargs):
            out = Path(cmd[cmd.index("--output-last-message") + 1])
            out.write_text("codex reply", encoding="utf-8")
            return _fake_completed_process(stdout="", returncode=0)

        mock_run = MagicMock(side_effect=fake_run)
        monkeypatch.setenv("CLAUDE_SMART_HOST", "codex")
        monkeypatch.setattr(ccp.subprocess, "run", mock_run)
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/codex")

        response = ClaudeCodeLLM().completion(
            model="claude-code/default",
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "ping"},
            ],
        )

        cmd = mock_run.call_args.args[0]
        assert cmd[:2] == ["/usr/local/bin/codex", "exec"]
        assert "-p" not in cmd
        assert "--append-system-prompt" not in cmd
        assert mock_run.call_args.kwargs["input"] == "Be terse.\n\n## Task\nUser: ping"
        assert mock_run.call_args.kwargs["env"]["CLAUDE_SMART_HOST"] == "codex"
        assert response.choices[0].message.content == "codex reply"  # type: ignore[union-attr]


class TestIsClaudeCodeAvailable:
    def test_requires_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        assert is_claude_code_available() is False

    def test_requires_cli_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: None)
        assert is_claude_code_available() is False

    def test_both_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        assert is_claude_code_available() is True

    def test_respects_cli_path_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """An executable at CLAUDE_SMART_CLI_PATH should be honoured."""
        fake_cli = tmp_path / "claude"
        fake_cli.write_text("#!/bin/sh\necho hi\n")
        fake_cli.chmod(0o755)
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(fake_cli))
        # Force PATH lookup to fail so the override is what matters.
        monkeypatch.setattr(ccp.shutil, "which", lambda _: None)
        assert is_claude_code_available() is True

    def test_codex_host_resolves_codex_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setenv("CLAUDE_SMART_HOST", "codex")
        monkeypatch.delenv("CLAUDE_SMART_CLI_PATH", raising=False)
        monkeypatch.delenv("PLUGIN_ROOT", raising=False)
        monkeypatch.delenv("CLAUDE_PLUGIN_ROOT", raising=False)
        monkeypatch.setattr(ccp.shutil, "which", lambda name: f"/usr/local/bin/{name}")

        assert is_claude_code_available() is True
        assert ccp._resolve_cli_path() == "/usr/local/bin/codex"  # noqa: SLF001


class TestRegisterIfEnabled:
    def test_disabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_CLI", raising=False)
        assert register_if_enabled() is False

    def test_enabled_but_cli_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: None)
        assert register_if_enabled() is False

    def test_enabled_with_cli_registers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        with patch.object(ccp.litellm, "custom_provider_map", None):
            assert register_if_enabled() is True
            providers = [entry["provider"] for entry in ccp.litellm.custom_provider_map]
            assert "claude-code" in providers

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        with patch.object(ccp.litellm, "custom_provider_map", None):
            register_if_enabled()
            register_if_enabled()
            providers = [entry["provider"] for entry in ccp.litellm.custom_provider_map]
            assert providers.count("claude-code") == 1

    def test_no_duplicate_when_preexisting_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If something else already registered claude-code, don't add a second."""
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(ccp, "_resolve_cli_path", lambda: "/usr/local/bin/claude")
        existing_handler = ClaudeCodeLLM()
        preexisting = [{"provider": "claude-code", "custom_handler": existing_handler}]
        with patch.object(ccp.litellm, "custom_provider_map", preexisting):
            assert register_if_enabled() is True
            entries = [
                e
                for e in ccp.litellm.custom_provider_map
                if e.get("provider") == "claude-code"
            ]
            assert len(entries) == 1
            assert entries[0]["custom_handler"] is existing_handler
