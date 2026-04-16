"""Tests for the first-run LLM wizard in ``services start``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from reflexio.cli.commands.services import _ensure_llm_configured


class TestEnsureLlmConfigured:
    """Covers the three branches of ``_ensure_llm_configured``:

    already configured, interactive first-run, and non-interactive first-run.
    The helper is what keeps a fresh ``pip install reflexio-ai`` from crashing
    inside uvicorn's lifespan when no LLM key is set in ``~/.reflexio/.env``.
    """

    def test_returns_silently_when_embedding_provider_present(
        self, tmp_path: Path
    ) -> None:
        """If an embedding-capable provider is available, no prompt fires."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["openai"],
            ),
            patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider"
            ) as mock_emb,
        ):
            _ensure_llm_configured(env)
        mock_llm.assert_not_called()
        mock_emb.assert_not_called()

    def test_returns_silently_when_mixed_providers_include_embedding(
        self, tmp_path: Path
    ) -> None:
        """A provider set with at least one embedding-capable entry must short-circuit."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["openai", "anthropic"],
            ),
            patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider"
            ) as mock_emb,
        ):
            _ensure_llm_configured(env)
        mock_llm.assert_not_called()
        mock_emb.assert_not_called()

    def test_prompts_when_no_providers_and_tty(self, tmp_path: Path) -> None:
        """No keys + interactive stdin → both wizard helpers run, env reloads with override."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=[],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_llm_provider",
                return_value=("OpenAI", "gpt-5-mini", "openai"),
            ) as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider",
                return_value=None,
            ) as mock_emb,
            patch("dotenv.load_dotenv") as mock_load,
        ):
            _ensure_llm_configured(env)
        mock_llm.assert_called_once_with(env)
        mock_emb.assert_called_once_with(env, "openai")
        mock_load.assert_called_once_with(dotenv_path=env, override=True)

    def test_exits_cleanly_when_no_providers_and_non_tty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No keys + non-interactive stdin → friendly error, exit 1, no prompts."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=[],
            ),
            patch("sys.stdin.isatty", return_value=False),
            patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as mock_llm,
            pytest.raises(typer.Exit) as exc_info,
        ):
            _ensure_llm_configured(env)
        assert exc_info.value.exit_code == 1
        mock_llm.assert_not_called()
        out = capsys.readouterr().out
        assert "not fully configured" in out
        assert str(env) in out
        assert "reflexio setup init" in out

    def test_prompts_only_for_embedding_when_llm_exists_without_embedding(
        self, tmp_path: Path
    ) -> None:
        """Anthropic-only env → skip LLM prompt, run embedding prompt with anthropic."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["anthropic"],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider",
                return_value="OpenAI",
            ) as mock_emb,
            patch("dotenv.load_dotenv"),
        ):
            _ensure_llm_configured(env)
        mock_llm.assert_not_called()
        mock_emb.assert_called_once_with(env, "anthropic")
