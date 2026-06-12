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
        """No keys + interactive stdin → both wizard helpers run, env reloads with override=True (file wins)."""
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
                return_value=("OpenAI", "gpt-5.4-mini", "openai"),
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

    def test_wizard_key_reaches_os_environ_after_reload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key the wizard writes to the .env FILE must win over the shipped
        empty placeholder already in ``os.environ`` after the reload.

        This is the F002 regression guard: with ``override=False`` the empty
        ``OPENAI_API_KEY`` placeholder would shadow the file value and the key
        the operator just entered would be lost. ``load_dotenv`` runs for real
        here (only the prompt helpers are mocked) so we exercise the actual
        file-wins behaviour.
        """
        env = tmp_path / ".env"

        # A fresh install ships an empty placeholder already present in the
        # process env; that is exactly what must NOT shadow the file value.
        monkeypatch.setenv("OPENAI_API_KEY", "")

        def _write_real_key(_env_path: Path) -> tuple[str, str, str]:
            # Stand in for the wizard's set_env_var: write to the FILE only.
            env.write_text("OPENAI_API_KEY=sk-real-entered-key\n")
            return ("OpenAI", "gpt-5.4-mini", "openai")

        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=[],
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_llm_provider",
                side_effect=_write_real_key,
            ),
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider",
                return_value=None,
            ),
        ):
            _ensure_llm_configured(env)

        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-real-entered-key"

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
        """Anthropic-only env + no chromadb → embedding prompt fires.

        With chromadb available the helper takes the local-fallback path and
        skips the prompt entirely (covered by
        ``test_services_start_proceeds_without_cloud_embedder_when_chromadb_present``).
        """
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["anthropic"],
            ),
            patch(
                "reflexio.server.llm.providers.local_embedding_provider"
                ".is_chromadb_importable",
                return_value=False,
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

    def test_local_only_provider_still_triggers_llm_wizard(
        self, tmp_path: Path
    ) -> None:
        """``providers == ["local"]`` (embedder-only) must NOT skip the LLM wizard.

        ``detect_available_providers()`` can legally surface only the local
        ONNX embedder when chromadb is importable but no LLM key is set.
        That state has ``has_embedding=True`` but no generation provider,
        so the request still has nothing to drive extraction with — we
        must prompt for an LLM provider rather than falling through.
        """
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["local"],
            ),
            patch(
                "reflexio.server.llm.providers.local_embedding_provider"
                ".is_chromadb_importable",
                return_value=True,
            ),
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_llm_provider",
                return_value=("OpenAI", "gpt-5.4-mini", "openai"),
            ) as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider",
                return_value=None,
            ) as mock_emb,
            patch("dotenv.load_dotenv"),
        ):
            _ensure_llm_configured(env)
        # Wizard ran — we did not silently fall through on the embedder-only state.
        mock_llm.assert_called_once_with(env)
        mock_emb.assert_called_once_with(env, "openai")

    def test_local_only_provider_non_tty_exits_with_generation_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Embedder-only providers in non-TTY mode → exit with a message naming the missing role."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["local"],
            ),
            patch(
                "reflexio.server.llm.providers.local_embedding_provider"
                ".is_chromadb_importable",
                return_value=True,
            ),
            patch("sys.stdin.isatty", return_value=False),
            pytest.raises(typer.Exit) as exc_info,
        ):
            _ensure_llm_configured(env)
        assert exc_info.value.exit_code == 1
        out = capsys.readouterr().out
        assert "no generation-capable LLM API key" in out

    def test_services_start_proceeds_without_cloud_embedder_when_chromadb_present(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Anthropic-only env + chromadb importable → no prompt, no exit.

        The runtime auto-detection (Layer A path 3) picks the local embedder,
        so ``services start`` should log the fallback note and continue
        without involving the user.
        """
        env = tmp_path / ".env"
        env.write_text("")
        import logging

        with (
            patch(
                "reflexio.server.llm.model_defaults.detect_available_providers",
                return_value=["anthropic"],
            ),
            patch(
                "reflexio.server.llm.providers.local_embedding_provider"
                ".is_chromadb_importable",
                return_value=True,
            ),
            patch("reflexio.cli.commands.setup_cmd._prompt_llm_provider") as mock_llm,
            patch(
                "reflexio.cli.commands.setup_cmd._prompt_embedding_provider"
            ) as mock_emb,
            caplog.at_level(logging.INFO, logger="reflexio.cli.commands.services"),
        ):
            # Should return cleanly; no prompts, no exit.
            _ensure_llm_configured(env)

        mock_llm.assert_not_called()
        mock_emb.assert_not_called()
        assert any(
            "Using local embedder as fallback" in record.message
            for record in caplog.records
        )


def test_embedding_only_start_skips_first_run_llm_guard(monkeypatch) -> None:
    from reflexio.cli.commands import services

    called_args = None

    def _capture_execute(args) -> None:
        nonlocal called_args
        called_args = args

    monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
    monkeypatch.setattr("reflexio.cli.env_loader.load_reflexio_env", lambda: None)
    monkeypatch.setattr(
        "reflexio.cli.bootstrap_config.resolve_storage", lambda _: "sqlite"
    )
    monkeypatch.setattr(services, "_ensure_llm_configured", lambda _: pytest.fail())
    monkeypatch.setattr(services.run_mod, "execute", _capture_execute)

    services.start(only="embedding")

    assert called_args is not None
    assert called_args.only == "embedding"
