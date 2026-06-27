"""Unit tests for setup_cmd helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from reflexio.cli.commands.setup_cmd import (
    _prompt_storage,
    _set_env_var,
    _write_embedding_model_to_org_config,
)
from reflexio.models.api_schema.service_schemas import WhoamiResponse


class TestSetEnvVar:
    """Tests for _set_env_var: new key, existing key, commented key, quoting."""

    def test_new_key_appended(self, tmp_path: Path) -> None:
        """A brand-new key is appended to an empty file."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "MY_KEY", "my_value")
        assert 'MY_KEY="my_value"' in env.read_text()

    def test_new_key_creates_file(self, tmp_path: Path) -> None:
        """If the .env file does not exist, it is created."""
        env = tmp_path / ".env"
        _set_env_var(env, "NEW_KEY", "val")
        assert env.exists()
        assert 'NEW_KEY="val"' in env.read_text()

    def test_existing_key_replaced(self, tmp_path: Path) -> None:
        """An active KEY=old line is replaced in-place."""
        env = tmp_path / ".env"
        env.write_text("OTHER=1\nAPI_KEY=old\nANOTHER=2\n")
        _set_env_var(env, "API_KEY", "new")
        lines = env.read_text().splitlines()
        assert lines[0] == "OTHER=1"
        assert lines[1] == 'API_KEY="new"'
        assert lines[2] == "ANOTHER=2"

    def test_commented_key_replaced(self, tmp_path: Path) -> None:
        """A commented-out # KEY=... line is replaced when no active line exists."""
        env = tmp_path / ".env"
        env.write_text("# API_KEY=old_value\n")
        _set_env_var(env, "API_KEY", "new_value")
        content = env.read_text()
        assert 'API_KEY="new_value"' in content
        assert "# API_KEY" not in content

    def test_active_preferred_over_commented(self, tmp_path: Path) -> None:
        """When both commented and active lines exist, the active one is updated."""
        env = tmp_path / ".env"
        env.write_text("# API_KEY=commented\nAPI_KEY=active\n")
        _set_env_var(env, "API_KEY", "updated")
        lines = env.read_text().splitlines()
        assert lines[0] == "# API_KEY=commented"
        assert lines[1] == 'API_KEY="updated"'

    def test_value_with_equals_sign_quoted(self, tmp_path: Path) -> None:
        """Values containing '=' are safely quoted."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "TOKEN", "abc=def=ghi")
        assert 'TOKEN="abc=def=ghi"' in env.read_text()

    def test_value_with_hash_quoted(self, tmp_path: Path) -> None:
        """Values containing '#' are safely quoted."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "TOKEN", "abc#comment")
        assert 'TOKEN="abc#comment"' in env.read_text()

    def test_file_permissions_restricted(self, tmp_path: Path) -> None:
        """After writing, the .env file should have mode 0o600."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "SECRET", "s3cret")
        mode = env.stat().st_mode & 0o777
        assert mode == 0o600

    def test_value_with_double_quotes(self, tmp_path: Path) -> None:
        """Double quotes in values are escaped to prevent .env breakage."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "KEY", 'val"ue')
        assert 'KEY="val\\"ue"' in env.read_text()

    def test_value_with_backslash(self, tmp_path: Path) -> None:
        """Backslashes in values are escaped before double-quote escaping."""
        env = tmp_path / ".env"
        env.write_text("")
        _set_env_var(env, "KEY", "val\\ue")
        assert 'KEY="val\\\\ue"' in env.read_text()

    def test_commented_with_spaces(self, tmp_path: Path) -> None:
        """Commented lines with extra spaces like '#  KEY=' are matched."""
        env = tmp_path / ".env"
        env.write_text("#  MY_KEY=old\n")
        _set_env_var(env, "MY_KEY", "new")
        content = env.read_text()
        assert 'MY_KEY="new"' in content
        assert "#" not in content.strip()


# ---------------------------------------------------------------------------
# _prompt_storage — the 3-option storage picker
# ---------------------------------------------------------------------------


class TestPromptStorage:
    """Covers the local/cloud/self-host branches of ``_prompt_storage``.

    Uses ``typer.prompt`` / ``typer.confirm`` patches because Typer's
    own CliRunner is heavyweight for this helper — we only care about
    the control flow and the resulting .env state.
    """

    def test_option_1_local_sqlite(self, tmp_path: Path) -> None:
        """Option 1 returns the SQLite label and writes REFLEXIO_URL."""
        env = tmp_path / ".env"
        env.write_text("")
        with patch("typer.prompt", return_value=1):
            label = _prompt_storage(env)
        assert label == "SQLite (local)"
        assert 'REFLEXIO_URL="http://localhost:8061"' in env.read_text()

    def test_option_2_cloud_writes_reflexio_url_and_api_key(
        self, tmp_path: Path
    ) -> None:
        """Option 2 writes REFLEXIO_URL + REFLEXIO_API_KEY and calls whoami()."""
        env = tmp_path / ".env"
        env.write_text("")

        # typer.prompt is called twice: once for the storage choice,
        # once for the API key. Mock them in order.
        prompts = [2, "rflx-test-key-123"]
        mock_client = MagicMock()
        mock_client.whoami.return_value = WhoamiResponse(
            success=True,
            org_id="42",
            storage_type="supabase",
            storage_label="https://jpkj...supabase.co",
            storage_configured=True,
        )

        with (
            patch("typer.prompt", side_effect=prompts),
            patch("reflexio.client.client.ReflexioClient", return_value=mock_client),
        ):
            label = _prompt_storage(env)

        assert label == "Managed Reflexio"
        content = env.read_text()
        assert 'REFLEXIO_URL="https://www.reflexio.ai"' in content
        assert 'REFLEXIO_API_KEY="rflx-test-key-123"' in content
        # No Supabase creds leaked into .env for the cloud path
        assert "SUPABASE_URL" not in content

    def test_option_2_whoami_failure_still_writes_env(self, tmp_path: Path) -> None:
        """A whoami() crash must not corrupt the wizard — env vars stay."""
        env = tmp_path / ".env"
        env.write_text("")

        mock_client = MagicMock()
        mock_client.whoami.side_effect = RuntimeError("network down")

        with (
            patch("typer.prompt", side_effect=[2, "rflx-key"]),
            patch("reflexio.client.client.ReflexioClient", return_value=mock_client),
        ):
            label = _prompt_storage(env)

        assert label == "Managed Reflexio"
        assert 'REFLEXIO_URL="https://www.reflexio.ai"' in env.read_text()

    def test_option_2_unconfigured_warns_but_succeeds(self, tmp_path: Path) -> None:
        """If the org has no storage configured, the wizard warns but finishes."""
        env = tmp_path / ".env"
        env.write_text("")

        mock_client = MagicMock()
        mock_client.whoami.return_value = WhoamiResponse(
            success=True,
            org_id="42",
            storage_type=None,
            storage_label=None,
            storage_configured=False,
        )

        with (
            patch("typer.prompt", side_effect=[2, "rflx-key"]),
            patch("reflexio.client.client.ReflexioClient", return_value=mock_client),
        ):
            label = _prompt_storage(env)

        assert label == "Managed Reflexio"

    def test_option_3_self_hosted_writes_url_and_api_key(self, tmp_path: Path) -> None:
        """Self-hosted prompts for URL (with localhost default) and API key."""
        env = tmp_path / ".env"
        env.write_text("")

        # typer.prompt is called three times: storage choice, URL, API key
        prompts = [3, "http://localhost:8081", "rflx-self-key"]
        with patch("typer.prompt", side_effect=prompts):
            label = _prompt_storage(env)

        assert label == "Self-hosted Reflexio"
        content = env.read_text()
        assert 'REFLEXIO_URL="http://localhost:8081"' in content
        assert 'REFLEXIO_API_KEY="rflx-self-key"' in content
        # No Supabase creds — self-hosted no longer asks for them
        assert "SUPABASE_URL" not in content

    def test_invalid_choice_exits(self, tmp_path: Path) -> None:
        """Choices outside 1/2/3 raise typer.Exit."""
        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch("typer.prompt", return_value=9),
            pytest.raises(typer.Exit),
        ):
            _prompt_storage(env)


# ---------------------------------------------------------------------------
# Embedding-provider step (Layer B): non-TTY behaviour, --embedding flag,
# org-config persistence
# ---------------------------------------------------------------------------


def _read_org_config(home: Path, org_id: str = "self-host-org") -> dict | None:
    """Read the JSON config file for an org under a fake ``$HOME``.

    Returns None when the file doesn't exist so tests can assert
    "no override was written" cleanly.
    """
    config_path = home / ".reflexio" / "configs" / f"config_{org_id}.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text())


class TestPromptEmbeddingProviderNonInteractive:
    """``_prompt_embedding_provider`` short-circuits when stdin is not a TTY."""

    def test_prompt_embedding_provider_non_tty_picks_local(
        self, tmp_path: Path
    ) -> None:
        """No TTY → return local default without ever calling ``typer.prompt``."""
        from reflexio.cli.commands.setup_cmd import _prompt_embedding_provider

        env = tmp_path / ".env"
        env.write_text("")
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch(
                "reflexio.server.llm.providers.local_embedding_provider"
                ".is_chromadb_importable",
                return_value=True,
            ),
            patch("typer.prompt") as mock_prompt,
        ):
            # 'anthropic' has no embedding support, so the prompt path runs.
            result = _prompt_embedding_provider(env, "anthropic")
        mock_prompt.assert_not_called()
        # Local was the first choice when chromadb is importable.
        assert result == "Local (MiniLM-L6-v2)"


class TestSetupInitEmbeddingStep:
    """``setup init`` includes the new embedding step and writes org config."""

    def _patch_home_and_chromadb(
        self, monkeypatch: pytest.MonkeyPatch, fake_home: Path
    ) -> None:
        """Redirect ``~`` to a tmp dir and force chromadb to look importable.

        ``LocalFileConfigStorage`` resolves its config path through
        ``Path.home() / ".reflexio" / "configs"`` when ``base_dir`` is None,
        so this is enough to keep the test from touching the real home dir.
        """
        fake_home.mkdir(parents=True, exist_ok=True)
        # Clear REFLEXIO_LOG_DIR — the OSS conftest sets it session-wide to
        # protect tests that don't patch home, but here we want `reflexio_home()`
        # to fall through to the patched `Path.home()`.
        monkeypatch.delenv("REFLEXIO_LOG_DIR", raising=False)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        # ``setup init`` queries chromadb importability via the module-level
        # helper. Force True so the local option appears as choice [1].
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".is_chromadb_importable",
            lambda: True,
        )

    def test_setup_init_includes_embedding_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wizard renders the new "Choose embedding provider:" prompt."""
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        self._patch_home_and_chromadb(monkeypatch, fake_home)
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        captured: list[str] = []

        def _echo(msg: str = "", *_: object, **__: object) -> None:
            captured.append(msg)

        # Prompts in order:
        #   1. Storage choice → 1 (SQLite)
        #   2. LLM provider → 2 (Anthropic — no embedding support)
        #   3. LLM API key
        #   4. New upfront embedding step in ``_choose_embedding_provider`` → 1
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("typer.prompt", side_effect=[1, 2, "sk-ant-test", 1]),
            patch("typer.echo", side_effect=_echo),
        ):
            init(skip_llm=False, embedding="auto")

        assert any("Choose embedding provider:" in line for line in captured)

    def test_setup_init_local_is_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pressing Enter at the embedding prompt picks local (chromadb available)."""
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        self._patch_home_and_chromadb(monkeypatch, fake_home)
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        # Storage 1, LLM provider 2, API key, embedding step (Enter → default
        # 1). typer.prompt with ``default=1`` returns 1 when the user presses
        # Enter — we model that by feeding the default value as the response.
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("typer.prompt", side_effect=[1, 2, "sk-ant-test", 1]),
            patch("typer.echo"),
        ):
            init(skip_llm=False, embedding="auto")

        cfg = _read_org_config(fake_home)
        assert cfg is not None
        assert cfg["llm_config"]["embedding_model_name"] == "local/minilm-l6-v2"

    def test_setup_init_local_choice_writes_org_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Picking local writes ``embedding_model_name`` to the org config file."""
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        self._patch_home_and_chromadb(monkeypatch, fake_home)
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("typer.prompt", side_effect=[1, 2, "sk-ant-test", 1]),
            patch("typer.echo"),
        ):
            init(skip_llm=False, embedding="auto")

        cfg_path = fake_home / ".reflexio" / "configs" / "config_self-host-org.json"
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg["llm_config"]["embedding_model_name"] == "local/minilm-l6-v2"

    def test_setup_init_embedding_flag_local(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--embedding local`` writes org config without prompting."""
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        self._patch_home_and_chromadb(monkeypatch, fake_home)
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        # Only storage + LLM provider prompts should fire — no embedding step.
        # We supply exactly 3 responses; if the wizard asked a 4th time
        # (i.e. the embedding step ran), ``side_effect`` would raise
        # ``StopIteration`` and the test would fail loudly.
        prompt_mock = patch("typer.prompt", side_effect=[1, 2, "sk-ant-test"])
        with (
            patch("sys.stdin.isatty", return_value=True),
            prompt_mock as mp,
            patch("typer.echo"),
        ):
            init(skip_llm=False, embedding="local")

        cfg = _read_org_config(fake_home)
        assert cfg is not None
        assert cfg["llm_config"]["embedding_model_name"] == "local/minilm-l6-v2"
        # Only 3 prompts: storage, LLM provider, API key.
        assert mp.call_count == 3

    def test_setup_init_embedding_flag_auto_no_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--embedding auto`` doesn't write ``embedding_model_name`` (non-TTY).

        Auto-detection at runtime decides; setup-init stays out of the way.
        """
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        self._patch_home_and_chromadb(monkeypatch, fake_home)
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        # Non-TTY path: --embedding auto → no embedding-step prompt, no write.
        with (
            patch("sys.stdin.isatty", return_value=False),
            patch("typer.prompt", side_effect=[1, 2, "sk-ant-test"]),
            patch("typer.echo"),
        ):
            init(skip_llm=False, embedding="auto")

        cfg = _read_org_config(fake_home)
        # Either the file doesn't exist or llm_config is unset / has no
        # embedding override — none of those count as "wrote an override".
        if cfg is not None:
            assert (
                cfg.get("llm_config") is None
                or cfg["llm_config"].get("embedding_model_name") is None
            )


class TestChooseEmbeddingProviderEdgeCases:
    """Hardening cases for ``_choose_embedding_provider``: chromadb gating,
    config-write ordering, and integration-command embedding-flag validation.
    """

    def test_local_flag_without_chromadb_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--embedding=local`` without chromadb fails fast and writes nothing."""
        from reflexio.cli.commands.setup_cmd import _choose_embedding_provider

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".is_chromadb_importable",
            lambda: False,
        )
        env_path = tmp_path / ".env"
        env_path.write_text("")

        with pytest.raises(typer.Exit) as exc:
            _choose_embedding_provider(env_path, embedding_flag="local")
        assert exc.value.exit_code == 1
        # No org-config file written for the broken override.
        assert _read_org_config(fake_home) is None

    def test_blank_cloud_key_does_not_persist_org_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Submitting an empty API key for OpenAI/Gemini must not mutate org config."""
        from reflexio.cli.commands.setup_cmd import _choose_embedding_provider

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".is_chromadb_importable",
            lambda: True,
        )
        # Strip any existing OPENAI_API_KEY so the prompt path runs.
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        env_path = tmp_path / ".env"
        env_path.write_text("")

        # Choices when chromadb is importable: [local, openai, gemini].
        # Pick choice 2 (openai), then submit an empty key.
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("typer.prompt", side_effect=[2, "   "]),
            patch("typer.echo"),
            pytest.raises(typer.Exit),
        ):
            _choose_embedding_provider(env_path, embedding_flag="auto")

        # No org-config file written when the key validation rejects.
        assert _read_org_config(fake_home) is None

    def test_setup_init_self_hosted_skips_embedding_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``setup init`` with Self-hosted Reflexio storage must NOT run the embedding step.

        The remote server owns its embedding config; a local override
        would just shadow whatever the operator picked there.
        """
        from reflexio.cli.commands.setup_cmd import init

        fake_home = tmp_path / "home"
        fake_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".is_chromadb_importable",
            lambda: True,
        )
        env_path = fake_home / ".reflexio" / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("")
        monkeypatch.setattr(
            "reflexio.cli.env_loader.ensure_user_env_for_setup",
            lambda: env_path,
        )

        # Storage choice 3 (Self-hosted), default URL accept, API key.
        # If the embedding step ran, side_effect would need a 4th value.
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(
                "typer.prompt",
                side_effect=[3, "http://localhost:8081", "rflx-key", 1],
            ),
            patch("typer.echo"),
        ):
            init(skip_llm=True, embedding="auto")

        # No embedding override persisted to org config.
        cfg = _read_org_config(fake_home)
        if cfg is not None:
            assert (
                cfg.get("llm_config") is None
                or cfg["llm_config"].get("embedding_model_name") is None
            )

    def test_openclaw_invalid_embedding_flag_exits(self) -> None:
        """``setup openclaw --embedding=opneai`` must fail fast on the typo."""
        from reflexio.cli.commands.setup_cmd import openclaw

        with pytest.raises(typer.Exit):
            openclaw(uninstall=False, embedding="opneai")


class TestOpenclawSetup:
    """Tests for the openclaw-smart install/uninstall/repair flows."""

    def test_plugin_id_constant(self) -> None:
        """The plugin id must match what openClaw and the TS shim register."""
        from reflexio.cli.commands.setup_cmd import _OPENCLAW_PLUGIN_ID

        assert _OPENCLAW_PLUGIN_ID == "reflexio-openclaw-smart"

    def test_write_openclaw_env_persists_keys(self, tmp_path: Path) -> None:
        """``_write_openclaw_env`` upserts OPENCLAW_BIN + USE_LOCAL_CLI=1."""
        from reflexio.cli.commands.setup_cmd import _write_openclaw_env

        env = tmp_path / ".env"
        _write_openclaw_env(env, "/usr/local/bin/openclaw")
        body = env.read_text()
        assert 'OPENCLAW_BIN="/usr/local/bin/openclaw"' in body
        assert 'OPENCLAW_SMART_USE_LOCAL_CLI="1"' in body

    def test_remove_env_keys_strips_lines(self, tmp_path: Path) -> None:
        """``_remove_env_keys`` drops the named keys, leaves others untouched."""
        from reflexio.cli.commands.setup_cmd import _remove_env_keys

        env = tmp_path / ".env"
        env.write_text(
            "OTHER=keep\n"
            'OPENCLAW_BIN="/usr/local/bin/openclaw"\n'
            'OPENCLAW_SMART_USE_LOCAL_CLI="1"\n'
            "STILL=keep\n"
        )
        _remove_env_keys(env, ("OPENCLAW_BIN", "OPENCLAW_SMART_USE_LOCAL_CLI"))
        remaining = env.read_text().splitlines()
        assert remaining == ["OTHER=keep", "STILL=keep"]

    def test_openclaw_rejects_conflicting_flags(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Repair, uninstall, and purge modes must be unambiguous."""
        from reflexio.cli import env_loader
        from reflexio.cli.commands.setup_cmd import openclaw

        env = tmp_path / ".env"
        env.write_text("")
        monkeypatch.setattr(env_loader, "ensure_user_env_for_setup", lambda: env)

        with pytest.raises(typer.Exit):
            openclaw(repair=True, uninstall=True, purge=False)
        with pytest.raises(typer.Exit):
            openclaw(repair=False, uninstall=False, purge=True)

    def test_install_openclaw_uses_new_plugin_id(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_install_openclaw_integration` passes the new plugin id to subprocess."""
        from reflexio.cli.commands import setup_cmd

        env_path = tmp_path / ".env"
        env_path.write_text("")

        monkeypatch.setattr(setup_cmd.shutil, "which", lambda _: "/usr/bin/openclaw")
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()
        monkeypatch.setattr(setup_cmd, "_openclaw_plugin_dir", lambda: plugin_dir)
        monkeypatch.setattr(setup_cmd, "_run_smart_install", lambda _p: None)

        calls: list[list[str]] = []

        class _Result:
            def __init__(self, stdout: str = "Status: loaded\n") -> None:
                self.stdout = stdout
                self.stderr = ""
                self.returncode = 0

        def fake_run(argv, **_kw):  # noqa: ANN001
            calls.append(list(argv))
            # CLI is invoked by absolute path (TOCTOU fix), so check the
            # subcommand position rather than the executable string.
            if argv[1:3] == ["plugins", "inspect"]:
                return _Result("Status: loaded\n")
            return _Result("")

        monkeypatch.setattr(setup_cmd.subprocess, "run", fake_run)

        ok = setup_cmd._install_openclaw_integration(env_path)

        assert ok is True
        flat_args = [arg for call in calls for arg in call]
        assert "reflexio-openclaw-smart" in flat_args
        assert "reflexio-federated" not in flat_args
        # Every install-side call should target the absolute openclaw_bin
        # path, not the bare "openclaw" string.
        for call in calls:
            assert call[0] == "/usr/bin/openclaw", (
                f"expected absolute CLI path, got {call[0]!r}"
            )
        body = env_path.read_text()
        assert 'OPENCLAW_BIN="/usr/bin/openclaw"' in body
        assert 'OPENCLAW_SMART_USE_LOCAL_CLI="1"' in body

    def test_install_openclaw_fails_if_conversation_access_not_persisted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setup must not report success if typed-hook access cannot be saved."""
        from reflexio.cli.commands import setup_cmd

        env_path = tmp_path / ".env"
        env_path.write_text("")
        plugin_dir = tmp_path / "plugin"
        plugin_dir.mkdir()

        monkeypatch.setattr(setup_cmd.shutil, "which", lambda _: "/usr/bin/openclaw")
        monkeypatch.setattr(setup_cmd, "_openclaw_plugin_dir", lambda: plugin_dir)

        class _Result:
            def __init__(self, returncode: int = 0) -> None:
                self.stdout = ""
                self.stderr = "denied"
                self.returncode = returncode

        def fake_run(argv, **_kw):  # noqa: ANN001
            if argv[1:3] == ["config", "set"]:
                return _Result(returncode=1)
            return _Result()

        monkeypatch.setattr(setup_cmd.subprocess, "run", fake_run)

        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(env_path)

    def test_uninstall_openclaw_reuses_openclaw_bin_from_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uninstall should remove the plugin even if PATH no longer has openclaw."""
        from reflexio.cli.commands import setup_cmd

        env_path = tmp_path / ".env"
        env_path.write_text('OPENCLAW_BIN="/opt/openclaw/bin/openclaw"\n')
        monkeypatch.setattr(setup_cmd.typer, "confirm", lambda *_a, **_kw: True)
        monkeypatch.setattr(setup_cmd.shutil, "which", lambda _: None)

        calls: list[list[str]] = []

        def fake_run(argv, **_kw):  # noqa: ANN001
            calls.append(list(argv))

            class _Result:
                returncode = 0
                stdout = ""
                stderr = ""

            return _Result()

        monkeypatch.setattr(setup_cmd.subprocess, "run", fake_run)

        setup_cmd._uninstall_openclaw(env_path=env_path, purge=False)

        assert calls
        assert all(call[0] == "/opt/openclaw/bin/openclaw" for call in calls)


class TestEnsureUserEnvForSetup:
    """Regression tests for the user-level .env target.

    ``setup init`` must always write to ``~/.reflexio/.env`` even when
    invoked from a directory that already contains its own ``.env``
    (e.g. a worktree root or unrelated project).
    """

    def test_setup_init_writes_to_user_home_not_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When CWD has an unrelated ``.env``, the new key lands in
        ``~/.reflexio/.env`` and the CWD ``.env`` is left untouched."""
        from reflexio.cli.env_loader import ensure_user_env_for_setup

        fake_home = tmp_path / "home"
        fake_cwd = tmp_path / "worktree"
        fake_cwd.mkdir(parents=True)
        cwd_env = fake_cwd / ".env"
        cwd_env.write_text("UNRELATED_VAR=do-not-touch\n")

        # Pre-create ~/.reflexio/.env so we don't depend on the bundled template.
        user_env = fake_home / ".reflexio" / ".env"
        user_env.parent.mkdir(parents=True, exist_ok=True)
        user_env.write_text("")

        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
        monkeypatch.chdir(fake_cwd)
        # Re-bind module-level constants that captured Path.home() at import time.
        monkeypatch.setattr(
            "reflexio.cli.env_loader._USER_ENV_DIR", fake_home / ".reflexio"
        )
        monkeypatch.setattr("reflexio.cli.env_loader._USER_ENV_FILE", user_env)

        resolved = ensure_user_env_for_setup()

        assert resolved == user_env
        assert resolved is not None and resolved.parent == fake_home / ".reflexio"
        # CWD .env was not selected and was not modified.
        assert cwd_env.read_text() == "UNRELATED_VAR=do-not-touch\n"

    def test_setup_init_creates_user_env_when_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no ``~/.reflexio/.env`` exists yet, the function creates it
        from the bundled template — never falling back to CWD."""
        from reflexio.cli.env_loader import ensure_user_env_for_setup

        fake_home = tmp_path / "home"
        fake_cwd = tmp_path / "worktree"
        fake_cwd.mkdir(parents=True)
        cwd_env = fake_cwd / ".env"
        cwd_env.write_text("CWD_ONLY=ignored\n")

        monkeypatch.setattr("pathlib.Path.home", lambda: fake_home)
        monkeypatch.chdir(fake_cwd)
        monkeypatch.setattr(
            "reflexio.cli.env_loader._USER_ENV_DIR", fake_home / ".reflexio"
        )
        monkeypatch.setattr(
            "reflexio.cli.env_loader._USER_ENV_FILE",
            fake_home / ".reflexio" / ".env",
        )

        # Mock the template so the test doesn't rely on .env.example presence.
        monkeypatch.setattr(
            "reflexio.cli.env_loader._find_env_example",
            lambda *_args, **_kwargs: "REFLEXIO_URL=\n",
        )

        resolved = ensure_user_env_for_setup()

        assert resolved == fake_home / ".reflexio" / ".env"
        assert resolved is not None and resolved.exists()
        # CWD .env was untouched.
        assert cwd_env.read_text() == "CWD_ONLY=ignored\n"


class TestWriteEmbeddingModelToOrgConfig:
    """The embedding-choice writer must target the same org the running no-auth
    server resolves (``REFLEXIO_DEFAULT_ORG_ID``-aware), not a hardcoded
    ``self-host-org`` — otherwise the storage backend and the embedding model
    land in different ``config_<org>.json`` files."""

    _STORAGE_PATCH = (
        "reflexio.server.services.configurator."
        "local_file_config_storage.LocalFileConfigStorage"
    )

    def test_writes_to_env_resolved_org(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "claude-smart")
        with patch(self._STORAGE_PATCH) as mock_storage:
            _write_embedding_model_to_org_config("local/minilm-l6-v2")
        mock_storage.assert_called_once_with("claude-smart")
        mock_storage.return_value.save_config.assert_called_once()

    def test_defaults_to_self_host_org_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("REFLEXIO_DEFAULT_ORG_ID", raising=False)
        with patch(self._STORAGE_PATCH) as mock_storage:
            _write_embedding_model_to_org_config("local/minilm-l6-v2")
        mock_storage.assert_called_once_with("self-host-org")
