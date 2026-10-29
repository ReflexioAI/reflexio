"""End-to-end ``reflexio setup`` flows through the real Typer app.

Each test gets a temp ``$HOME`` (``Path.home`` + ``REFLEXIO_LOG_DIR``), so the
wizard writes a real ``~/.reflexio/.env`` and org config under ``tmp_path``.
Only process and network boundaries are faked: the ``openclaw``/``node``
binaries (``subprocess.run`` + ``shutil.which``), the managed-server
``whoami`` call, and the OAuth browser login.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from reflexio.cli import codex_auth, env_loader
from reflexio.cli.app import create_app
from reflexio.cli.commands import setup_cmd
from reflexio.models.api_schema.service_schemas import WhoamiResponse


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated ``$HOME`` with an empty ``~/.reflexio/.env``."""
    snapshot = dict(os.environ)
    fake_home = tmp_path / "home"
    reflexio_dir = fake_home / ".reflexio"
    reflexio_dir.mkdir(parents=True)
    (reflexio_dir / ".env").write_text("")
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("REFLEXIO_LOG_DIR", str(fake_home))
    monkeypatch.setattr(env_loader, "_USER_ENV_DIR", reflexio_dir)
    monkeypatch.setattr(env_loader, "_USER_ENV_FILE", reflexio_dir / ".env")
    for name in (
        "REFLEXIO_ENV_FILE",
        "REFLEXIO_NONINTERACTIVE",
        "REFLEXIO_URL",
        "REFLEXIO_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "REFLEXIO_ORG_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    yield fake_home
    os.environ.clear()
    os.environ.update(snapshot)


@pytest.fixture
def cli():
    runner = CliRunner()
    app = create_app()

    def invoke(*args: str, input: str | None = None):
        return runner.invoke(app, ["setup", *args], input=input)

    return invoke


@pytest.fixture
def interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    """CliRunner's stdin is never a TTY; treat the session as interactive."""
    monkeypatch.setattr(setup_cmd, "_is_non_interactive", lambda: False)


@pytest.fixture
def local_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reflexio.server.llm.providers.local_embedding_provider"
        ".are_local_embedding_dependencies_available",
        lambda: True,
    )


def _env(home: Path) -> str:
    return (home / ".reflexio" / ".env").read_text()


def _org_config(home: Path) -> dict | None:
    path = home / ".reflexio" / "configs" / "config_self-host-org.json"
    return json.loads(path.read_text()) if path.exists() else None


def _embedding_model(home: Path) -> str | None:
    config = _org_config(home)
    if config is None:
        return None
    return (config.get("llm_config") or {}).get("embedding_model_name")


def _whoami(monkeypatch: pytest.MonkeyPatch, response=None, error=None) -> MagicMock:
    client = MagicMock()
    if error is not None:
        client.whoami.side_effect = error
    else:
        client.whoami.return_value = response
    factory = MagicMock(return_value=client)
    monkeypatch.setattr("reflexio.client.client.ReflexioClient", factory)
    return factory


# ---------------------------------------------------------------------------
# setup init
# ---------------------------------------------------------------------------


class TestInit:
    def test_invalid_embedding_flag_is_rejected_before_any_write(self, home, cli):
        result = cli("init", "--embedding", "bogus")
        assert result.exit_code == 1
        assert "--embedding must be one of" in result.output
        assert _env(home) == ""

    def test_missing_env_file_and_template_exits(self, home, cli, monkeypatch):
        (home / ".reflexio" / ".env").unlink()
        monkeypatch.setattr(env_loader, "_find_env_example", lambda _pkg: None)
        result = cli("init")
        assert result.exit_code == 1
        assert "could not locate or create a .env file" in result.output

    def test_local_sqlite_with_anthropic_and_local_embeddings(
        self, home, cli, interactive, local_embeddings
    ):
        # storage=1 (SQLite), provider=2 (Anthropic), key, embedding=1 (local)
        result = cli("init", input="1\n2\nsk-ant-test\n1\n")
        assert result.exit_code == 0, result.output
        env = _env(home)
        assert 'REFLEXIO_URL="http://localhost:8061"' in env
        assert 'ANTHROPIC_API_KEY="sk-ant-test"' in env
        assert _embedding_model(home) == "local/minilm-l6-v2"
        assert "LLM Provider: Anthropic (claude-sonnet-5)" in result.output
        assert "Embedding Provider: Local (MiniLM-L6-v2)" in result.output
        assert "Storage: SQLite (local)" in result.output

    def test_invalid_llm_choice_exits_without_writing_a_key(self, home, cli):
        result = cli("init", "--embedding", "auto", input="1\n99\n")
        assert result.exit_code == 1
        assert "choice must be between 1 and" in result.output
        assert "API_KEY" not in _env(home)

    def test_blank_llm_key_exits(self, home, cli):
        result = cli("init", input="1\n1\n   \n")
        assert result.exit_code == 1
        assert "API key cannot be empty" in result.output
        assert "OPENAI_API_KEY" not in _env(home)

    def test_skip_llm_non_interactive_leaves_embedding_to_runtime(
        self, home, cli, monkeypatch
    ):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "1")
        result = cli("init", "--skip-llm", input="1\n")
        assert result.exit_code == 0, result.output
        assert "Skipping LLM provider per --skip-llm." in result.output
        assert "LLM Provider:" not in result.output
        assert "Embedding Provider:" not in result.output
        assert _org_config(home) is None

    def test_explicit_cloud_embedding_flag_persists_without_prompt(self, home, cli):
        result = cli("init", "--skip-llm", "--embedding", "gemini", input="1\n")
        assert result.exit_code == 0, result.output
        assert _embedding_model(home) == "gemini/gemini-embedding-001"
        assert "Embedding Provider: Gemini" in result.output

    def test_cloud_embedding_choice_reuses_existing_key(
        self, home, cli, interactive, local_embeddings, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-already-set")
        # storage=1, skip LLM, embedding=2 (OpenAI) -> no key prompt
        result = cli("init", "--skip-llm", input="1\n2\n")
        assert result.exit_code == 0, result.output
        assert "Enter your OPENAI_API_KEY" not in result.output
        assert "OPENAI_API_KEY" not in _env(home)
        assert _embedding_model(home) == "text-embedding-3-small"

    def test_cloud_embedding_choice_collects_missing_key(
        self, home, cli, interactive, local_embeddings
    ):
        result = cli("init", "--skip-llm", input="1\n3\ngem-key\n")
        assert result.exit_code == 0, result.output
        assert 'GEMINI_API_KEY="gem-key"' in _env(home)
        assert _embedding_model(home) == "gemini/gemini-embedding-001"

    def test_invalid_embedding_choice_exits_without_org_config(
        self, home, cli, interactive, local_embeddings
    ):
        result = cli("init", "--skip-llm", input="1\n9\n")
        assert result.exit_code == 1
        assert "choice must be between 1 and 3" in result.output
        assert _org_config(home) is None

    def test_invalid_storage_choice_exits(self, home, cli):
        result = cli("init", input="7\n")
        assert result.exit_code == 1
        assert "choice must be 1, 2, or 3" in result.output


class TestManagedAndSelfHosted:
    def test_managed_verified_account_skips_llm_and_embedding(
        self, home, cli, monkeypatch
    ):
        factory = _whoami(
            monkeypatch,
            WhoamiResponse(
                success=True,
                org_id="org-cov-managed",
                storage_type="supabase",
                storage_label="Supabase (proj)",
                storage_configured=True,
            ),
        )
        result = cli("init", input="2\nrk-managed\n")
        assert result.exit_code == 0, result.output
        from reflexio.defaults import DEFAULT_SERVER_URL

        factory.assert_called_once_with(
            api_key="rk-managed", url_endpoint=DEFAULT_SERVER_URL
        )
        env = _env(home)
        assert f'REFLEXIO_URL="{DEFAULT_SERVER_URL}"' in env
        assert 'REFLEXIO_API_KEY="rk-managed"' in env
        assert "Org ID:        org-cov-managed" in result.output
        assert "Supabase (proj)  [configured]" in result.output
        assert "Skipping LLM provider" in result.output
        assert _org_config(home) is None

    def test_managed_unconfigured_storage_warns(self, home, cli, monkeypatch):
        _whoami(
            monkeypatch,
            WhoamiResponse(
                success=True,
                org_id="org-cov-unconfigured",
                storage_label="none",
                storage_configured=False,
            ),
        )
        result = cli("init", input="2\nrk\n")
        assert result.exit_code == 0, result.output
        assert "Storage type:  unconfigured" in result.output
        assert "[unconfigured]" in result.output
        assert "Your org has no storage configured" in result.output

    def test_managed_self_host_identity_warns_key_not_validated(
        self, home, cli, monkeypatch
    ):
        _whoami(monkeypatch, WhoamiResponse(success=True, org_id="self-host-org"))
        result = cli("init", input="2\nrk\n")
        assert result.exit_code == 0, result.output
        assert "Your API key was NOT validated" in result.output
        assert "Org ID:" not in result.output

    def test_managed_failed_verification_still_keeps_env(self, home, cli, monkeypatch):
        _whoami(
            monkeypatch, WhoamiResponse(success=False, org_id="", message="bad key")
        )
        result = cli("init", input="2\nrk\n")
        assert result.exit_code == 0, result.output
        assert "account verification failed: bad key" in result.output
        assert 'REFLEXIO_API_KEY="rk"' in _env(home)

    def test_managed_unreachable_server_is_reported(self, home, cli, monkeypatch):
        _whoami(monkeypatch, error=ConnectionError("refused"))
        result = cli("init", input="2\nrk\n")
        assert result.exit_code == 0, result.output
        assert "could not verify account — ConnectionError: refused" in result.output

    def test_managed_blank_key_exits_before_writing(self, home, cli, monkeypatch):
        factory = _whoami(
            monkeypatch, WhoamiResponse(success=True, org_id="org-cov-blank")
        )
        result = cli("init", input="2\n  \n")
        assert result.exit_code == 1
        assert "API key cannot be empty" in result.output
        assert _env(home) == ""
        factory.assert_not_called()

    def test_self_hosted_writes_url_and_skips_embedding(self, home, cli):
        result = cli(
            "init", input="3\nhttps://reflexio.internal\nrk-self\n1\nsk-openai\n"
        )
        assert result.exit_code == 0, result.output
        env = _env(home)
        assert 'REFLEXIO_URL="https://reflexio.internal"' in env
        assert 'REFLEXIO_API_KEY="rk-self"' in env
        assert 'OPENAI_API_KEY="sk-openai"' in env
        assert "Choose embedding provider" not in result.output
        assert _org_config(home) is None

    def test_self_hosted_blank_key_exits(self, home, cli):
        result = cli("init", input="3\n\n   \n")
        assert result.exit_code == 1
        assert "API key cannot be empty" in result.output
        assert "REFLEXIO_URL" not in _env(home)


# ---------------------------------------------------------------------------
# Legacy first-run embedding prompt (used by ``services start``)
# ---------------------------------------------------------------------------


class TestPromptEmbeddingProvider:
    @pytest.fixture(autouse=True)
    def _cloud_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".are_local_embedding_dependencies_available",
            lambda: False,
        )

    def _prompts(self, monkeypatch: pytest.MonkeyPatch, answers: list) -> list:
        asked: list = []

        def prompt(text: str, **_kw: object):
            asked.append(text)
            return answers.pop(0)

        monkeypatch.setattr(setup_cmd.typer, "prompt", prompt)
        return asked

    def test_embedding_capable_llm_needs_no_prompt(self, tmp_path: Path):
        assert setup_cmd._prompt_embedding_provider(tmp_path / ".env", "openai") is None

    def test_non_interactive_returns_first_cloud_choice_without_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "true")
        env = tmp_path / ".env"
        assert setup_cmd._prompt_embedding_provider(env, "anthropic") == "OpenAI"
        assert not env.exists()

    def test_interactive_choice_writes_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive
    ):
        asked = self._prompts(monkeypatch, [2, "gem-key"])
        env = tmp_path / ".env"
        assert setup_cmd._prompt_embedding_provider(env, "deepseek") == "Gemini"
        assert asked == ["Choice", "Enter your GEMINI_API_KEY"]
        assert 'GEMINI_API_KEY="gem-key"' in env.read_text()

    def test_interactive_local_choice_needs_no_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive
    ):
        monkeypatch.setattr(
            "reflexio.server.llm.providers.local_embedding_provider"
            ".are_local_embedding_dependencies_available",
            lambda: True,
        )
        asked = self._prompts(monkeypatch, [1])
        env = tmp_path / ".env"
        assert (
            setup_cmd._prompt_embedding_provider(env, "anthropic")
            == "Local (MiniLM-L6-v2)"
        )
        assert asked == ["Choice"]
        assert not env.exists()

    @pytest.mark.parametrize("answers", [[5], [1, "  "]])
    def test_invalid_choice_or_blank_key_exits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interactive, answers
    ):
        self._prompts(monkeypatch, answers)
        env = tmp_path / ".env"
        with pytest.raises(typer.Exit):
            setup_cmd._prompt_embedding_provider(env, "anthropic")
        assert not env.exists()

    def test_env_flag_makes_session_non_interactive(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "yes")
        assert setup_cmd._is_non_interactive() is True


# ---------------------------------------------------------------------------
# setup openclaw
# ---------------------------------------------------------------------------


class _Proc:
    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _FakeOpenclaw:
    """Fake ``openclaw``/``node`` binaries behind ``subprocess.run``."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        binaries: tuple[str, ...] = ("openclaw", "node"),
        inspect: _Proc | None = None,
        config_get: _Proc | None = None,
        fail: dict[tuple[str, ...], _Proc] | None = None,
    ) -> None:
        self.calls: list[tuple[list[str], str | None]] = []
        self.inspect = inspect or _Proc(
            json.dumps({"plugin": {"status": "loaded", "hookCount": 3}})
        )
        self.config_get = config_get or _Proc("{}")
        self.fail = fail or {}
        monkeypatch.setattr(
            setup_cmd.shutil,
            "which",
            lambda name: f"/usr/bin/{name}" if name in binaries else None,
        )
        monkeypatch.setattr(setup_cmd.subprocess, "run", self)

    def __call__(self, argv, **kw):  # noqa: ANN001
        argv = list(argv)
        self.calls.append((argv, kw.get("input")))
        for prefix, result in self.fail.items():
            if tuple(argv[1 : 1 + len(prefix)]) == prefix:
                return result
        if argv[0] == "/usr/bin/node":
            Path(argv[2]).mkdir(parents=True)
            return _Proc()
        if argv[1:3] == ["plugins", "inspect"]:
            return self.inspect
        if argv[1:3] == ["config", "get"]:
            return self.config_get
        return _Proc()

    def subcommands(self) -> list[tuple[str, ...]]:
        return [tuple(argv[1:3]) for argv, _ in self.calls]


class TestOpenclawCommand:
    def test_local_setup_installs_plugin_and_prints_summary(
        self, home, cli, monkeypatch
    ):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "1")
        fake = _FakeOpenclaw(monkeypatch)
        result = cli("openclaw", input="1\nsk-openai\n1\n")
        assert result.exit_code == 0, result.output
        assert 'OPENAI_API_KEY="sk-openai"' in _env(home)
        assert ("plugins", "install") in fake.subcommands()
        assert ("plugins", "enable") in fake.subcommands()
        # The bundled plugin is built with node first; every openclaw call
        # goes through the absolute path resolved once.
        assert fake.calls[0][0][0] == "/usr/bin/node"
        assert all(argv[0] == "/usr/bin/openclaw" for argv, _ in fake.calls[1:])
        assert "Plugin installed and loaded" in result.output
        assert "LLM Provider: OpenAI (gpt-5.4-mini)" in result.output
        assert "1. Start Reflexio: reflexio services start" in result.output
        assert "Embedding Provider:" not in result.output

    def test_remote_setup_skips_embedding_and_local_start_hint(
        self, home, cli, monkeypatch
    ):
        fake = _FakeOpenclaw(monkeypatch)
        result = cli(
            "openclaw",
            "--embedding",
            "openai",
            input="1\nsk-openai\n3\nhttps://r.internal\nrk-oc\n",
        )
        assert result.exit_code == 0, result.output
        assert _org_config(home) is None
        assert "Start Reflexio" not in result.output
        assert "1. Start a conversation" in result.output
        patch = next(
            json.loads(stdin)
            for argv, stdin in fake.calls
            if argv[1:3] == ["config", "patch"] and stdin
        )
        config = patch["plugins"]["entries"][setup_cmd._OPENCLAW_PLUGIN_ID]["config"]
        assert config == {"url": "https://r.internal", "apiKey": "rk-oc"}

    def test_local_setup_with_embedding_flag_reports_embedder(
        self, home, cli, monkeypatch
    ):
        _FakeOpenclaw(monkeypatch)
        result = cli("openclaw", "--embedding", "gemini", input="2\nsk-ant\n1\n")
        assert result.exit_code == 0, result.output
        assert _embedding_model(home) == "gemini/gemini-embedding-001"
        assert "Embedding Provider: Gemini" in result.output

    def test_unloaded_plugin_fails_setup(self, home, cli, monkeypatch):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "1")
        _FakeOpenclaw(monkeypatch, inspect=_Proc(returncode=1))
        result = cli("openclaw", input="1\nsk\n1\n")
        assert result.exit_code == 1
        assert "plugin not loaded" in result.output
        assert "Setup complete!" not in result.output

    def test_missing_env_file_exits(self, home, cli, monkeypatch):
        (home / ".reflexio" / ".env").unlink()
        monkeypatch.setattr(env_loader, "_find_env_example", lambda _pkg: None)
        result = cli("openclaw")
        assert result.exit_code == 1
        assert "could not locate or create a .env file" in result.output

    def test_missing_openclaw_cli_exits(self, home, cli, monkeypatch):
        monkeypatch.setenv("REFLEXIO_NONINTERACTIVE", "1")
        fake = _FakeOpenclaw(monkeypatch, binaries=())
        result = cli("openclaw", input="1\nsk\n1\n")
        assert result.exit_code == 1
        assert "openclaw CLI not found" in result.output
        assert fake.calls == []


class TestOpenclawInstallSteps:
    def test_missing_node_stops_before_install(self, tmp_path, monkeypatch):
        fake = _FakeOpenclaw(monkeypatch, binaries=("openclaw",))
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env")
        assert ("plugins", "install") not in fake.subcommands()

    def test_plugin_build_failure_stops_before_install(self, tmp_path, monkeypatch):
        fake = _FakeOpenclaw(monkeypatch)
        original = fake.__call__

        def run(argv, **kw):  # noqa: ANN001
            if argv[0] == "/usr/bin/node":
                fake.calls.append((list(argv), None))
                return _Proc(returncode=2, stderr="esbuild missing")
            return original(argv, **kw)

        monkeypatch.setattr(setup_cmd.subprocess, "run", run)
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env")
        assert ("plugins", "install") not in fake.subcommands()

    @pytest.mark.parametrize(
        "stdout", ["not json", json.dumps(["not", "an", "object"])]
    )
    def test_unparseable_inspect_output_is_not_loaded(
        self, stdout, tmp_path, monkeypatch
    ):
        _FakeOpenclaw(monkeypatch, inspect=_Proc(stdout))
        assert setup_cmd._openclaw_plugin_loaded("/usr/bin/openclaw") is False

    def test_unreadable_existing_settings_stop_install(self, tmp_path, monkeypatch):
        fake = _FakeOpenclaw(
            monkeypatch,
            config_get=_Proc(
                json.dumps({"error": {"message": "permission denied"}}), 1
            ),
        )
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env", "/plugin")
        assert ("plugins", "install") not in fake.subcommands()

    def test_non_object_existing_settings_stop_install(self, tmp_path, monkeypatch):
        fake = _FakeOpenclaw(monkeypatch, config_get=_Proc(json.dumps([1, 2])))
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env", "/plugin")
        assert ("plugins", "install") not in fake.subcommands()

    def test_non_object_policy_section_stops_install(self, tmp_path, monkeypatch):
        properties = json.loads(
            (setup_cmd._OPENCLAW_PLUGIN_DIR / "openclaw.plugin.json").read_text()
        )["configSchema"]["properties"]
        section = next(name for name, p in properties.items() if "properties" in p)
        fake = _FakeOpenclaw(
            monkeypatch, config_get=_Proc(json.dumps({section: "not-a-dict"}))
        )
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env", "/plugin")
        assert ("config", "unset") not in fake.subcommands()

    def test_failed_unset_stops_before_install(self, tmp_path, monkeypatch):
        fake = _FakeOpenclaw(
            monkeypatch,
            config_get=_Proc(json.dumps({"url": "http://old"})),
            fail={("config", "unset"): _Proc(returncode=1)},
        )
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env", "/plugin")
        assert ("plugins", "install") not in fake.subcommands()

    def test_failed_restore_is_reported(self, tmp_path, monkeypatch, capsys):
        fake = _FakeOpenclaw(
            monkeypatch,
            config_get=_Proc(json.dumps({"url": "http://old"})),
            fail={
                ("plugins", "enable"): _Proc(returncode=1, stderr="enable failed"),
                ("config", "patch"): _Proc(returncode=0),
            },
        )
        original = fake.__call__
        patches = 0

        def run(argv, **kw):  # noqa: ANN001
            nonlocal patches
            if list(argv[1:3]) == ["config", "patch"]:
                patches += 1
                fake.calls.append((list(argv), kw.get("input")))
                # First patch (install) succeeds; the rollback patch fails.
                return _Proc(returncode=0 if patches == 1 else 1)
            return original(argv, **kw)

        monkeypatch.setattr(setup_cmd.subprocess, "run", run)
        with pytest.raises(typer.Exit):
            setup_cmd._install_openclaw_integration(tmp_path / ".env", "/plugin")
        out = capsys.readouterr().out
        assert "could not restore previous plugin settings" in out
        assert "'openclaw plugins enable' failed: enable failed" in out
        rollback = json.loads(fake.calls[-1][1] or "{}")
        assert rollback["plugins"]["entries"][setup_cmd._OPENCLAW_PLUGIN_ID] == {
            "config": {"url": "http://old"}
        }


class TestOpenclawUninstall:
    def test_uninstall_removes_markers_legacy_keys_and_purges_state(
        self, home, cli, monkeypatch
    ):
        fake = _FakeOpenclaw(monkeypatch)
        reflexio_dir = home / ".reflexio"
        (reflexio_dir / ".env").write_text('OPENCLAW_SMART_USE_LOCAL_CLI="1"\nKEEP=1\n')
        (reflexio_dir / ".setup_complete_openclaw").write_text("")
        state = home / ".openclaw-smart"
        (state / "sessions").mkdir(parents=True)
        result = cli("openclaw", "--uninstall", "--purge", input="y\n")
        assert result.exit_code == 0, result.output
        assert fake.subcommands() == [
            ("plugins", "disable"),
            ("plugins", "uninstall"),
        ]
        assert _env(home) == "KEEP=1\n"
        assert not (reflexio_dir / ".setup_complete_openclaw").exists()
        assert not state.exists()
        assert f"Removed plugin state: {state}" in result.output
        assert "fully removed" in result.output

    def test_uninstall_without_cli_warns_and_keeps_state_without_purge(
        self, home, cli, monkeypatch
    ):
        fake = _FakeOpenclaw(monkeypatch, binaries=())
        state = home / ".openclaw-smart"
        state.mkdir()
        result = cli("openclaw", "--uninstall", input="y\n")
        assert result.exit_code == 0, result.output
        assert "openclaw CLI not found in OPENCLAW_BIN or PATH" in result.output
        assert fake.calls == []
        assert state.exists()

    def test_declined_confirmation_changes_nothing(self, home, cli, monkeypatch):
        fake = _FakeOpenclaw(monkeypatch)
        (home / ".reflexio" / ".env").write_text('OPENCLAW_BIN="/x"\n')
        result = cli("openclaw", "--uninstall", input="n\n")
        assert result.exit_code == 1
        assert fake.calls == []
        assert _env(home) == 'OPENCLAW_BIN="/x"\n'


class TestEnvFileHelpers:
    def test_remove_env_keys_on_missing_file_is_noop(self, tmp_path: Path):
        setup_cmd._remove_env_keys(tmp_path / ".env", ("A",))
        assert not (tmp_path / ".env").exists()

    def test_remove_env_keys_leaves_unmatched_file_untouched(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("A=1")  # no trailing newline: a rewrite would add one
        setup_cmd._remove_env_keys(env, ("B",))
        assert env.read_text() == "A=1"

    def test_remove_last_key_leaves_empty_file(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("A=1\n")
        setup_cmd._remove_env_keys(env, ("A",))
        assert env.read_text() == ""

    def test_read_env_key(self, tmp_path: Path):
        env = tmp_path / ".env"
        env.write_text("# A=commented\n\nB=\nA='quoted'\n")
        assert setup_cmd._read_env_key(env, "A") == "quoted"
        assert setup_cmd._read_env_key(env, "B") is None
        assert setup_cmd._read_env_key(env, "C") is None
        assert setup_cmd._read_env_key(None, "A") is None


# ---------------------------------------------------------------------------
# setup openai-codex
# ---------------------------------------------------------------------------


@pytest.fixture
def token_path(home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    auth = home / ".reflexio" / "auth"
    path = auth / "openai-codex.json"
    monkeypatch.setattr(codex_auth, "REFLEXIO_AUTH_DIR", auth)
    monkeypatch.setattr(codex_auth, "REFLEXIO_CODEX_TOKENS_PATH", path)
    return path


def _tokens(**overrides: object) -> codex_auth.CodexTokens:
    values: dict = {
        "access_token": "at",
        "refresh_token": "rt",
        "account_id": "acct-0123456789",
        "expires_at": int(time.time()) + 3600,
        "plan_type": "max-x20",
        "email": "u@example.com",
    }
    values.update(overrides)
    return codex_auth.CodexTokens(**values)


class TestOpenaiCodex:
    def test_logout_removes_saved_tokens(self, cli, token_path):
        codex_auth.save_tokens(_tokens())
        result = cli("openai-codex", "--logout")
        assert result.exit_code == 0, result.output
        assert not token_path.exists()
        assert f"Removed {token_path}" in result.output

    def test_logout_without_tokens(self, cli, token_path):
        result = cli("openai-codex", "--logout")
        assert result.exit_code == 0
        assert "No saved Codex tokens to remove." in result.output

    def test_show_without_tokens_exits(self, cli, token_path):
        result = cli("openai-codex", "--show")
        assert result.exit_code == 1
        assert f"No tokens at {token_path}." in result.output

    @pytest.mark.parametrize(
        "account_id,expected", [("acct-0123456789", "...23456789"), ("", "(empty)")]
    )
    def test_show_prints_metadata_without_secrets(
        self, cli, token_path, account_id, expected
    ):
        codex_auth.save_tokens(_tokens(account_id=account_id))
        result = cli("openai-codex", "--show")
        assert result.exit_code == 0, result.output
        assert "email:     u@example.com" in result.output
        assert "plan_type: max-x20" in result.output
        assert expected in result.output
        assert "expired:   False" in result.output
        assert "rt" not in result.output.split()

    def test_login_success_saves_and_reports(self, cli, token_path, monkeypatch):
        calls: list[dict] = []

        def login(**kwargs: object) -> codex_auth.CodexTokens:
            calls.append(kwargs)
            tokens = _tokens()
            codex_auth.save_tokens(tokens)
            return tokens

        monkeypatch.setattr(codex_auth, "login_interactive", login)
        result = cli("openai-codex", "--no-browser", "--timeout", "7")
        assert result.exit_code == 0, result.output
        assert calls == [{"open_browser": False, "timeout_s": 7}]
        assert "Sign-in successful." in result.output
        assert "email:     u@example.com" in result.output
        assert codex_auth.load_tokens_raw() == _tokens(
            expires_at=codex_auth.load_tokens_raw().expires_at  # type: ignore[union-attr]
        )

    def test_login_without_email_omits_email_line(self, cli, token_path, monkeypatch):
        monkeypatch.setattr(
            codex_auth, "login_interactive", lambda **_: _tokens(email="")
        )
        result = cli("openai-codex")
        assert result.exit_code == 0, result.output
        assert "email:" not in result.output

    @pytest.mark.parametrize(
        "error,message",
        [
            (TimeoutError("no callback"), "Timed out: no callback"),
            (ValueError("state mismatch"), "Login failed: state mismatch"),
        ],
    )
    def test_login_failure_exits(self, cli, token_path, monkeypatch, error, message):
        def login(**_: object) -> codex_auth.CodexTokens:
            raise error

        monkeypatch.setattr(codex_auth, "login_interactive", login)
        result = cli("openai-codex")
        assert result.exit_code == 1
        assert message in result.output
        assert not token_path.exists()
