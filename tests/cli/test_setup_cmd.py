"""Unit tests for setup_cmd helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from reflexio.cli.commands.setup_cmd import (
    InstallLocation,
    _detect_install_locations,
    _install_claude_code_integration,
    _prompt_install_location,
    _prompt_storage,
    _remove_from_dir,
    _set_env_var,
    _write_marker,
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
        assert 'REFLEXIO_URL="http://localhost:8081"' in env.read_text()

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
# _prompt_install_location — location picker
# ---------------------------------------------------------------------------


class TestPromptInstallLocation:
    """Covers the interactive install location picker."""

    def test_choice_1_returns_all_projects(self) -> None:
        """Choice 1 returns ALL_PROJECTS."""
        with patch("typer.prompt", return_value=1):
            result = _prompt_install_location()
        assert result == InstallLocation.ALL_PROJECTS

    def test_choice_2_returns_current_project(self) -> None:
        """Choice 2 returns CURRENT_PROJECT."""
        with patch("typer.prompt", return_value=2):
            result = _prompt_install_location()
        assert result == InstallLocation.CURRENT_PROJECT

    def test_default_is_all_projects(self) -> None:
        """Default prompt value is 1 (ALL_PROJECTS)."""
        with patch("typer.prompt", return_value=1) as mock_prompt:
            _prompt_install_location()
        mock_prompt.assert_called_once_with("Choice", type=int, default=1)

    def test_invalid_choice_exits(self) -> None:
        """Choices outside 1/2 raise typer.Exit."""
        with (
            patch("typer.prompt", return_value=5),
            pytest.raises(typer.Exit),
        ):
            _prompt_install_location()


# ---------------------------------------------------------------------------
# _install_claude_code_integration — both locations
# ---------------------------------------------------------------------------


class TestInstallClaudeCodeIntegration:
    """Covers install to project-level and user-level locations."""

    def test_project_level_creates_files(self, tmp_path: Path) -> None:
        """Project-level install creates skill, rules, hooks, and marker."""
        _install_claude_code_integration(
            tmp_path, location=InstallLocation.CURRENT_PROJECT
        )
        claude_dir = tmp_path / ".claude"
        assert (claude_dir / "skills" / "reflexio" / "SKILL.md").exists()
        assert (claude_dir / "rules" / "reflexio.md").exists()
        assert (claude_dir / "settings.json").exists()
        # Marker file
        marker = claude_dir / "skills" / "reflexio" / ".installed-by-reflexio"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["location"] == "current_project"

    def test_user_level_creates_files_in_home(self, tmp_path: Path) -> None:
        """User-level install creates files under the target dir (simulating ~)."""
        _install_claude_code_integration(
            tmp_path, location=InstallLocation.ALL_PROJECTS
        )
        claude_dir = tmp_path / ".claude"
        assert (claude_dir / "skills" / "reflexio" / "SKILL.md").exists()
        assert (claude_dir / "rules" / "reflexio.md").exists()
        assert (claude_dir / "settings.json").exists()
        marker = claude_dir / "skills" / "reflexio" / ".installed-by-reflexio"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["location"] == "all_projects"
        assert "installed_at" in data

    def test_expert_mode_installs_command(self, tmp_path: Path) -> None:
        """Expert mode also installs the reflexio-extract command."""
        _install_claude_code_integration(
            tmp_path, expert=True, location=InstallLocation.CURRENT_PROJECT
        )
        cmd = tmp_path / ".claude" / "commands" / "reflexio-extract" / "SKILL.md"
        assert cmd.exists()

    def test_normal_mode_no_command(self, tmp_path: Path) -> None:
        """Normal mode does not install the reflexio-extract command."""
        _install_claude_code_integration(
            tmp_path, location=InstallLocation.CURRENT_PROJECT
        )
        cmd = tmp_path / ".claude" / "commands" / "reflexio-extract" / "SKILL.md"
        assert not cmd.exists()

    def test_hooks_in_settings_json(self, tmp_path: Path) -> None:
        """Hooks are written to settings.json with correct events."""
        _install_claude_code_integration(
            tmp_path, location=InstallLocation.ALL_PROJECTS
        )
        settings = json.loads(
            (tmp_path / ".claude" / "settings.json").read_text()
        )
        assert "SessionStart" in settings["hooks"]
        assert "UserPromptSubmit" in settings["hooks"]

    def test_idempotent_install(self, tmp_path: Path) -> None:
        """Running install twice doesn't corrupt files or duplicate hooks."""
        for _ in range(2):
            _install_claude_code_integration(
                tmp_path, location=InstallLocation.ALL_PROJECTS
            )
        settings = json.loads(
            (tmp_path / ".claude" / "settings.json").read_text()
        )
        # Each event should have exactly one hook entry
        assert len(settings["hooks"]["SessionStart"]) == 1
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1


# ---------------------------------------------------------------------------
# _detect_install_locations / _remove_from_dir / uninstall
# ---------------------------------------------------------------------------


class TestUninstallDetection:
    """Covers marker-based install detection and removal."""

    def test_detect_no_installs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns empty list when nothing is installed."""
        fake_home = tmp_path / "empty_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        locations = _detect_install_locations(tmp_path / "project")
        assert locations == []

    def test_detect_project_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detects project-level install via marker file."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        project = tmp_path / "project"
        project.mkdir()
        _install_claude_code_integration(
            project, location=InstallLocation.CURRENT_PROJECT
        )
        locations = _detect_install_locations(project)
        found_locs = {loc for loc, _ in locations}
        assert InstallLocation.CURRENT_PROJECT in found_locs
        assert InstallLocation.ALL_PROJECTS not in found_locs

    def test_detect_user_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Detects user-level install via marker file in home dir."""
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        _install_claude_code_integration(
            fake_home, location=InstallLocation.ALL_PROJECTS
        )
        locations = _detect_install_locations(tmp_path / "project")
        found_locs = {loc for loc, _ in locations}
        assert InstallLocation.ALL_PROJECTS in found_locs

    def test_remove_from_dir_cleans_all_files(self, tmp_path: Path) -> None:
        """_remove_from_dir removes skill, rules, commands, and hooks."""
        _install_claude_code_integration(
            tmp_path, expert=True, location=InstallLocation.CURRENT_PROJECT
        )
        claude_dir = tmp_path / ".claude"
        assert (claude_dir / "skills" / "reflexio").exists()
        assert (claude_dir / "rules" / "reflexio.md").exists()
        assert (claude_dir / "commands" / "reflexio-extract").exists()

        _remove_from_dir(tmp_path)

        assert not (claude_dir / "skills" / "reflexio").exists()
        assert not (claude_dir / "rules" / "reflexio.md").exists()
        assert not (claude_dir / "commands" / "reflexio-extract").exists()
        # settings.json should have empty hooks
        settings = json.loads((claude_dir / "settings.json").read_text())
        assert "hooks" not in settings or not settings.get("hooks")

    def test_marker_file_metadata(self, tmp_path: Path) -> None:
        """Marker file contains location and installed_at fields."""
        marker = tmp_path / ".marker"
        _write_marker(marker, InstallLocation.ALL_PROJECTS)
        data = json.loads(marker.read_text())
        assert data["location"] == "all_projects"
        assert "installed_at" in data


# ---------------------------------------------------------------------------
# CLI flag mutual exclusion (tested via the command function directly)
# ---------------------------------------------------------------------------


class TestClaudeCodeSetupFlags:
    """Tests for --global / --project-dir mutual exclusion."""

    def test_global_and_project_dir_mutual_exclusion(self) -> None:
        """Passing both --global and --project-dir raises typer.Exit."""
        from reflexio.cli.commands.setup_cmd import claude_code_setup

        with pytest.raises(typer.Exit):
            claude_code_setup(
                uninstall=False,
                expert=False,
                project_dir=Path("/tmp"),
                global_install=True,
            )
