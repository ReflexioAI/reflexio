"""App factory smoke tests for the Reflexio CLI."""

from __future__ import annotations

import typer

from reflexio import __version__ as _VERSION  # noqa: N812


class TestCreateApp:
    """Tests for the create_app() factory function."""

    def test_returns_typer_app(self, app: typer.Typer) -> None:
        """create_app() should return a Typer instance."""
        assert isinstance(app, typer.Typer)

    def test_all_command_groups_registered(self, runner, app) -> None:
        """--help output should list all 11 command groups."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        expected_groups = [
            "interactions",
            "user-profiles",
            "agent-playbooks",
            "user-playbooks",
            "config",
            "auth",
            "status",
            "api",
            "doctor",
            "services",
            "setup",
        ]
        for group in expected_groups:
            assert group in result.output, f"Missing command group: {group}"

    def test_version_flag(self, runner, app) -> None:
        """--version should print the version string and exit 0."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert _VERSION in result.output

    def test_help_shows_description(self, runner, app) -> None:
        """--help should include the CLI description."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Reflexio CLI" in result.output

    def test_top_level_shortcuts_exist(self, runner, app) -> None:
        """Top-level shortcuts publish, search, context should appear in --help."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        for shortcut in ("publish", "search", "context"):
            assert shortcut in result.output, f"Missing shortcut: {shortcut}"
