"""Unit tests for ``reflexio config storage``.

These don't spin up the server — they patch ``client.get_my_config()``
to return fixed ``MyConfigResponse`` values and assert the CLI renders
the expected output. The goal is to pin the contract of:

- masking behaviour by default (storage)
- the --reveal confirmation prompt
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app
from reflexio.models.api_schema.service_schemas import MyConfigResponse


@pytest.fixture
def runner() -> CliRunner:
    # Combine stderr into result.output — print_info goes to stderr via
    # plain print(), which CliRunner still captures when mix_stderr=True.
    return CliRunner()


@pytest.fixture
def cli_app():
    return create_app()


def _make_supabase_response() -> MyConfigResponse:
    return MyConfigResponse(
        success=True,
        storage_type="supabase",
        storage_config={
            "url": "https://jpkjckbyxrdefzomiyse.supabase.co",
            "key": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.verysecrettoken",
            "db_url": "postgresql://postgres.abc:pw@host.supabase.com:6543/postgres",
        },
    )


class TestConfigStorage:
    def test_masks_by_default(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["config", "storage"])
        assert result.exit_code == 0, result.output
        # The full key must never appear in any captured output
        assert "verysecrettoken" not in result.output
        # Masked form appears
        assert "supabase" in result.output

    def test_reveal_requires_confirmation(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            # Decline the confirmation prompt → raises typer.Abort
            result = runner.invoke(
                cli_app, ["config", "storage", "--reveal"], input="n\n"
            )
        # Abort exit code is 1
        assert result.exit_code != 0

    def test_reveal_confirmed_prints_raw(self, runner: CliRunner, cli_app) -> None:
        mock_client = MagicMock()
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "storage", "--reveal"], input="y\n"
            )
        assert result.exit_code == 0
        assert "verysecrettoken" in result.output


class TestConfigLocal:
    """Tests for ``reflexio config local`` — reads local config, no server needed."""

    _PATCH_LOAD = "reflexio.cli.bootstrap_config.load_storage_from_config"
    _PATCH_RESOLVE = "reflexio.cli.bootstrap_config.resolve_storage"

    def test_human_readable_output(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="sqlite"),
            patch(self._PATCH_RESOLVE, return_value="sqlite"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "Persisted storage: sqlite" in result.output
        assert "Resolved storage:  sqlite" in result.output
        assert "mode: local" in result.output

    def test_human_readable_no_persisted(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value=None),
            patch(self._PATCH_RESOLVE, return_value="sqlite"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "(not set)" in result.output

    def test_json_mode(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="supabase"),
            patch(self._PATCH_RESOLVE, return_value="supabase"),
        ):
            result = runner.invoke(cli_app, ["--json", "config", "local"])
        assert result.exit_code == 0, result.output
        import json

        envelope = json.loads(result.output)
        assert envelope["ok"] is True
        data = envelope["data"]
        assert data["persisted_storage"] == "supabase"
        assert data["resolved_storage"] == "supabase"
        assert data["resolved_mode"] == "cloud"
        assert "config_file" in data

    def test_cloud_mode_for_supabase(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="supabase"),
            patch(self._PATCH_RESOLVE, return_value="supabase"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "mode: cloud" in result.output

    def test_local_mode_for_disk(self, runner: CliRunner, cli_app) -> None:
        with (
            patch(self._PATCH_LOAD, return_value="disk"),
            patch(self._PATCH_RESOLVE, return_value="disk"),
        ):
            result = runner.invoke(cli_app, ["config", "local"])
        assert result.exit_code == 0, result.output
        assert "mode: local" in result.output
