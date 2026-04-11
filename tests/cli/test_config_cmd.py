"""Unit tests for ``reflexio config`` storage + pull commands.

These don't spin up the server — they patch ``client.get_my_config()``
to return fixed ``MyConfigResponse`` values and assert the CLI renders
or writes the expected output. The goal is to pin the contract of:

- masking behaviour by default (storage)
- the --reveal confirmation prompt
- .env file writing (pull)
- the clobber-guard on pull without --force
- the supabase-only restriction on pull
"""

from __future__ import annotations

from pathlib import Path
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


class TestConfigPull:
    def test_writes_env_file(self, runner: CliRunner, cli_app, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        mock_client = MagicMock()
        mock_client.base_url = "https://reflexio.ai"
        mock_client.api_key = "rflx-test-key"
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "pull", "--env-file", str(env_file)]
            )
        assert result.exit_code == 0, result.output
        content = env_file.read_text()
        assert 'SUPABASE_URL="https://jpkjckbyxrdefzomiyse.supabase.co"' in content
        assert "SUPABASE_KEY=" in content
        assert "SUPABASE_DB_URL=" in content
        assert 'REFLEXIO_URL="https://reflexio.ai"' in content
        assert 'REFLEXIO_API_KEY="rflx-test-key"' in content

    def test_refuses_clobber_without_force(
        self, runner: CliRunner, cli_app, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('SUPABASE_URL="https://existing.example"\n')
        mock_client = MagicMock()
        mock_client.base_url = "https://reflexio.ai"
        mock_client.api_key = "rflx-key"
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "pull", "--env-file", str(env_file)]
            )
        assert result.exit_code != 0
        # Existing line is intact — the pull was refused
        assert "https://existing.example" in env_file.read_text()

    def test_force_overwrites(self, runner: CliRunner, cli_app, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text('SUPABASE_URL="https://old.example"\n')
        mock_client = MagicMock()
        mock_client.base_url = "https://reflexio.ai"
        mock_client.api_key = "rflx-key"
        mock_client.get_my_config.return_value = _make_supabase_response()
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app,
                ["config", "pull", "--force", "--env-file", str(env_file)],
            )
        assert result.exit_code == 0, result.output
        content = env_file.read_text()
        assert "jpkjckbyxrdefzomiyse" in content
        assert "old.example" not in content

    def test_refuses_non_supabase_storage(
        self, runner: CliRunner, cli_app, tmp_path: Path
    ) -> None:
        env_file = tmp_path / ".env"
        mock_client = MagicMock()
        mock_client.base_url = "http://localhost:8081"
        mock_client.api_key = ""
        mock_client.get_my_config.return_value = MyConfigResponse(
            success=True,
            storage_type="sqlite",
            storage_config={"db_path": "/tmp/reflexio.db"},
        )
        with patch(
            "reflexio.cli.commands.config_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["config", "pull", "--env-file", str(env_file)]
            )
        assert result.exit_code != 0
        assert not env_file.exists() or "SUPABASE" not in env_file.read_text()
