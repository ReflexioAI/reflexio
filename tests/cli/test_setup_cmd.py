"""Unit tests for setup_cmd helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from reflexio.cli.commands.setup_cmd import (
    _prompt_storage,
    _set_env_var,
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
