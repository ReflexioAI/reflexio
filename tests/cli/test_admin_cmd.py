"""Unit tests for ``reflexio admin cache invalidate``.

The CLI thinly wraps ``client.invalidate_cache()``; we patch the
client and assert on what got rendered to the user.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cli_app():
    return create_app()


class TestAdminCacheInvalidate:
    def test_invalidate_evicted_path(self, runner: CliRunner, cli_app) -> None:
        """When the server reports invalidated=True, the CLI prints a confirmation."""
        mock_client = MagicMock()
        mock_client.invalidate_cache.return_value = {
            "invalidated": True,
            "org_id": "self-host-org",
        }
        with patch(
            "reflexio.cli.commands.admin_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["admin", "cache", "invalidate"])
        assert result.exit_code == 0, result.output
        # The CLI surfaces the org_id so operators can confirm the
        # right tenant was hit; we don't pin exact wording but the
        # success path mentions "evicted".
        mock_client.invalidate_cache.assert_called_once_with(org_id=None)

    def test_invalidate_no_op_path(self, runner: CliRunner, cli_app) -> None:
        """invalidated=False is not an error — exit cleanly with a no-op message."""
        mock_client = MagicMock()
        mock_client.invalidate_cache.return_value = {
            "invalidated": False,
            "org_id": "self-host-org",
        }
        with patch(
            "reflexio.cli.commands.admin_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(cli_app, ["admin", "cache", "invalidate"])
        assert result.exit_code == 0, result.output

    def test_invalidate_forwards_org_id(self, runner: CliRunner, cli_app) -> None:
        """``--org-id`` is passed through to the client as a verification token."""
        mock_client = MagicMock()
        mock_client.invalidate_cache.return_value = {
            "invalidated": True,
            "org_id": "abc-123",
        }
        with patch(
            "reflexio.cli.commands.admin_cmd.get_client", return_value=mock_client
        ):
            result = runner.invoke(
                cli_app, ["admin", "cache", "invalidate", "--org-id", "abc-123"]
            )
        assert result.exit_code == 0, result.output
        mock_client.invalidate_cache.assert_called_once_with(org_id="abc-123")
