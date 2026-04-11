"""Shared fixtures for CLI tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from reflexio.cli.app import create_app

# All modules that import get_client at the top level.
_GET_CLIENT_TARGETS = [
    "reflexio.cli.commands.interactions.get_client",
    "reflexio.cli.commands.agent_playbooks.get_client",
    "reflexio.cli.commands.user_playbooks.get_client",
    "reflexio.cli.commands.profiles.get_client",
    "reflexio.cli.commands.config_cmd.get_client",
    "reflexio.cli.commands.shortcuts.get_client",
    "reflexio.cli.state.get_client",
]


@pytest.fixture
def runner():
    """Provide a Typer CliRunner for invoking commands."""
    return CliRunner()


@pytest.fixture
def app():
    """Create a fresh Typer app for each test."""
    return create_app()


@pytest.fixture
def mock_client():
    """Mock ReflexioClient injected via get_client.

    Patches ``get_client`` in every command module so that calls to
    ``get_client(ctx)`` return this mock regardless of where the
    function was imported.
    """
    client = MagicMock()
    patches = [patch(target, return_value=client) for target in _GET_CLIENT_TARGETS]
    for p in patches:
        p.start()
    yield client
    for p in patches:
        p.stop()
