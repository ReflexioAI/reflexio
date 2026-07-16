"""Shared fixtures for LLM provider tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.storage_base import BaseStorage

# ``reflexio.server`` loads ``./.env`` + ``~/.reflexio/.env`` into os.environ at
# import time (and litellm's import-time dotenv walk-up can pull in a parent
# repo's .env). Provider tests assert against the DEFAULT host/port code paths,
# so a developer machine that opts into the codex host or a custom embedding
# port would otherwise flip these tests onto different branches and fail them.
_MACHINE_ENV_OVERRIDES = (
    "CLAUDE_SMART_HOST",
    "CLAUDE_SMART_USE_LOCAL_CLI",
    "CLAUDE_SMART_CLI_PATH",
    "CLAUDE_SMART_CODEX_PATH",
    "EMBEDDING_PORT",
)


@pytest.fixture(autouse=True)
def _isolate_machine_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip machine-specific env overrides so provider tests stay hermetic.

    Tests that exercise a non-default host/port explicitly set these vars via
    their own ``monkeypatch.setenv`` calls, which run after this fixture.
    """
    for var in _MACHINE_ENV_OVERRIDES:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def storage(monkeypatch: pytest.MonkeyPatch) -> Generator[BaseStorage]:
    """Yield a fresh, isolated SQLiteStorage instance with migrations applied."""
    # SQLiteStorage.__init__ resolves a generation model via provider
    # auto-detection, which needs SOME provider key in the env. Guarantee one
    # so the fixture works on machines without real API keys.
    if not os.environ.get("OPENAI_API_KEY"):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-mock")
    with tempfile.TemporaryDirectory() as temp_dir:
        from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            yield SQLiteStorage(org_id="llm_test", db_path=f"{temp_dir}/reflexio.db")
