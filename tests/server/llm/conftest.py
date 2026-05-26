"""Shared fixtures for LLM provider tests."""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.storage_base import BaseStorage


@pytest.fixture
def storage() -> Generator[BaseStorage]:
    """Yield a fresh, isolated SQLiteStorage instance with migrations applied."""
    with tempfile.TemporaryDirectory() as temp_dir:
        from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

        with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
            yield SQLiteStorage(org_id="llm_test", db_path=f"{temp_dir}/reflexio.db")
