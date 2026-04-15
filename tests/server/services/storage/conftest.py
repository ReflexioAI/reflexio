"""Shared fixtures for storage contract tests.

The parametrized ``storage`` fixture runs every contract test against each
locally-testable backend (SQLiteStorage).  SupabaseStorage is excluded
because it requires a live Supabase instance.
"""

import tempfile
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.server.services.storage.storage_base import BaseStorage


@pytest.fixture(params=["sqlite"])
def storage(request: pytest.FixtureRequest) -> Generator[BaseStorage]:
    """Yield a fresh, isolated storage instance for each backend."""
    backend = request.param

    with tempfile.TemporaryDirectory() as temp_dir:
        if backend == "sqlite":
            from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

            with patch.object(
                SQLiteStorage, "_get_embedding", return_value=[0.0] * 512
            ):
                yield SQLiteStorage(
                    org_id="contract_test", db_path=f"{temp_dir}/reflexio.db"
                )
