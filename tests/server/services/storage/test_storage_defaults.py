"""Tests for LOCAL_STORAGE_PATH default resolution and the SQLite db_path fallback.

Covers the env-var consolidation that replaced SQLITE_FILE_DIRECTORY with
LOCAL_STORAGE_PATH and moved the default data directory to ~/.reflexio/data.
"""

import importlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


def test_local_storage_path_defaults_to_home_reflexio_data() -> None:
    """With LOCAL_STORAGE_PATH unset, reflexio.server.LOCAL_STORAGE_PATH
    resolves to ~/.reflexio/data."""
    from reflexio.cli.paths import reflexio_home

    expected = str(reflexio_home() / "data")

    env = {k: v for k, v in os.environ.items() if k != "LOCAL_STORAGE_PATH"}
    import reflexio.server as server_module

    try:
        with patch.dict(os.environ, env, clear=True):
            reloaded = importlib.reload(server_module)
            assert expected == reloaded.LOCAL_STORAGE_PATH
    finally:
        # Restore module with the original process environment so later
        # tests see the usual value. Must run after patch.dict exits.
        importlib.reload(server_module)


def test_local_storage_path_empty_string_falls_back_to_default() -> None:
    """LOCAL_STORAGE_PATH='' (blank) also falls back to ~/.reflexio/data
    rather than resolving to an empty path."""
    from reflexio.cli.paths import reflexio_home

    expected = str(reflexio_home() / "data")

    import reflexio.server as server_module

    try:
        with patch.dict(os.environ, {"LOCAL_STORAGE_PATH": ""}):
            reloaded = importlib.reload(server_module)
            assert expected == reloaded.LOCAL_STORAGE_PATH
    finally:
        # Restore module with the original process environment so later
        # tests see the usual value. Must run after patch.dict exits.
        importlib.reload(server_module)


def test_sqlite_storage_uses_local_storage_path_when_db_path_none() -> None:
    """SQLiteStorage(db_path=None) resolves to LOCAL_STORAGE_PATH/reflexio.db."""
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("reflexio.server.LOCAL_STORAGE_PATH", temp_dir),
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        storage = SQLiteStorage(org_id="0", db_path=None)
        assert storage.db_path == str(Path(temp_dir) / "reflexio.db")


def test_local_storage_path_honors_reflexio_log_dir_override(tmp_path: Path) -> None:
    """When ``REFLEXIO_LOG_DIR`` is set, the default ``LOCAL_STORAGE_PATH``
    rebases off it: ``<REFLEXIO_LOG_DIR>/.reflexio/data`` instead of
    ``~/.reflexio/data``."""
    expected = str(tmp_path / ".reflexio" / "data")

    import reflexio.server as server_module

    try:
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_STORAGE_PATH"}
        env["REFLEXIO_LOG_DIR"] = str(tmp_path)
        with patch.dict(os.environ, env, clear=True):
            reloaded = importlib.reload(server_module)
            assert expected == reloaded.LOCAL_STORAGE_PATH
    finally:
        importlib.reload(server_module)


def test_sqlite_storage_explicit_db_path_overrides_env() -> None:
    """An explicit db_path argument takes precedence over LOCAL_STORAGE_PATH."""
    with (
        tempfile.TemporaryDirectory() as env_dir,
        tempfile.TemporaryDirectory() as explicit_dir,
        patch("reflexio.server.LOCAL_STORAGE_PATH", env_dir),
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        explicit_path = str(Path(explicit_dir) / "custom.db")
        storage = SQLiteStorage(org_id="0", db_path=explicit_path)
        assert storage.db_path == explicit_path
