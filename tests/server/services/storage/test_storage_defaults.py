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

    import reflexio.server as server_module

    assert (
        Path(server_module.LOCAL_STORAGE_PATH).resolve()
        == Path(os.environ["LOCAL_STORAGE_PATH"]).resolve()
    )

    env = {k: v for k, v in os.environ.items() if k != "LOCAL_STORAGE_PATH"}

    try:
        with (
            patch.dict(os.environ, env, clear=True),
            patch("reflexio.cli.env_loader.load_reflexio_env"),
        ):
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


def test_sqlite_storage_derives_db_path_from_identity_when_db_path_none() -> None:
    """SQLiteStorage(db_path=None) resolves to a file named for its identity.

    Previously every identity resolved to one shared ``reflexio.db``.
    """
    with (
        tempfile.TemporaryDirectory() as temp_dir,
        patch("reflexio.server.LOCAL_STORAGE_PATH", temp_dir),
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        storage = SQLiteStorage(org_id="0", db_path=None)
        assert storage.db_path == str(Path(temp_dir) / "reflexio_0.db")


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


# ---------------------------------------------------------------------------
# Dataset isolation: which file an identity resolves to, and when it adopts one.
#
# Design: docs/superpowers/specs/2026-09-02-oss-dataset-isolation-fix-design.md
# ---------------------------------------------------------------------------


def _storage(org_id: str, root: str):
    with (
        patch("reflexio.server.LOCAL_STORAGE_PATH", root),
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        return SQLiteStorage(org_id=org_id, db_path=None)


def test_distinct_identities_resolve_to_distinct_files(tmp_path: Path) -> None:
    """I-SEP: two identities under one root never share a file."""
    a = _storage("tenant-a", str(tmp_path))
    b = _storage("tenant-b", str(tmp_path))
    try:
        assert a.db_path != b.db_path
    finally:
        a.conn.close()
        b.conn.close()


def test_existing_install_adopts_its_legacy_database(tmp_path: Path) -> None:
    """I-ADOPT: an installation with history never starts on an empty file."""
    legacy = tmp_path / "reflexio.db"
    seeded = _storage_at("incumbent", str(legacy))
    try:
        seeded.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, content, created_at,"
            " last_modified_timestamp) VALUES (?,?,?,?,?)",
            ("p1", "u1", "history", 1, 1),
        )
        seeded.conn.commit()
    finally:
        seeded.conn.close()

    adopted = _storage("incumbent", str(tmp_path))
    try:
        assert adopted.db_path == str(legacy)
        rows = adopted.conn.execute("SELECT content FROM profiles").fetchall()
        assert [r["content"] for r in rows] == ["history"]
    finally:
        adopted.conn.close()


def test_second_identity_does_not_adopt_a_claimed_database(tmp_path: Path) -> None:
    """I-CLAIM: adoption is first-claimer-wins, so it isolates rather than shares.

    The guard against the rule that would leave an already-commingled install
    commingled forever.
    """
    legacy = tmp_path / "reflexio.db"
    seeded = _storage_at("incumbent", str(legacy))
    try:
        seeded.conn.execute(
            "INSERT INTO profiles (profile_id, user_id, content, created_at,"
            " last_modified_timestamp) VALUES (?,?,?,?,?)",
            ("p1", "u1", "incumbent-only", 1, 1),
        )
        seeded.conn.commit()
    finally:
        seeded.conn.close()

    first = _storage("incumbent", str(tmp_path))
    first.conn.close()

    second = _storage("newcomer", str(tmp_path))
    try:
        assert second.db_path != str(legacy)
        assert (
            second.conn.execute("SELECT count(*) AS n FROM profiles").fetchone()["n"]
            == 0
        )
    finally:
        second.conn.close()

    # the incumbent's file is untouched
    reopened = _storage("incumbent", str(tmp_path))
    try:
        assert reopened.db_path == str(legacy)
        assert (
            reopened.conn.execute("SELECT count(*) AS n FROM profiles").fetchone()["n"]
            == 1
        )
    finally:
        reopened.conn.close()


def test_adoption_is_idempotent(tmp_path: Path) -> None:
    """Re-opening as the incumbent re-adopts without rewriting the claim."""
    legacy = tmp_path / "reflexio.db"
    _storage_at("incumbent", str(legacy)).conn.close()

    first = _storage("incumbent", str(tmp_path))
    first.conn.close()
    claimed_at = _claim_row(legacy)

    second = _storage("incumbent", str(tmp_path))
    try:
        assert second.db_path == str(legacy)
    finally:
        second.conn.close()
    assert _claim_row(legacy) == claimed_at


def _claim_row(path: Path):
    import sqlite3

    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT org_id, claimed_at FROM _dataset_identity WHERE k = 1"
        ).fetchone()
    finally:
        conn.close()


def _storage_at(org_id: str, db_path: str):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        return SQLiteStorage(org_id=org_id, db_path=db_path)
