"""Tests for ``clear-all`` target resolution.

``_resolve_clear_all_targets`` had no coverage: every existing ``clear-all``
test patches it out and exercises only the command wrapper around it. That is
how it kept returning the whole storage root as one directory target long after
dataset isolation started putting one ``reflexio_<org>.db`` per identity in that
root -- so clearing one identity destroyed every other identity's database, plus
any sibling artifact (the enterprise ``sql_app.db``, the ``disk_*`` trees) that
``derive_db_path`` documents as untouched.

These tests build a real root on disk and assert what survives, not just what
goes.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from openclaw_smart import cli

OTHER_ORG = "other-org"
OUR_ORG = "self-host-org"


@pytest.fixture
def root(monkeypatch, tmp_path) -> Path:
    """An isolated storage root, with no reflexio config on disk."""
    storage = tmp_path / "data"
    storage.mkdir()
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(storage))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", OUR_ORG)
    # _load_reflexio_config() reads a module-level path; point it at nothing so
    # these cases exercise the default (unconfigured) branch.
    monkeypatch.setattr(cli, "_REFLEXIO_CONFIG_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(cli, "_REFLEXIO_ENV_PATH", tmp_path / "absent.env")
    return storage


def _make_db(path: Path, claimed_by: str | None) -> None:
    """Create a SQLite file, optionally carrying a dataset identity claim."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS junk (x INTEGER)")
        if claimed_by is not None:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS _dataset_identity ("
                " k INTEGER PRIMARY KEY CHECK (k = 1),"
                " org_id TEXT NOT NULL,"
                " claimed_at TEXT NOT NULL)"
            )
            conn.execute(
                "INSERT INTO _dataset_identity (k, org_id, claimed_at)"
                " VALUES (1, ?, '2026-01-01T00:00:00Z')",
                (claimed_by,),
            )
        conn.commit()
    finally:
        conn.close()


def _paths(targets) -> set[Path]:
    return {t.path for t in targets}


def _clear() -> None:
    """Resolve and actually remove, so assertions can look at the filesystem.

    Asserting a path is merely absent from the target list passes against the
    bug too -- the root target destroys files it never names.
    """
    for target in cli._resolve_clear_all_targets():
        cli._remove_clear_all_target(target)


def test_does_not_target_the_storage_root_itself(root):
    """The root is shared; deleting it as a directory is the whole bug."""
    _make_db(root / f"reflexio_{OUR_ORG}.db", claimed_by=OUR_ORG)

    targets = cli._resolve_clear_all_targets()

    assert root.resolve() not in _paths(targets)


def test_targets_our_database_and_its_sidecars(root):
    our_db = root / f"reflexio_{OUR_ORG}.db"
    _make_db(our_db, claimed_by=OUR_ORG)
    for suffix in ("-wal", "-shm", "-journal"):
        Path(f"{our_db}{suffix}").write_text("")

    paths = _paths(cli._resolve_clear_all_targets())

    assert our_db.resolve() in paths
    for suffix in ("-wal", "-shm", "-journal"):
        assert Path(f"{our_db}{suffix}").resolve() in paths


def test_spares_another_identitys_database(root):
    """The regression this file exists for."""
    ours = root / f"reflexio_{OUR_ORG}.db"
    _make_db(ours, claimed_by=OUR_ORG)
    theirs = root / f"reflexio_{OTHER_ORG}.db"
    _make_db(theirs, claimed_by=OTHER_ORG)

    _clear()

    assert theirs.exists(), "another identity's database was destroyed"
    assert not ours.exists(), "our own database should still be cleared"


def test_spares_sibling_artifacts_in_the_same_root(root):
    """``derive_db_path`` promises these are untouched; honor that here too."""
    _make_db(root / f"reflexio_{OUR_ORG}.db", claimed_by=OUR_ORG)
    enterprise = root / "sql_app.db"
    _make_db(enterprise, claimed_by=None)
    stray = root / "notes.txt"
    stray.write_text("keep me")

    _clear()

    assert enterprise.exists(), "the enterprise database was destroyed"
    assert stray.read_text() == "keep me"
    assert root.exists(), "the shared storage root itself was destroyed"


def test_adopts_a_legacy_database_this_identity_claimed(root):
    """Upgraders keep using ``reflexio.db``; clearing must still reach it."""
    legacy = root / "reflexio.db"
    _make_db(legacy, claimed_by=OUR_ORG)

    paths = _paths(cli._resolve_clear_all_targets())

    assert legacy.resolve() in paths


def test_spares_a_legacy_database_another_identity_claimed(root):
    legacy = root / "reflexio.db"
    _make_db(legacy, claimed_by=OTHER_ORG)

    _clear()

    assert legacy.exists(), "a legacy database owned by another identity was destroyed"


def test_targets_an_unclaimed_legacy_database(root):
    """No claim row means nobody has opened it since isolation landed.

    That is the pre-upgrade file this installation would adopt on its next
    start, so it is ours to clear.
    """
    legacy = root / "reflexio.db"
    _make_db(legacy, claimed_by=None)

    paths = _paths(cli._resolve_clear_all_targets())

    assert legacy.resolve() in paths


def test_resolution_creates_nothing(root):
    """Resolution must be read-only: no mkdir, no claim, no empty database.

    ``resolve_sqlite_db_path`` upstream deliberately mutates (it mkdirs the root
    and writes a claim row). Reusing it here would create a database in order to
    delete it.
    """
    before = {p.name for p in root.iterdir()}

    cli._resolve_clear_all_targets()

    assert {p.name for p in root.iterdir()} == before


def test_missing_root_resolves_without_creating_it(monkeypatch, tmp_path):
    absent = tmp_path / "never-created"
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(absent))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", OUR_ORG)
    monkeypatch.setattr(cli, "_REFLEXIO_CONFIG_PATH", tmp_path / "absent.json")
    monkeypatch.setattr(cli, "_REFLEXIO_ENV_PATH", tmp_path / "absent.env")

    cli._resolve_clear_all_targets()

    assert not absent.exists()


def test_derived_filename_matches_the_canonical_resolver():
    """Anti-drift guard for the one thing this module duplicates.

    The plugin derives ``reflexio_<org>.db`` itself rather than importing
    ``_dataset_path``: that import costs ~1.7s through the storage package's
    ``__init__``, and ``openclaw-smart-hook`` runs per session event. The
    duplication is only safe while the two agree, so assert it against the
    canonical implementation. Tests may pay the import cost the CLI cannot.
    """
    # Importing `reflexio` runs its dotenv loader, which writes REFLEXIO_URL and
    # friends into os.environ. Left alone that leaks into every later test in
    # the run -- it silently broke test_reflexio_adapter's default-URL case,
    # which passes in isolation and failed only in a full-suite run.
    original_environ = dict(os.environ)
    try:
        from reflexio.server.services.storage.sqlite_storage._dataset_path import (
            LEGACY_DB_FILENAME,
            derive_db_path,
        )

        assert cli._LEGACY_DB_FILENAME == LEGACY_DB_FILENAME
        for org in ("self-host-org", "acme", "a.b-c_1"):
            assert cli._derive_db_filename(org) == derive_db_path("/root", org).name
    finally:
        os.environ.clear()
        os.environ.update(original_environ)
