"""Unit coverage for dataset-file resolution.

Design: ``docs/superpowers/specs/2026-09-02-oss-dataset-isolation-fix-design.md``
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from reflexio.server.services.storage.sqlite_storage._dataset_path import (
    DatasetIdentityError,
    claim_or_read_identity,
    derive_db_path,
    resolve_sqlite_db_path,
    stored_identity_labels,
    validate_dataset_identity,
)


@pytest.mark.parametrize(
    "identity",
    ["a/b", "../escape", "/absolute", "", ".hidden", "-leading", "x" * 65, "a b"],
)
def test_unusable_identities_are_rejected(identity: str) -> None:
    """I-VALID: rejected, never rewritten.

    Rewriting could map two distinct identities onto one file, which is the defect
    this module exists to prevent.
    """
    with pytest.raises(DatasetIdentityError):
        validate_dataset_identity(identity)


@pytest.mark.parametrize(
    "identity", ["0", "acme", "self-host-org", "claude-smart", "a.b_c-1"]
)
def test_realistic_identities_are_accepted(identity: str) -> None:
    assert validate_dataset_identity(identity) == identity


def test_derived_path_stays_inside_the_root(tmp_path: Path) -> None:
    assert derive_db_path(tmp_path, "acme").parent == tmp_path


def test_claim_is_first_writer_wins(tmp_path: Path) -> None:
    db = tmp_path / "d.db"
    assert claim_or_read_identity(db, "first") == "first"
    assert claim_or_read_identity(db, "second") == "first"


def test_derived_file_claimed_by_another_identity_fails_closed(tmp_path: Path) -> None:
    """The case-fold collision: two legal identities, one filename.

    On a case-insensitive filesystem ``Acme`` and ``acme`` derive the same path, and
    no charset validator can catch it -- the claim has to.
    """
    derived = derive_db_path(tmp_path, "acme")
    claim_or_read_identity(derived, "someone-else")
    with pytest.raises(DatasetIdentityError, match="already claimed"):
        resolve_sqlite_db_path(tmp_path, "acme")


def test_fresh_install_uses_the_derived_path(tmp_path: Path) -> None:
    resolved = resolve_sqlite_db_path(tmp_path, "acme")
    assert resolved == str(tmp_path / "reflexio_acme.db")


def test_legacy_file_with_foreign_labels_is_not_adopted(tmp_path: Path) -> None:
    """I-LABEL: route around it and warn, rather than refuse to start."""
    legacy = tmp_path / "reflexio.db"
    conn = sqlite3.connect(legacy)
    try:
        conn.execute(
            "CREATE TABLE share_links (org_id TEXT NOT NULL, token TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO share_links VALUES ('someone-else', 'shr_x')")
        conn.commit()
    finally:
        conn.close()

    resolved = resolve_sqlite_db_path(tmp_path, "acme")
    assert resolved == str(tmp_path / "reflexio_acme.db")


def test_label_scan_tolerates_a_table_without_the_column(tmp_path: Path) -> None:
    """A table created before ``org_id`` existed never gains it.

    No migration adds the column, so an unguarded read would raise
    ``no such column`` on a sufficiently old file.
    """
    legacy = tmp_path / "reflexio.db"
    conn = sqlite3.connect(legacy)
    try:
        conn.execute("CREATE TABLE share_links (token TEXT NOT NULL)")
        conn.commit()
    finally:
        conn.close()

    assert stored_identity_labels(legacy) is None


def test_label_scan_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert stored_identity_labels(tmp_path / "absent.db") is None


def _claim_in_subprocess(args: tuple[str, str]) -> str:
    path, org_id = args
    from reflexio.server.services.storage.sqlite_storage._dataset_path import (
        claim_or_read_identity,
    )

    return claim_or_read_identity(Path(path), org_id)


def test_concurrent_claims_across_processes_agree_on_one_winner(tmp_path: Path) -> None:
    """The in-process initialization lock is a ``threading.Lock``.

    The service runs multiple uvicorn workers by default, so two *processes* can race
    to claim an unclaimed file. ``BEGIN IMMEDIATE`` is what makes that safe.
    """
    from concurrent.futures import ProcessPoolExecutor

    db = tmp_path / "contended.db"
    identities = [f"tenant-{i}" for i in range(6)]

    with ProcessPoolExecutor(max_workers=3) as pool:
        winners = set(
            pool.map(_claim_in_subprocess, [(str(db), i) for i in identities])
        )

    assert len(winners) == 1, f"claims disagreed on the owner: {winners}"

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("SELECT org_id FROM _dataset_identity").fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] in identities
