"""Resolve which SQLite file a dataset identity owns.

Historically every caller resolved to one file, ``<LOCAL_STORAGE_PATH>/reflexio.db``,
regardless of the identity it was constructed with. Of the persistent tables only 11
carry an ``org_id`` column; the other 32 -- ``profiles``, ``requests``,
``interactions``, ``user_playbooks`` among them -- have none, and their reads are
unscoped. Two identities sharing that file therefore read each other's rows.

The sharpest case is not two local plugins: a self-host deployment never passes
``base_dir``, so *every* org it serves resolved to the same file.

This module derives the file from the identity instead, and adopts an existing
database rather than starting empty next to it.

Adoption is **first claimer wins**, not "adopt whatever is there". Adopting on every
open would let a second identity attach to the same file, leaving an already-commingled
installation commingled forever -- and those installations are the only ones with the
bug. See ``docs/superpowers/specs/2026-09-02-oss-dataset-isolation-fix-design.md``.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

LEGACY_DB_FILENAME = "reflexio.db"

IDENTITY_CLAIM_TABLE = "_dataset_identity"

# Tables carrying an ``org_id`` column. Read only to decide whether an unclaimed
# legacy file plausibly belongs to the opening identity; a tie-breaker, not a
# decision procedure -- several are written rarely or not at all.
_ATTRIBUTED_TABLES = (
    "_agent_runs",
    "_pending_tool_calls",
    "audit_events",
    "braintrust_connection",
    "imported_score",
    "learning_jobs",
    "lineage_event",
    "purge_operation_targets",
    "purge_operations",
    "share_links",
    "subject_write_barriers",
)

#: The erasure write-barrier table, read on its own by
#: :func:`barrier_identity_labels`. Named separately because a foreign label
#: HERE refuses adoption outright, while a foreign label in any other attributed
#: table only warns.
_WRITE_BARRIER_TABLE = "subject_write_barriers"

# A dataset identity becomes a path component, so it is constrained to characters
# that cannot traverse or escape. Rejected, never rewritten: slugifying would map
# two distinct identities onto one file, which is the defect this module exists to
# fix.
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class DatasetIdentityError(RuntimeError):
    """A dataset file cannot be resolved for the requested identity."""


def validate_dataset_identity(org_id: str) -> str:
    """Return *org_id* if it is usable as a filename component.

    Args:
        org_id (str): The caller-supplied dataset identity.

    Returns:
        str: The identity, unchanged.

    Raises:
        DatasetIdentityError: If it could traverse, escape, or collide.
    """
    if not _IDENTITY_RE.fullmatch(org_id or ""):
        raise DatasetIdentityError(
            f"dataset identity {org_id!r} cannot be used as a storage filename: "
            "it must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$. It is rejected rather "
            "than rewritten, because rewriting could map two identities onto one file."
        )
    return org_id


def derive_db_path(root: str | Path, org_id: str) -> Path:
    """Return the file *org_id* owns under *root*.

    A flat filename rather than a subdirectory, so sibling artifacts in the same
    directory -- the enterprise ``sql_app.db``, the ``disk_*`` trees -- are untouched.

    Args:
        root (str | Path): The storage root directory.
        org_id (str): The dataset identity.

    Returns:
        Path: The derived database path.
    """
    validate_dataset_identity(org_id)
    return Path(root) / f"reflexio_{org_id}.db"


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def stored_identity_labels(path: Path) -> set[str] | None:
    """Return the identities appearing in *path*'s attributed tables.

    Every read is guarded by ``PRAGMA table_info``: no migration adds ``org_id`` to
    these tables, so one created by a build predating the column never gains it, and
    an unguarded read would raise ``no such column``.

    Args:
        path (Path): The database to inspect.

    Returns:
        set[str] | None: The labels found, or ``None`` if the file carries no
        readable attribution at all.
    """
    if not path.exists():
        return None
    labels: set[str] = set()
    saw_column = False
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        for table in _ATTRIBUTED_TABLES:
            if not _table_exists(conn, table):
                continue
            columns = {
                str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")
            }
            if "org_id" not in columns:
                continue
            saw_column = True
            for (value,) in conn.execute(
                f"SELECT DISTINCT org_id FROM {table} WHERE org_id IS NOT NULL"  # noqa: S608
            ):
                if value:
                    labels.add(str(value))
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return labels if saw_column else None


def barrier_identity_labels(path: Path) -> set[str]:
    """Identities appearing in *path*'s erasure write-barrier table.

    A sibling of :func:`stored_identity_labels` rather than a change to it,
    because the two answer different questions and the difference decides
    whether a commingled file may be adopted at all. ``stored_identity_labels``
    unions eleven tables into one flat set, which is the right answer for "does
    this file plausibly belong to us" and the wrong one for "whose erasure state
    would we be taking custody of".

    ``subject_write_barriers`` is the sharp case: a barrier is a standing refusal
    to write for an erased subject. Adopting a file holding another identity's
    barriers means this process becomes the one enforcing them -- or, far worse,
    silently not enforcing them for rows it does not know are barred.

    Returns:
        set[str]: Labels found, empty when the table is absent or unreadable.
        Deliberately not ``None``: absence of evidence here must not read as a
        foreign label, or an old file would become unopenable.
    """
    if not path.exists():
        return set()
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error:
        return set()
    labels: set[str] = set()
    try:
        if not _table_exists(conn, _WRITE_BARRIER_TABLE):
            return set()
        columns = {
            str(row[1])
            for row in conn.execute(f"PRAGMA table_info({_WRITE_BARRIER_TABLE})")
        }
        if "org_id" not in columns:
            return set()
        for (value,) in conn.execute(
            f"SELECT DISTINCT org_id FROM {_WRITE_BARRIER_TABLE} "  # noqa: S608
            f"WHERE org_id IS NOT NULL"
        ):
            if value:
                labels.add(str(value))
    except sqlite3.Error:
        return set()
    finally:
        conn.close()
    return labels


def claim_or_read_identity(path: Path, org_id: str) -> str:
    """Claim *path* for *org_id*, or return the identity that already holds it.

    The claim is a row rather than a marker file or a PRAGMA: ``application_id`` and
    ``user_version`` are 32-bit integers and cannot hold an identity string, and a
    sidecar file can desync from the database it describes.

    ``BEGIN IMMEDIATE`` takes a cross-process write lock, which is required rather
    than defensive -- the service runs multiple uvicorn workers by default, and the
    in-process initialization lock in ``_base`` is a ``threading.Lock``.

    Args:
        path (Path): The database to claim.
        org_id (str): The identity claiming it.

    Returns:
        str: The identity that owns *path* -- *org_id* if the claim succeeded.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {IDENTITY_CLAIM_TABLE} ("
            " k INTEGER PRIMARY KEY CHECK (k = 1),"
            " org_id TEXT NOT NULL,"
            " claimed_at TEXT NOT NULL)"
        )
        conn.execute(
            f"INSERT INTO {IDENTITY_CLAIM_TABLE} (k, org_id, claimed_at)"
            " VALUES (1, ?, ?) ON CONFLICT(k) DO NOTHING",
            (org_id, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        )
        row = conn.execute(
            f"SELECT org_id FROM {IDENTITY_CLAIM_TABLE} WHERE k = 1"
        ).fetchone()
        conn.commit()
    finally:
        conn.close()
    return str(row[0]) if row else org_id


def resolve_sqlite_db_path(root: str | Path, org_id: str) -> str:
    """Return the SQLite file *org_id* should open under *root*.

    Adopts an existing unclaimed database so an installation with history never
    starts on an empty file; refuses to adopt one another identity already holds.

    Args:
        root (str | Path): The storage root directory.
        org_id (str): The dataset identity.

    Returns:
        str: The resolved database path.

    Raises:
        DatasetIdentityError: If the identity is unusable, or its own derived file
            is already claimed by a different identity.
    """
    validate_dataset_identity(org_id)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    derived = derive_db_path(root_path, org_id)
    legacy = root_path / LEGACY_DB_FILENAME

    if derived.exists():
        owner = claim_or_read_identity(derived, org_id)
        if owner != org_id:
            # Reachable on a case-insensitive filesystem, where two legal
            # identities differing only in case derive one filename.
            raise DatasetIdentityError(
                f"{derived} is already claimed by dataset {owner!r}, but was opened "
                f"as {org_id!r}. Refusing to share one file between two identities."
            )
        return str(derived)

    if not legacy.exists():
        claim_or_read_identity(derived, org_id)
        return str(derived)

    labels = stored_identity_labels(legacy)
    if labels and org_id not in labels:
        logger.warning(
            "Not adopting %s for dataset %r: it holds rows labelled %s. Using %s instead.",
            legacy,
            org_id,
            sorted(labels),
            derived,
        )
        claim_or_read_identity(derived, org_id)
        return str(derived)

    # COMMINGLED: our label AND someone else's. The check above only refuses when
    # we are ABSENT, so this file -- the self-host multi-org install this module
    # calls the sharpest case -- used to be adopted wholesale. The adopter then
    # read the other identity's rows across every attributed table, and the 28
    # tenant tables carry no `org_id` column at all, so there was nothing
    # downstream to scope them. The other identity's history was simultaneously
    # orphaned in a file it would never open again, and the only log line emitted
    # named the loser, not the adopter.
    foreign = (labels or set()) - {org_id}
    if foreign:
        barrier_foreign = barrier_identity_labels(legacy) - {org_id}
        if barrier_foreign:
            # Erasure state is not adoptable. Taking custody of another
            # identity's write barriers means either enforcing refusals we
            # cannot attribute, or silently not enforcing them -- and an erasure
            # that stops being enforced is not recoverable by noticing later.
            raise DatasetIdentityError(
                f"{legacy} holds erasure write barriers for dataset(s) "
                f"{sorted(barrier_foreign)} as well as {org_id!r}. Refusing to "
                f"adopt a file carrying another identity's erasure state. Split "
                f"the file per identity before opening it."
            )
        logger.warning(
            "Adopting %s for dataset %r, but it also holds rows labelled %s. "
            "Those rows are readable by %r and will not be visible to their own "
            "dataset. Split the file per identity to separate them.",
            legacy,
            org_id,
            sorted(foreign),
            org_id,
        )

    owner = claim_or_read_identity(legacy, org_id)
    if owner == org_id:
        return str(legacy)

    logger.warning(
        "%s is claimed by dataset %r; opening %s for %r instead.",
        legacy,
        owner,
        derived,
        org_id,
    )
    claim_or_read_identity(derived, org_id)
    return str(derived)
