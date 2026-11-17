"""SQLite dataset creation proof owned by the existing path/constructor seam.

Only exclusive atomic publication of a newly created file may choose its policy.
The retained SEALED root precedes ordinary migrations. Missing roots in existing
files never become a fresh install, even when their application tables are empty.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from pathlib import Path

from reflexio.profile_authoring_protocol import (
    PROFILE_AUTHORING_POLICY,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningPolicyAuthority,
    LearningTransitionBlockedError,
    _change_learning_policy_authority_db,
)

AUTHORITY_DDL = """CREATE TABLE IF NOT EXISTS _learning_policy_authority (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    authority TEXT NOT NULL CHECK (json_valid(authority))
)"""


def _provenance(path: str, org_id: str, nonce: str) -> str:
    identity: list[str | int] = ["sqlite-first-dataset", org_id, nonce]
    if path == ":memory:":
        identity.append("private-memory-connection")
    else:
        stat = Path(path).stat()
        identity.extend((stat.st_dev, stat.st_ino))
    return hashlib.sha256(
        json.dumps(identity, separators=(",", ":")).encode()
    ).hexdigest()


def _retain_creation(conn: sqlite3.Connection, path: str, org_id: str) -> None:
    nonce = uuid.uuid4().hex
    root = LearningPolicyAuthority(
        1,
        uuid.uuid4().hex,
        0,
        "sealed",
        PROFILE_AUTHORING_POLICY,
        PROFILE_AUTHORING_POLICY,
        "sqlite-first:" + nonce,
        "first_dataset",
        nonce,
        _provenance(path, org_id, nonce),
        True,
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(AUTHORITY_DDL)
        conn.execute(
            "INSERT INTO _learning_policy_authority(singleton,authority) VALUES(1,?)",
            (json.dumps(root.to_dict(), sort_keys=True, separators=(",", ":")),),
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def reserve_new_sqlite_dataset(path: str, org_id: str) -> None:
    """Publish a self-contained SEALED file with exclusive create semantics.

    A private same-directory file retains the proof and is checkpointed/closed
    before atomic hard-link publication. Losing creators discard only their own
    unpublished file. No opener can see an authority-less creation reservation.
    Existing actual targets are never replaced or initialized by this helper.
    """
    if path == ":memory:":
        return  # The constructor owns the actual private connection instead.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return  # Optimization only: the atomic link below establishes ownership.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.learning-policy.", delete=False
        ) as reservation:
            temporary = Path(reservation.name)
        conn = sqlite3.connect(temporary)
        try:
            if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError("SQLite creation requires its supported WAL owner")
            _retain_creation(conn, str(temporary), org_id)
            if conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
                raise RuntimeError("SQLite creation proof could not be checkpointed")
        finally:
            conn.close()
        try:
            os.link(temporary, target)
        except FileExistsError:
            return
        # Match the maintained config-file publisher's directory durability.
        # Windows has no POSIX directory fsync; publication remains atomic.
        if os.name == "posix":
            descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            for suffix in ("", "-wal", "-shm", "-journal"):
                Path(str(temporary) + suffix).unlink(missing_ok=True)


def retain_private_memory_creation(conn: sqlite3.Connection, org_id: str) -> None:
    """SQLite guarantees that this fresh private ':memory:' connection is new."""
    _retain_creation(conn, ":memory:", org_id)


def complete_sqlite_dataset_creation(
    conn: sqlite3.Connection, path: str, org_id: str
) -> None:
    """Resume only matching retained first-creation proof after normal migration.

    No authority in an old file is initialized here. A crash before the retained
    proof needs explicit upgrade/recovery; a retained matching proof can retry.
    """
    from ..storage_base.transition_closure import derive_project_transition_closure_db
    from ._extraction_stream import SQLiteStreamSQL

    conn.execute(AUTHORITY_DDL)
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        db = SQLiteStreamSQL(conn)
        raw = db.read_learning_policy_authority()
        if raw is None:
            conn.commit()
            return
        authority = LearningPolicyAuthority.from_dict(raw)
        if not authority.bootstrap or authority.initialization_kind != "first_dataset":
            conn.commit()
            return
        if (
            authority.provenance_sha256
            != _provenance(path, org_id, authority.initialization_nonce)
            or authority.active_policy != PROFILE_AUTHORING_POLICY
            or authority.target_policy != PROFILE_AUTHORING_POLICY
            or authority.operation_id
            != "sqlite-first:" + authority.initialization_nonce
        ):
            raise LearningTransitionBlockedError(
                "SQLite first-dataset initialization requires its retained source/file owner"
            )
        closure = derive_project_transition_closure_db(db, org_id=org_id, project_id="")
        if not closure.ready or closure.blockers:
            raise LearningTransitionBlockedError(
                "SQLite first-dataset initialization closure is incomplete"
            )
        if authority.phase == "sealed":
            authority = _change_learning_policy_authority_db(
                db,
                action="arm",
                operation_id=authority.operation_id or "",
                expected_nonce=authority.authority_nonce,
                expected_generation=authority.generation,
            )
        if authority.phase != "armed":
            raise LearningTransitionBlockedError(
                "SQLite first-dataset initialization is not held"
            )
        _change_learning_policy_authority_db(
            db,
            action="activate",
            operation_id=authority.operation_id or "",
            expected_nonce=authority.authority_nonce,
            expected_generation=authority.generation,
        )
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
