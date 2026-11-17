"""New-file provisioning and explicit qualified SQLite policy establishment.

New files retain a SEALED creation root before ordinary migrations. Existing
files require the explicit stopped-host upgrade owner; normal construction never
chooses their policy, even when application tables are empty.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class SQLiteDatasetUpgradeResult:
    """A new establishment has provenance; a retry retains only its digest."""

    authority: LearningPolicyAuthority
    established: bool
    provenance: dict[str, Any] | None


def _upgrade_refused(reason: str) -> LearningTransitionBlockedError:
    return LearningTransitionBlockedError(
        f"Local dataset upgrade refused: {reason}. "
        "Keep the original dataset and use its supported recovery owner; "
        "this command never repairs or discards retained work."
    )


def _catalog(conn: sqlite3.Connection, limit: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT type,name,tbl_name AS 'table',sql FROM sqlite_schema "
            "ORDER BY type,name LIMIT ?",
            (limit,),
        )
    ]


def _owned_identifier(name: str) -> str:
    # Only names from the packaged, independently qualified schema contract.
    return '"' + name.replace('"', '""') + '"'


def _check_upgrade_schema(conn: sqlite3.Connection, contract: dict[str, Any]) -> bool:
    expected = contract["catalog"]
    observed = _catalog(conn, len(expected) + 2)
    authority_objects = [
        row for row in observed if row["table"] == "_learning_policy_authority"
    ]
    if authority_objects:
        reference = sqlite3.connect(":memory:")
        reference.row_factory = sqlite3.Row
        try:
            reference.execute(AUTHORITY_DDL)
            if authority_objects != _catalog(reference, 2):
                raise _upgrade_refused("unsupported authority schema")
        finally:
            reference.close()
    if [row for row in observed if row not in authority_objects] != expected:
        raise _upgrade_refused("unsupported historical schema")
    for name, shape in contract["tables"].items():
        if shape.get("virtual"):
            continue  # Exact catalog/ownership proof, never query a virtual parent.
        columns = [
            row["name"]
            for row in conn.execute(
                f"PRAGMA table_info({_owned_identifier(name)})"  # noqa: S608 -- Packaged schema names, not observed identifiers.
            )
        ]
        if columns != shape["columns"]:
            raise _upgrade_refused("unsupported historical columns")
    return bool(authority_objects)


def _upgrade_identity(conn: sqlite3.Connection) -> str:
    from ._dataset_path import DatasetIdentityError, validate_dataset_identity

    rows = conn.execute("SELECT k,org_id FROM _dataset_identity LIMIT 2").fetchall()
    if len(rows) != 1 or rows[0]["k"] != 1:
        raise _upgrade_refused("retained identity is unavailable")
    org_id = rows[0]["org_id"]
    if not isinstance(org_id, str):
        raise _upgrade_refused("retained identity is invalid")
    try:
        return validate_dataset_identity(org_id)
    except DatasetIdentityError:
        raise _upgrade_refused("retained identity is invalid") from None


def _check_upgrade_scope(
    conn: sqlite3.Connection, contract: dict[str, Any], org_id: str
) -> None:
    for name in contract["ownership"]["direct_scope"]:
        columns = contract["tables"][name]["columns"]
        predicates = []
        parameters = []
        if "org_id" in columns:
            predicates.append("org_id IS NULL OR org_id != ?")
            parameters.append(org_id)
        if "project_id" in columns:
            predicates.append("project_id IS NULL OR project_id != ''")
        if conn.execute(
            f"SELECT 1 FROM {_owned_identifier(name)} WHERE "  # noqa: S608 -- Packaged schema names; predicates are fixed above.
            + " OR ".join(predicates)
            + " LIMIT 1",
            parameters,
        ).fetchone():
            raise _upgrade_refused("retained scope differs from its identity")
    known_tables = tuple(contract["tables"])
    placeholders = ",".join("?" for _ in known_tables)
    if conn.execute(
        f"SELECT 1 FROM sqlite_sequence WHERE name NOT IN ({placeholders}) LIMIT 1",  # noqa: S608 -- Parameter placeholders only.
        known_tables,
    ).fetchone():
        raise _upgrade_refused("unknown sequence owner")


def upgrade_existing_sqlite_dataset(path: Path) -> SQLiteDatasetUpgradeResult:
    """Explicitly establish future policy on one qualified, stopped local cohort.

    Stop affected services/coding hosts and filesystem changes first. This never
    invokes normal storage construction, migrations, operator env or inference.
    Unknown schemas and retained independent work remain held, without cleanup.
    """
    from ..storage_base.transition_closure import derive_project_transition_closure_db
    from ._extraction_stream import SQLiteStreamSQL

    if not path.is_absolute():
        raise _upgrade_refused("an explicit absolute existing file is required")
    target = path.resolve(strict=True)
    if not target.is_file():
        raise _upgrade_refused("an existing regular file is required")
    original = target.stat()
    contract_bytes = (
        Path(__file__).with_name("_historical_schema_70daf32c.json").read_bytes()
    )
    contract = json.loads(contract_bytes)
    conn = sqlite3.connect(target.as_uri() + "?mode=rw", uri=True)
    conn.row_factory = sqlite3.Row

    def check_target() -> None:
        current = target.stat()
        if (current.st_dev, current.st_ino) != (original.st_dev, original.st_ino):
            raise _upgrade_refused("the selected file changed")

    try:
        check_target()
        conn.execute("BEGIN IMMEDIATE")
        check_target()
        has_authority = _check_upgrade_schema(conn, contract)
        org_id = _upgrade_identity(conn)
        _check_upgrade_scope(conn, contract, org_id)
        db = SQLiteStreamSQL(conn)
        raw = db.read_learning_policy_authority() if has_authority else None
        if raw is not None:
            retained = LearningPolicyAuthority.from_dict(raw)
            if (
                retained.phase != "open"
                or retained.active_policy != PROFILE_AUTHORING_POLICY
            ):
                raise _upgrade_refused("retained authority needs its transition owner")
            check_target()
            conn.commit()
            return SQLiteDatasetUpgradeResult(retained, False, None)
        for name in contract["empty_controls"]:
            if conn.execute(
                f"SELECT 1 FROM {_owned_identifier(name)} LIMIT 1"  # noqa: S608 -- Exact packaged ordinary-control table list.
            ).fetchone():
                raise _upgrade_refused(f"retained independent control {name}")
        stall = conn.execute("SELECT id,stalled FROM stall_state LIMIT 2").fetchall()
        if len(stall) != 1 or tuple(stall[0]) != (1, 0):
            raise _upgrade_refused("retained learning is stalled or unknown")
        closure = derive_project_transition_closure_db(db, org_id=org_id, project_id="")
        if not closure.ready or closure.blockers:
            raise _upgrade_refused(
                "retained execution or financial closure is incomplete"
            )
        nonce = uuid.uuid4().hex
        provenance = {
            "protocol": "sqlite-qualified-upgrade-v1",
            "source_schema_sha256": hashlib.sha256(contract_bytes).hexdigest(),
            "org_id": org_id,
            "file_identity": [original.st_dev, original.st_ino],
            "policy": PROFILE_AUTHORING_POLICY,
            "initialization_nonce": nonce,
            "closure_counts": closure.counts,
        }
        root = LearningPolicyAuthority(
            1,
            uuid.uuid4().hex,
            0,
            "sealed",
            PROFILE_AUTHORING_POLICY,
            PROFILE_AUTHORING_POLICY,
            "sqlite-upgrade:" + nonce,
            "qualified_upgrade",
            nonce,
            hashlib.sha256(
                json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            True,
        )
        conn.execute(AUTHORITY_DDL)
        conn.execute(
            "INSERT INTO _learning_policy_authority(singleton,authority) VALUES(1,?)",
            (json.dumps(root.to_dict(), sort_keys=True, separators=(",", ":")),),
        )
        for action in ("arm", "activate"):
            root = _change_learning_policy_authority_db(
                db,
                action=action,
                operation_id="sqlite-upgrade:" + nonce,
                expected_nonce=root.authority_nonce,
                expected_generation=root.generation,
            )
        check_target()
        if _upgrade_identity(conn) != org_id:
            raise _upgrade_refused("the retained identity changed")
        _check_upgrade_scope(conn, contract, org_id)
        if (
            LearningPolicyAuthority.from_dict(db.read_learning_policy_authority())
            != root
        ):
            raise _upgrade_refused("the published authority changed")
        conn.commit()
        return SQLiteDatasetUpgradeResult(root, True, provenance)
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


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
