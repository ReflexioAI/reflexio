"""Regression coverage for reverting the session-outcome identity schema."""

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Request,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage import _base as sqlite_base

pytestmark = pytest.mark.integration


_IDENTITY_SCHEMA = """
CREATE TABLE session_outcomes (
    outcome_id TEXT NOT NULL UNIQUE,
    outcome_revision INTEGER NOT NULL CHECK (outcome_revision >= 1),
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure', 'unknown')),
    occurred_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    label TEXT,
    value REAL,
    metadata TEXT,
    outcome_contract_digest TEXT NOT NULL,
    finalized_trajectory_digest TEXT NOT NULL,
    governance_subject_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
"""


def _storage(db_path: str) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        return SQLiteStorage(org_id="downgrade-org", db_path=db_path)


def test_identity_schema_is_downgraded_and_current_writes_resume(tmp_path) -> None:
    db_path = str(tmp_path / "identity-schema.db")
    storage = _storage(db_path)
    for session_id in ("kept", "unknown", "new"):
        storage.add_request(
            Request(
                request_id=f"request-{session_id}",
                user_id="user-1",
                session_id=session_id,
                source="test",
                created_at=100,
            )
        )
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(_IDENTITY_SCHEMA)
    storage.conn.executemany(
        """INSERT INTO session_outcomes (
               outcome_id, outcome_revision, user_id, session_id, outcome,
               occurred_at, source, outcome_contract_digest,
               finalized_trajectory_digest, governance_subject_ref, created_at
           ) VALUES (?, 1, 'user-1', ?, ?, 101, 'test', ?, ?, ?, 102)""",
        [
            ("outcome-kept", "kept", "success", "a" * 64, "b" * 64, "old-ref"),
            (
                "outcome-unknown",
                "unknown",
                "unknown",
                "c" * 64,
                "d" * 64,
                "old-ref",
            ),
        ],
    )
    storage.conn.commit()
    storage.conn.close()

    migrated = _storage(db_path)

    assert {
        row["name"]
        for row in migrated.conn.execute("PRAGMA table_info(session_outcomes)")
    } == {
        "user_id",
        "session_id",
        "outcome",
        "occurred_at",
        "source",
        "label",
        "value",
        "metadata",
        "governance_subject_ref",
        "created_at",
    }
    records = migrated.get_session_outcomes(GetSessionOutcomesRequest())
    assert [(record.session_id, record.outcome) for record in records] == [
        ("kept", SessionOutcomeKind.SUCCESS)
    ]

    request = SetSessionOutcomeRequest(
        session_id="new", outcome=SessionOutcomeKind.FAILURE, occurred_at=101
    )
    result = migrated.record_session_outcome(
        request,
        created_at=102,
        expected_context=migrated.get_session_outcome_context("new"),
    )
    assert result.recorded is True


@pytest.mark.parametrize("with_subject_column", [False, True])
def test_legacy_schema_backfills_required_subject_ref(
    tmp_path, with_subject_column: bool
) -> None:
    db_path = str(tmp_path / f"legacy-subject-{with_subject_column}.db")
    storage = _storage(db_path)
    storage.conn.execute("DROP TABLE session_outcomes")
    subject_column = "governance_subject_ref TEXT," if with_subject_column else ""
    storage.conn.executescript(
        f"""CREATE TABLE session_outcomes (
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure')),
            occurred_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            label TEXT,
            value REAL,
            metadata TEXT,
            {subject_column}
            created_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, session_id)
        );"""
    )
    columns = "user_id, session_id, outcome, occurred_at, source, created_at"
    values: tuple[object, ...] = (
        "legacy-user",
        "legacy-session",
        "success",
        101,
        "legacy-source",
        102,
    )
    if with_subject_column:
        columns += ", governance_subject_ref"
        values += (None,)
    placeholders = ", ".join("?" for _ in values)
    storage.conn.execute(
        f"INSERT INTO session_outcomes ({columns}) VALUES ({placeholders})",  # noqa: S608
        values,
    )
    storage.conn.commit()
    storage.conn.close()

    migrated = _storage(db_path)

    row = migrated.conn.execute(
        "SELECT governance_subject_ref FROM session_outcomes"
    ).fetchone()
    assert row is not None
    assert row["governance_subject_ref"] == migrated._subject_ref_for_user_id(
        "legacy-user"
    )


def test_legacy_empty_subject_default_is_rebuilt_and_backfilled(tmp_path) -> None:
    db_path = str(tmp_path / "legacy-empty-subject-default.db")
    storage = _storage(db_path)
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(
        """CREATE TABLE session_outcomes (
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure')),
            occurred_at INTEGER NOT NULL,
            source TEXT NOT NULL,
            label TEXT,
            value REAL,
            metadata TEXT,
            governance_subject_ref TEXT NOT NULL DEFAULT '',
            created_at INTEGER NOT NULL,
            PRIMARY KEY (user_id, session_id)
        );"""
    )
    storage.conn.execute(
        """INSERT INTO session_outcomes (
               user_id, session_id, outcome, occurred_at, source, created_at
           ) VALUES ('legacy-user', 'legacy-session', 'success', 101, 'legacy', 102)"""
    )
    storage.conn.commit()
    storage.conn.close()

    migrated = _storage(db_path)

    row = migrated.conn.execute(
        "SELECT governance_subject_ref FROM session_outcomes"
    ).fetchone()
    assert row is not None
    assert row["governance_subject_ref"] == migrated._subject_ref_for_user_id(
        "legacy-user"
    )


def test_sqlite_versions_without_returning_and_drop_column_are_rejected(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite_base.sqlite3, "sqlite_version_info", (3, 34, 1))

    with pytest.raises(RuntimeError, match="SQLite 3.35.0 or newer"):
        _storage(str(tmp_path / "old-sqlite.db"))
