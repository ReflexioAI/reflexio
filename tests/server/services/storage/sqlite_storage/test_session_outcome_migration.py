"""Regression coverage for the SQLite session-outcome identity migration."""

import json
from hashlib import sha256

import pytest

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Request,
    SessionOutcomeKind,
)
from reflexio.server.services.storage.session_outcome_identity import (
    outcome_contract_digest,
    trajectory_digest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import (
    _canonical_session_snapshot,
)

pytestmark = pytest.mark.integration

_LEGACY_SESSION_OUTCOMES_DDL = """
CREATE TABLE session_outcomes (
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('success', 'failure')),
    occurred_at INTEGER NOT NULL,
    source TEXT NOT NULL,
    label TEXT,
    value REAL,
    metadata TEXT,
    governance_subject_ref TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
"""


def _legacy_session_outcomes_ddl(*, with_subject_column: bool) -> str:
    subject_column = "governance_subject_ref TEXT," if with_subject_column else ""
    return f"""
CREATE TABLE session_outcomes (
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
);
"""


def _identity_complete_session_outcomes_ddl(*, with_subject_column: bool) -> str:
    subject_column = "governance_subject_ref TEXT," if with_subject_column else ""
    return f"""
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
    {subject_column}
    created_at INTEGER NOT NULL,
    PRIMARY KEY (user_id, session_id)
);
"""


def test_migration_preserves_populated_legacy_outcomes_with_unambiguous_ids(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "legacy-session-outcomes.db")
    storage = SQLiteStorage(org_id="legacy-session-outcomes", db_path=db_path)
    legacy_rows = [
        {
            "user_id": "a:b",
            "session_id": "c",
            "outcome": "success",
            "occurred_at": 101,
            "source": "legacy-source-one",
            "label": "resolved:one",
            "value": 2.5,
            "metadata": {"nested": {"a": 1}, "legacy": True},
            "governance_subject_ref": "subject:a:b:c",
            "created_at": 102,
        },
        {
            "user_id": "a",
            "session_id": "b:c",
            "outcome": "failure",
            "occurred_at": 201,
            "source": "legacy-source-two",
            "label": None,
            "value": None,
            "metadata": None,
            "governance_subject_ref": "subject:a:b:c:two",
            "created_at": 202,
        },
    ]
    for row in legacy_rows:
        storage.add_request(
            Request(
                request_id=f"request-{row['session_id']}",
                user_id=row["user_id"],
                session_id=row["session_id"],
                source=row["source"],
                created_at=row["occurred_at"] - 1,
            )
        )
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(_LEGACY_SESSION_OUTCOMES_DDL)
    for row in legacy_rows:
        storage.conn.execute(
            """INSERT INTO session_outcomes (
                user_id, session_id, outcome, occurred_at, source, label, value,
                metadata, governance_subject_ref, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["user_id"],
                row["session_id"],
                row["outcome"],
                row["occurred_at"],
                row["source"],
                row["label"],
                row["value"],
                json.dumps(row["metadata"], sort_keys=True)
                if row["metadata"] is not None
                else None,
                row["governance_subject_ref"],
                row["created_at"],
            ),
        )
    storage.conn.commit()
    storage.conn.close()

    migrated = SQLiteStorage(org_id="legacy-session-outcomes", db_path=db_path)
    records = migrated.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["c", "b:c"])
    )
    records_by_session = {record.session_id: record for record in records}
    rows_by_session = {
        row["session_id"]: row
        for row in migrated.conn.execute(
            "SELECT * FROM session_outcomes WHERE session_id IN (?, ?)", ("c", "b:c")
        ).fetchall()
    }

    assert set(records_by_session) == {"c", "b:c"}
    assert records_by_session["c"].outcome_id == sha256(b'["a:b","c"]').hexdigest()
    assert records_by_session["b:c"].outcome_id == sha256(b'["a","b:c"]').hexdigest()
    assert records_by_session["c"].outcome_id != records_by_session["b:c"].outcome_id

    for row in legacy_rows:
        record = records_by_session[row["session_id"]]
        assert record.outcome_revision == 1
        assert record.user_id == row["user_id"]
        assert record.session_id == row["session_id"]
        assert record.outcome == SessionOutcomeKind(row["outcome"])
        assert record.occurred_at == row["occurred_at"]
        assert record.source == row["source"]
        assert record.label == row["label"]
        assert record.value == row["value"]
        assert record.metadata == row["metadata"]
        assert record.created_at == row["created_at"]
        assert (
            rows_by_session[row["session_id"]]["governance_subject_ref"]
            == row["governance_subject_ref"]
        )
        assert record.outcome_contract_digest == outcome_contract_digest(
            source=row["source"],
            schema_version=1,
            allowed_values={"success", "failure", "unknown"},
            finalization_rule="first_write",
        )
        assert record.finalized_trajectory_digest == trajectory_digest(
            _canonical_session_snapshot(migrated.conn, row["session_id"])
        )


def test_migration_prefetches_trajectory_inputs_in_bounded_chunks(tmp_path) -> None:
    db_path = str(tmp_path / "legacy-session-outcome-query-scaling.db")
    storage = SQLiteStorage(org_id="legacy-query-scaling", db_path=db_path)
    row_count = 501
    storage.conn.executemany(
        """INSERT INTO requests (
               request_id, user_id, created_at, source, session_id
           ) VALUES (?, ?, ?, ?, ?)""",
        [
            (
                f"request-{index}",
                f"user-{index}",
                str(index),
                "legacy-source",
                f"session-{index}",
            )
            for index in range(row_count)
        ],
    )
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(_LEGACY_SESSION_OUTCOMES_DDL)
    storage.conn.executemany(
        """INSERT INTO session_outcomes (
               user_id, session_id, outcome, occurred_at, source,
               governance_subject_ref, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                f"user-{index}",
                f"session-{index}",
                "success",
                index,
                "legacy-source",
                f"subject-{index}",
                index,
            )
            for index in range(row_count)
        ],
    )
    storage.conn.commit()

    statements: list[str] = []
    storage.conn.set_trace_callback(statements.append)
    try:
        storage._migrate_session_outcomes_schema()
    finally:
        storage.conn.set_trace_callback(None)

    trajectory_input_queries = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith("SELECT")
        and ("FROM requests" in statement or "FROM interactions" in statement)
    ]
    assert len(trajectory_input_queries) == 4
    assert (
        storage.conn.execute("SELECT COUNT(*) FROM session_outcomes").fetchone()[0]
        == row_count
    )


@pytest.mark.parametrize(
    ("with_subject_column", "stored_subject_ref"),
    [(False, None), (True, None), (True, "   ")],
    ids=["absent", "null", "whitespace"],
)
def test_migration_derives_missing_legacy_governance_subject_ref(
    tmp_path, with_subject_column: bool, stored_subject_ref: str | None
) -> None:
    db_path = str(tmp_path / f"legacy-subject-{with_subject_column}.db")
    storage = SQLiteStorage(org_id="legacy-subject", db_path=db_path)
    storage.add_request(
        Request(
            request_id="legacy-subject-request",
            user_id="legacy-user",
            session_id="legacy-session",
            source="legacy-source",
            created_at=100,
        )
    )
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(
        _legacy_session_outcomes_ddl(with_subject_column=with_subject_column)
    )
    values = (
        "legacy-user",
        "legacy-session",
        "success",
        101,
        "legacy-source",
        102,
    )
    if with_subject_column:
        storage.conn.execute(
            """INSERT INTO session_outcomes (
                   user_id, session_id, outcome, occurred_at, source, created_at,
                   governance_subject_ref
               ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (*values, stored_subject_ref),
        )
    else:
        storage.conn.execute(
            """INSERT INTO session_outcomes (
                   user_id, session_id, outcome, occurred_at, source, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            values,
        )
    storage.conn.commit()
    storage.conn.close()

    migrated = SQLiteStorage(org_id="legacy-subject", db_path=db_path)

    row = migrated.conn.execute(
        "SELECT governance_subject_ref FROM session_outcomes WHERE session_id = ?",
        ("legacy-session",),
    ).fetchone()
    assert row is not None
    assert row["governance_subject_ref"] == migrated._subject_ref_for_user_id(
        "legacy-user"
    )


@pytest.mark.parametrize("with_subject_column", [False, True])
def test_identity_complete_migration_backfills_missing_governance_subject_ref(
    tmp_path, with_subject_column: bool
) -> None:
    db_path = str(tmp_path / f"complete-subject-{with_subject_column}.db")
    storage = SQLiteStorage(org_id="complete-subject", db_path=db_path)
    storage.conn.execute("DROP TABLE session_outcomes")
    storage.conn.executescript(
        _identity_complete_session_outcomes_ddl(with_subject_column=with_subject_column)
    )
    values = (
        "stable-outcome-id",
        1,
        "complete-user",
        "complete-session",
        "unknown",
        101,
        "complete-source",
        "a" * 64,
        "b" * 64,
        102,
    )
    if with_subject_column:
        storage.conn.execute(
            """INSERT INTO session_outcomes (
                   outcome_id, outcome_revision, user_id, session_id, outcome,
                   occurred_at, source, outcome_contract_digest,
                   finalized_trajectory_digest, created_at,
                   governance_subject_ref
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (*values, None),
        )
    else:
        storage.conn.execute(
            """INSERT INTO session_outcomes (
                   outcome_id, outcome_revision, user_id, session_id, outcome,
                   occurred_at, source, outcome_contract_digest,
                   finalized_trajectory_digest, created_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            values,
        )
    storage.conn.commit()
    storage.conn.close()

    migrated = SQLiteStorage(org_id="complete-subject", db_path=db_path)

    row = migrated.conn.execute(
        """SELECT outcome_id, governance_subject_ref
           FROM session_outcomes WHERE session_id = ?""",
        ("complete-session",),
    ).fetchone()
    assert row is not None
    assert row["outcome_id"] == "stable-outcome-id"
    assert row["governance_subject_ref"] == migrated._subject_ref_for_user_id(
        "complete-user"
    )
