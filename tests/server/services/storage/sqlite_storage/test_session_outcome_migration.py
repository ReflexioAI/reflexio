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
