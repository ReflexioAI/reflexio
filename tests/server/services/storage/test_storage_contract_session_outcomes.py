"""Session outcome storage contract."""

from typing import cast

import pytest

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Request,
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import (
    _canonical_session_snapshot,
)
from reflexio.server.services.storage.storage_base import BaseStorage


def test_first_write_preserves_outcome_fields(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="r1",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="s1",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
        label="booked",
        value=12.0,
        metadata={"crm": "test"},
    )
    context = storage.get_session_outcome_context("s1")
    first = storage.record_session_outcome(
        request, created_at=102, expected_context=context
    )
    duplicate = storage.record_session_outcome(
        request.model_copy(update={"outcome": "failure"}),
        created_at=103,
        expected_context=context,
    )

    assert first.recorded is True
    assert duplicate.recorded is False
    assert duplicate.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    assert duplicate.source == "published"
    records = storage.get_session_outcomes(GetSessionOutcomesRequest(label="booked"))
    assert len(records) == 1
    assert records[0].outcome_id
    assert records[0].outcome_revision == 1
    outcome_contract_digest = records[0].outcome_contract_digest
    finalized_trajectory_digest = records[0].finalized_trajectory_digest
    assert outcome_contract_digest is not None
    assert finalized_trajectory_digest is not None
    assert len(outcome_contract_digest) == 64
    assert len(finalized_trajectory_digest) == 64
    assert records[0].outcome == SessionOutcomeKind.SUCCESS
    assert records[0].value == 12.0
    assert records[0].metadata == {"crm": "test"}


def test_generic_retention_cannot_delete_finalized_session_outcomes(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="retention-r1",
            user_id="u1",
            session_id="retention-session",
            source="published",
            created_at=100,
        )
    )
    storage.record_session_outcome(
        SetSessionOutcomeRequest(
            session_id="retention-session",
            outcome=SessionOutcomeKind.SUCCESS,
            occurred_at=101,
        ),
        created_at=102,
        expected_context=storage.get_session_outcome_context("retention-session"),
    )

    with pytest.raises(ValueError, match="Unknown retention target: session_outcomes"):
        storage.delete_oldest_retention_target_rows("session_outcomes", 1)  # type: ignore[attr-defined]

    [record] = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["retention-session"])
    )
    assert record.outcome == SessionOutcomeKind.SUCCESS


def test_exact_finalization_retry_is_idempotent(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="retry-r1",
            user_id="u1",
            session_id="exact-retry",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="exact-retry",
        outcome=SessionOutcomeKind.UNKNOWN,
        occurred_at=101,
        metadata={"reason": "not enough information"},
    )
    context = storage.get_session_outcome_context("exact-retry")

    first = storage.record_session_outcome(
        request, created_at=102, expected_context=context
    )
    retry = storage.record_session_outcome(
        request, created_at=103, expected_context=context
    )

    assert first.recorded is True
    assert retry.recorded is False
    assert retry.reason is None
    assert retry.outcome_id == first.outcome_id
    assert retry.outcome_revision == first.outcome_revision == 1
    assert retry.outcome_contract_digest == first.outcome_contract_digest
    assert retry.finalized_trajectory_digest == first.finalized_trajectory_digest


def test_legacy_all_null_identity_exact_retry_uses_available_context(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="legacy-r1",
            user_id="legacy-user",
            session_id="legacy-retry",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="legacy-retry",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
        metadata={"legacy": True},
    )
    first = storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("legacy-retry"),
    )
    assert first.recorded is True
    sqlite_storage = cast(SQLiteStorage, storage)
    sqlite_storage.conn.execute(
        "CREATE TABLE legacy_session_outcomes AS SELECT * FROM session_outcomes"
    )
    sqlite_storage.conn.execute("DROP TABLE session_outcomes")
    sqlite_storage.conn.execute(
        "ALTER TABLE legacy_session_outcomes RENAME TO session_outcomes"
    )
    sqlite_storage.conn.execute(
        """UPDATE session_outcomes
           SET outcome_id = NULL, outcome_revision = NULL,
               outcome_contract_digest = NULL,
               finalized_trajectory_digest = NULL
           WHERE session_id = ?""",
        ("legacy-retry",),
    )
    sqlite_storage.conn.commit()

    retry = storage.record_session_outcome(
        request,
        created_at=103,
        expected_context=storage.get_session_outcome_context("legacy-retry"),
    )

    assert retry.recorded is False
    assert retry.reason is None
    assert retry.outcome_id is None
    assert retry.outcome_revision is None
    assert retry.outcome_contract_digest is None
    assert retry.finalized_trajectory_digest is None


def test_legacy_all_null_identity_changed_payload_still_conflicts(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="legacy-conflict-r1",
            user_id="legacy-user",
            session_id="legacy-conflict",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="legacy-conflict",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("legacy-conflict"),
    )
    sqlite_storage = cast(SQLiteStorage, storage)
    sqlite_storage.conn.execute(
        "CREATE TABLE legacy_session_outcomes AS SELECT * FROM session_outcomes"
    )
    sqlite_storage.conn.execute("DROP TABLE session_outcomes")
    sqlite_storage.conn.execute(
        "ALTER TABLE legacy_session_outcomes RENAME TO session_outcomes"
    )
    sqlite_storage.conn.execute(
        """UPDATE session_outcomes
           SET outcome_id = NULL, outcome_revision = NULL,
               outcome_contract_digest = NULL,
               finalized_trajectory_digest = NULL
           WHERE session_id = ?""",
        ("legacy-conflict",),
    )
    sqlite_storage.conn.commit()

    retry = storage.record_session_outcome(
        request.model_copy(update={"outcome": SessionOutcomeKind.FAILURE}),
        created_at=103,
        expected_context=storage.get_session_outcome_context("legacy-conflict"),
    )

    assert retry.recorded is False
    assert retry.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION


def test_legacy_all_null_identity_changed_governance_context_conflicts(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="legacy-governance-r1",
            user_id="legacy-user",
            session_id="legacy-governance-conflict",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="legacy-governance-conflict",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context(
            "legacy-governance-conflict"
        ),
    )
    sqlite_storage = cast(SQLiteStorage, storage)
    sqlite_storage.conn.execute(
        "CREATE TABLE legacy_session_outcomes AS SELECT * FROM session_outcomes"
    )
    sqlite_storage.conn.execute("DROP TABLE session_outcomes")
    sqlite_storage.conn.execute(
        "ALTER TABLE legacy_session_outcomes RENAME TO session_outcomes"
    )
    sqlite_storage.conn.execute(
        """UPDATE session_outcomes
           SET outcome_id = NULL, outcome_revision = NULL,
               outcome_contract_digest = NULL,
               finalized_trajectory_digest = NULL
           WHERE session_id = ?""",
        ("legacy-governance-conflict",),
    )
    sqlite_storage.conn.execute(
        "UPDATE requests SET governance_subject_ref = ? WHERE session_id = ?",
        (
            sqlite_storage._subject_ref_for_user_id("different-user"),
            "legacy-governance-conflict",
        ),
    )
    sqlite_storage.conn.commit()

    retry = storage.record_session_outcome(
        request,
        created_at=103,
        expected_context=storage.get_session_outcome_context(
            "legacy-governance-conflict"
        ),
    )

    assert retry.recorded is False
    assert retry.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION


def test_sqlite_canonical_snapshot_loads_interactions_in_one_query(
    storage: BaseStorage,
) -> None:
    sqlite_storage = cast(SQLiteStorage, storage)
    for index in range(3):
        storage.add_request(
            Request(
                request_id=f"snapshot-r{index}",
                user_id="snapshot-user",
                session_id="snapshot-session",
                source="published",
                created_at=100 + index,
            )
        )
    statements: list[str] = []
    sqlite_storage.conn.set_trace_callback(statements.append)
    try:
        snapshot = _canonical_session_snapshot(sqlite_storage.conn, "snapshot-session")
    finally:
        sqlite_storage.conn.set_trace_callback(None)

    interaction_queries = [
        statement for statement in statements if "FROM interactions" in statement
    ]
    assert [item["request"]["request_id"] for item in snapshot["requests"]] == [
        "snapshot-r0",
        "snapshot-r1",
        "snapshot-r2",
    ]
    assert len(interaction_queries) == 1


def test_changed_contract_identity_is_conflicting_finalization(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="contract-r1",
            user_id="u1",
            session_id="contract-conflict",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="contract-conflict",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    first = storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("contract-conflict"),
    )
    sqlite_storage = cast(SQLiteStorage, storage)
    sqlite_storage.conn.execute(
        """UPDATE session_outcomes SET outcome_contract_digest = ?
           WHERE session_id = ?""",
        ("0" * 64, "contract-conflict"),
    )
    sqlite_storage.conn.commit()

    changed_contract = storage.record_session_outcome(
        request,
        created_at=103,
        expected_context=storage.get_session_outcome_context("contract-conflict"),
    )

    assert first.recorded is True
    assert changed_contract.recorded is False
    assert (
        changed_contract.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    )


def test_changed_session_trajectory_is_conflicting_finalization(
    storage: BaseStorage,
) -> None:
    storage.add_request(
        Request(
            request_id="trajectory-r1",
            user_id="u1",
            session_id="trajectory-conflict",
            source="published",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="trajectory-conflict",
        outcome=SessionOutcomeKind.FAILURE,
        occurred_at=101,
    )
    first = storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("trajectory-conflict"),
    )
    storage.add_request(
        Request(
            request_id="trajectory-r2",
            user_id="u1",
            session_id="trajectory-conflict",
            source="published",
            created_at=103,
        )
    )

    changed_trajectory = storage.record_session_outcome(
        request,
        created_at=104,
        expected_context=storage.get_session_outcome_context("trajectory-conflict"),
    )

    assert first.recorded is True
    assert changed_trajectory.recorded is False
    assert (
        changed_trajectory.reason
        == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    )


def test_unknown_session_is_rejected(storage: BaseStorage) -> None:
    context = storage.get_session_outcome_context("missing")
    result = storage.record_session_outcome(
        SetSessionOutcomeRequest(
            session_id="missing",
            outcome=SessionOutcomeKind.FAILURE,
            occurred_at=1,
        ),
        created_at=2,
        expected_context=context,
    )
    assert result.recorded is False
    assert result.reason == "unknown_session"


def test_stale_first_request_context_cannot_commit(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="later",
            user_id="u1",
            session_id="race",
            source="later-source",
            created_at=200,
        )
    )
    stale = storage.get_session_outcome_context("race")
    storage.add_request(
        Request(
            request_id="earlier",
            user_id="u1",
            session_id="race",
            source="canonical-source",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="race", outcome=SessionOutcomeKind.FAILURE, occurred_at=201
    )

    stale_result = storage.record_session_outcome(
        request, created_at=202, expected_context=stale
    )
    fresh = storage.get_session_outcome_context("race")
    fresh_result = storage.record_session_outcome(
        request, created_at=202, expected_context=fresh
    )

    assert stale_result.context_changed is True
    assert fresh_result.recorded is True
    assert fresh_result.source == "canonical-source"


def test_empty_request_source_is_preserved(storage: BaseStorage) -> None:
    storage.add_request(
        Request(
            request_id="empty-source-r1",
            user_id="u1",
            session_id="empty-source",
            source="",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="empty-source",
        outcome=SessionOutcomeKind.FAILURE,
        occurred_at=101,
    )
    result = storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("empty-source"),
    )

    assert result.source == ""
    [record] = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["empty-source"])
    )
    assert record.source == ""


def test_clear_outcomes_survives_governance_secret_rotation(
    storage: BaseStorage, monkeypatch
) -> None:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "old-secret")
    storage.add_request(
        Request(
            request_id="rotated-r1",
            user_id="rotated-user",
            session_id="rotated-session",
            source="test",
            created_at=100,
        )
    )
    request = SetSessionOutcomeRequest(
        session_id="rotated-session",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    storage.record_session_outcome(
        request,
        created_at=102,
        expected_context=storage.get_session_outcome_context("rotated-session"),
    )

    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "new-secret")
    counts = storage.clear_session_outcomes_for_user("rotated-user")

    assert counts == {"session_outcomes": 1}
    assert storage.get_session_outcomes(GetSessionOutcomesRequest()) == []
