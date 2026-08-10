"""Session outcome storage contract."""

from typing import Any, cast

import pytest

from reflexio.models.api_schema.domain import (
    GetSessionOutcomesRequest,
    Interaction,
    Request,
    SessionOutcomeFailureReason,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_session_trajectory,
    trajectory_digest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
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


class _NoFetchAllCursor:
    def __init__(self, cursor: Any, fetch_sizes: list[int]) -> None:
        self._cursor = cursor
        self._fetch_sizes = fetch_sizes

    def fetchall(self) -> Any:
        raise AssertionError("session outcome finalization must not call fetchall")

    def fetchmany(self, size: int) -> Any:
        self._fetch_sizes.append(size)
        return self._cursor.fetchmany(size)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


class _NoFetchAllConnection:
    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self.fetch_sizes: list[int] = []

    def execute(self, *args: Any, **kwargs: Any) -> _NoFetchAllCursor:
        return _NoFetchAllCursor(
            self._connection.execute(*args, **kwargs), self.fetch_sizes
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def test_sqlite_large_finalization_streams_complete_digest_and_preserves_retry_contract(
    storage: BaseStorage,
) -> None:
    sqlite_storage = cast(SQLiteStorage, storage)
    session_id = "large-streamed-session"
    request_id = "large-streamed-request"
    storage.add_request(
        Request(
            request_id=request_id,
            user_id="stream-user",
            session_id=session_id,
            source="published",
            created_at=100,
        )
    )
    interactions = [
        Interaction(
            interaction_id=index + 1,
            user_id="stream-user",
            request_id=request_id,
            created_at=101 + index,
            content=f"complete-row-{index}",
            token_count=index,
        )
        for index in range(300)
    ]
    sqlite_storage.add_user_interactions_bulk(
        "stream-user", interactions, embeddings_prepared=True
    )
    request_rows = sqlite_storage.conn.execute(
        """SELECT request_id, user_id, created_at, source, agent_version, session_id,
                  evaluation_only, retrieval_experiment_id, retrieval_experiment_arm
           FROM requests WHERE session_id = ?
           ORDER BY created_at ASC, request_id ASC""",
        (session_id,),
    ).fetchall()
    interaction_rows = sqlite_storage.conn.execute(
        """SELECT interaction_id, user_id, request_id, created_at, content, role,
                  token_count, user_action, user_action_description,
                  interacted_image_url, image_encoding, shadow_content,
                  expert_content, tools_used, citations, retrieved_learnings
           FROM interactions WHERE request_id = ?
           ORDER BY created_at ASC, interaction_id ASC""",
        (request_id,),
    ).fetchall()
    expected_digest = trajectory_digest(
        canonical_session_trajectory(
            session_id,
            [dict(row) for row in request_rows],
            {request_id: [dict(row) for row in interaction_rows]},
        )
    )
    outcome = SetSessionOutcomeRequest(
        session_id=session_id,
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=500,
        metadata={"complete": True},
    )
    raw_connection = sqlite_storage.conn
    guarded_connection = _NoFetchAllConnection(raw_connection)
    cast(Any, sqlite_storage).conn = guarded_connection
    try:
        context = storage.get_session_outcome_context(session_id)
        first = storage.record_session_outcome(
            outcome, created_at=501, expected_context=context
        )
        retry = storage.record_session_outcome(
            outcome,
            created_at=502,
            expected_context=storage.get_session_outcome_context(session_id),
        )
    finally:
        cast(Any, sqlite_storage).conn = raw_connection

    sqlite_storage.add_user_interactions_bulk(
        "stream-user",
        [
            Interaction(
                interaction_id=301,
                user_id="stream-user",
                request_id=request_id,
                created_at=401,
                content="post-finalization-row",
            )
        ],
        embeddings_prepared=True,
    )
    guarded_connection = _NoFetchAllConnection(raw_connection)
    cast(Any, sqlite_storage).conn = guarded_connection
    try:
        conflict = storage.record_session_outcome(
            outcome,
            created_at=503,
            expected_context=storage.get_session_outcome_context(session_id),
        )
    finally:
        cast(Any, sqlite_storage).conn = raw_connection

    assert first.recorded is True
    assert first.finalized_trajectory_digest == expected_digest
    assert retry.recorded is False
    assert retry.reason is None
    assert retry.finalized_trajectory_digest == expected_digest
    assert conflict.recorded is False
    assert conflict.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    assert guarded_connection.fetch_sizes
    assert len(set(guarded_connection.fetch_sizes)) == 1


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
