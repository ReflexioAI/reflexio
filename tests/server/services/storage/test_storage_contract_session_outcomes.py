"""Session outcome storage contract."""

from threading import RLock
from typing import Any, cast
from unittest.mock import MagicMock

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
from reflexio.server.services.storage.sqlite_storage._base import (
    _TRAJECTORY_FETCH_SIZE,
)
from reflexio.server.services.storage.sqlite_storage._session_outcomes import (
    SessionOutcomeStoreMixin,
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


def test_sqlite_legacy_null_source_finalization_retry_is_idempotent(
    storage: BaseStorage,
) -> None:
    sqlite_storage = cast(SQLiteStorage, storage)
    storage.add_request(
        Request(
            request_id="legacy-null-source-r1",
            user_id="u1",
            session_id="legacy-null-source",
            source="",
            created_at=100,
        )
    )
    schema_sql = sqlite_storage.conn.execute(
        "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = 'requests'"
    ).fetchone()["sql"]
    nullable_schema_sql = schema_sql.replace(
        "source TEXT NOT NULL DEFAULT ''", "source TEXT DEFAULT ''", 1
    )
    sqlite_storage.conn.execute("PRAGMA writable_schema = ON")
    sqlite_storage.conn.execute(
        "UPDATE sqlite_schema SET sql = ? WHERE type = 'table' AND name = 'requests'",
        (nullable_schema_sql,),
    )
    schema_version = sqlite_storage.conn.execute("PRAGMA schema_version").fetchone()[0]
    sqlite_storage.conn.execute(f"PRAGMA schema_version = {schema_version + 1}")
    sqlite_storage.conn.execute("PRAGMA writable_schema = OFF")
    sqlite_storage.conn.execute(
        "UPDATE requests SET source = NULL WHERE request_id = ?",
        ("legacy-null-source-r1",),
    )
    sqlite_storage.conn.commit()

    request = SetSessionOutcomeRequest(
        session_id="legacy-null-source",
        outcome=SessionOutcomeKind.SUCCESS,
        occurred_at=101,
    )
    context = storage.get_session_outcome_context("legacy-null-source")
    first = storage.record_session_outcome(
        request, created_at=102, expected_context=context
    )
    retry = storage.record_session_outcome(
        request,
        created_at=103,
        expected_context=storage.get_session_outcome_context("legacy-null-source"),
    )

    assert context.source == ""
    assert first.recorded is True
    assert first.source == ""
    assert retry.recorded is False
    assert retry.reason is None
    assert retry.source == ""


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
    initial_retry_connection = _NoFetchAllConnection(raw_connection)
    cast(Any, sqlite_storage).conn = initial_retry_connection
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
    conflict_connection = _NoFetchAllConnection(raw_connection)
    cast(Any, sqlite_storage).conn = conflict_connection
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
    assert initial_retry_connection.fetch_sizes
    assert conflict_connection.fetch_sizes
    assert all(
        size == _TRAJECTORY_FETCH_SIZE for size in initial_retry_connection.fetch_sizes
    )
    assert all(
        size == _TRAJECTORY_FETCH_SIZE for size in conflict_connection.fetch_sizes
    )


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


def test_sqlite_context_normalizes_nullable_request_source() -> None:
    existing_cursor = MagicMock()
    existing_cursor.fetchone.return_value = None
    first_cursor = MagicMock()
    first_cursor.fetchone.return_value = {
        "user_id": "u1",
        "source": None,
        "created_at": "1970-01-01T00:01:40+00:00",
    }
    counts_cursor = MagicMock()
    counts_cursor.fetchone.return_value = {"user_count": 1, "source_count": 1}
    connection = MagicMock()

    def execute(statement: str, _parameters: tuple[str]) -> MagicMock:
        if "FROM session_outcomes" in statement:
            return existing_cursor
        if "ORDER BY created_at ASC" in statement:
            return first_cursor
        if "COUNT(DISTINCT user_id)" in statement:
            return counts_cursor
        raise AssertionError(
            f"unexpected SQL in nullable-source context test: {statement}"
        )

    connection.execute.side_effect = execute
    reader = SessionOutcomeStoreMixin()
    cast(Any, reader).conn = connection
    cast(Any, reader)._lock = RLock()

    context = reader.get_session_outcome_context("nullable-source")

    assert context.source == ""
    assert context.first_request_at == 100


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


# ---------------------------------------------------------------------------
# Displacement: a customer's own report replaces an outcome the tuner inferred.
#
# Exactly one of the four write situations changes. The three that do NOT are
# asserted here too, because "the customer always wins" would be the wrong fix
# and these are what distinguish it from the right one.
# ---------------------------------------------------------------------------


def _seed_request(storage: BaseStorage, *, session_id: str = "s1") -> None:
    storage.add_request(
        Request(
            request_id=f"r-{session_id}",
            user_id="u1",
            session_id=session_id,
            source="published",
            created_at=100,
        )
    )


def _write(
    storage: BaseStorage,
    *,
    outcome: SessionOutcomeKind,
    is_inferred: bool,
    created_at: int,
    label: str | None = None,
    session_id: str = "s1",
):
    request = SetSessionOutcomeRequest(
        session_id=session_id,
        outcome=outcome,
        occurred_at=101,
        label=label,
    )
    context = storage.get_session_outcome_context(session_id)
    return storage.record_session_outcome(
        request,
        created_at=created_at,
        expected_context=context,
        is_inferred=is_inferred,
    )


def test_a_customer_report_displaces_an_inferred_outcome(
    storage: BaseStorage,
) -> None:
    """THE POINT OF THE WHOLE CHANGE.

    Before this, the tuner's guess took the session's only slot and the
    customer's real report was refused forever -- which is the only reason the
    bridge withheld verdicts for seven days.
    """
    _seed_request(storage)
    inferred = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    assert inferred.recorded is True
    assert inferred.outcome_revision == 1

    reported = _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=103,
        label="customer",
    )
    assert reported.recorded is True
    assert reported.reason is None
    # The session's SECOND outcome -- the first time this column has moved.
    assert reported.outcome_revision == 2

    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert len(records) == 1, "displacement must not leave two rows"
    assert records[0].outcome == SessionOutcomeKind.SUCCESS
    assert records[0].label == "customer"


def test_the_displaced_outcome_is_kept_for_analysis(storage: BaseStorage) -> None:
    """The judge-versus-customer comparison IS the analysis, so the displaced
    row is archived whole rather than summarised."""
    _seed_request(storage)
    _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=103,
    )

    sqlite_storage = cast(Any, storage)
    row = sqlite_storage.conn.execute(
        "SELECT is_inferred, superseded_outcome FROM session_outcomes "
        "WHERE session_id = ?",
        ("s1",),
    ).fetchone()
    assert not row["is_inferred"], "the surviving row is the customer's, not a guess"
    archived = __import__("json").loads(row["superseded_outcome"])
    assert archived["outcome"] == "failure"
    assert archived["label"] == "inferred_from_agent_success_evaluation"
    assert archived["outcome_revision"] == 1
    assert archived["displaced_at"] == 103


def test_a_customer_outcome_is_still_immutable(storage: BaseStorage) -> None:
    """THE GUARANTEE THAT MUST NOT MOVE.

    Only the machine's guess became displaceable. A customer's own report is
    as final as it ever was -- if this passes only because everything is
    mutable now, the change is a regression rather than a feature.
    """
    _seed_request(storage)
    _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=102,
    )
    second = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=False,
        created_at=103,
    )

    assert second.recorded is False
    assert second.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert records[0].outcome == SessionOutcomeKind.SUCCESS


def test_the_tuner_never_displaces_a_customer_outcome(storage: BaseStorage) -> None:
    """The bridge must not overwrite a real report with a guess, in either
    direction -- it subtracts existing outcomes before writing, and this is the
    storage-level backstop for when it does not."""
    _seed_request(storage)
    _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=102,
    )
    tuner = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=103,
    )

    assert tuner.recorded is False
    assert tuner.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION


def test_the_tuner_repeating_its_own_write_is_an_idempotent_retry(
    storage: BaseStorage,
) -> None:
    """NOT a displacement. An inferred row plus an inferred writer is the
    bridge retrying itself; treating that as displacement would bump the
    revision and archive the row as a copy of itself on every pass."""
    _seed_request(storage)
    first = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    retry = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )

    assert first.recorded is True
    assert retry.recorded is False
    assert retry.reason is None, "an exact repeat is accepted, not conflicting"
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert records[0].outcome_revision == 1


def test_displacement_refuses_to_cross_a_governance_subject(
    storage: BaseStorage,
) -> None:
    """A TENANCY REFUSAL, not bookkeeping.

    The outcome row is found by ``session_id``, but it is ERASED by ``user_id``
    and gated by ``governance_subject_ref``. A session's earliest request can
    change owners after the inferred write -- here the original first request
    is deleted while a second user's request remains. Displacing would rewrite
    the row to the new owner while ``superseded_outcome`` still held the
    ORIGINAL subject's outcome, so erasing the original user would no longer
    reach it: a completed erasure that quietly leaves the subject's outcome
    behind under a stranger's key.
    """
    storage.add_request(
        Request(
            request_id="r-first",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=100,
        )
    )
    storage.add_request(
        Request(
            request_id="r-second",
            user_id="u2",
            session_id="s1",
            source="published",
            created_at=101,
        )
    )
    inferred = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    assert inferred.recorded is True
    assert inferred.user_id == "u1"

    storage.delete_request("r-first")

    reported = _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=103,
    )

    assert reported.recorded is False
    assert reported.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    assert reported.user_id == "u1", "the refusal reports the STORED owner"
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert len(records) == 1
    assert records[0].user_id == "u1", "the row must not be re-owned"
    assert records[0].outcome == SessionOutcomeKind.FAILURE
    assert records[0].outcome_revision == 1
    # The decisive consequence: erasing the original subject still reaches it.
    assert storage.clear_session_outcomes_for_user("u1") == {"session_outcomes": 1}


def test_displacement_refuses_a_new_owner_that_shares_the_stored_subject_ref(
    storage: BaseStorage,
) -> None:
    """The USER ID is checked in its own right, not merely via the subject ref.

    Today a subject reference is derived from the user id, so the two checks
    usually agree and either alone would refuse. They are not the same check.
    ``user_id`` is what ``clear_session_outcomes_for_user`` erases by; the
    subject ref is what the write barrier gates on. If subjects ever became
    coarser than one-per-user -- a per-organisation subject, say -- two users
    would share a ref and the ref comparison would stop refusing anything.
    This pins the column erasure actually keys on, with the ref forced equal so
    only the user-id comparison can answer.
    """
    storage.add_request(
        Request(
            request_id="r-first",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=100,
        )
    )
    storage.add_request(
        Request(
            request_id="r-second",
            user_id="u2",
            session_id="s1",
            source="published",
            created_at=101,
        )
    )
    inferred = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    assert inferred.recorded is True
    assert inferred.user_id == "u1"

    connection = cast(Any, storage).conn
    stored_ref = connection.execute(
        "SELECT governance_subject_ref FROM session_outcomes WHERE session_id = ?",
        ("s1",),
    ).fetchone()["governance_subject_ref"]
    connection.execute(
        "UPDATE requests SET governance_subject_ref = ? WHERE request_id = ?",
        (stored_ref, "r-second"),
    )
    connection.commit()
    storage.delete_request("r-first")

    reported = _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=103,
    )

    assert reported.recorded is False
    assert reported.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert records[0].user_id == "u1", "the row must not be re-owned"
    assert records[0].outcome_revision == 1


def test_displacement_refuses_a_subject_ref_the_row_was_never_filed_under(
    storage: BaseStorage, monkeypatch
) -> None:
    """The SUBJECT REF is checked in its own right, not just the user id.

    Requests store the governance ref they were written with, so rotating the
    governance secret leaves two requests from the SAME user carrying
    DIFFERENT refs. Displacing across that would re-key the row -- archive
    included -- to a subject reference whose write barrier was never checked
    against the archived content.

    Refusing is the conservative answer and it is chosen deliberately: the cost
    is one rejected report in a rare window, where accepting silently re-files
    a data subject's outcome. The settled-row retry check above already treats
    a differing stored ref as a mismatch, so this is the same rule, not a new
    one.
    """
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "old-secret")
    storage.add_request(
        Request(
            request_id="r-old-epoch",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=100,
        )
    )
    inferred = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=True,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    assert inferred.recorded is True

    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "new-secret")
    storage.add_request(
        Request(
            request_id="r-new-epoch",
            user_id="u1",
            session_id="s1",
            source="published",
            created_at=101,
        )
    )
    storage.delete_request("r-old-epoch")

    reported = _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=103,
    )

    assert reported.recorded is False
    assert reported.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert records[0].outcome == SessionOutcomeKind.FAILURE
    assert records[0].outcome_revision == 1


def test_the_bridge_label_alone_does_not_make_an_outcome_displaceable(
    storage: BaseStorage,
) -> None:
    """Displaceability comes from the INTERNAL flag, never from the label.

    ``label`` is a field on ``SetSessionOutcomeRequest`` -- the public request
    body -- so a customer can send the bridge's exact provenance string. If
    that string were ever treated as provenance (by a read, or by a migration
    backfilling historical rows), a caller could mark their own outcome
    displaceable, which is the precise thing keeping ``is_inferred`` off the
    request model exists to prevent.
    """
    _seed_request(storage)
    _write(
        storage,
        outcome=SessionOutcomeKind.SUCCESS,
        is_inferred=False,
        created_at=102,
        label="inferred_from_agent_success_evaluation",
    )
    second = _write(
        storage,
        outcome=SessionOutcomeKind.FAILURE,
        is_inferred=False,
        created_at=103,
    )

    assert second.recorded is False
    assert second.reason == SessionOutcomeFailureReason.CONFLICTING_FINALIZATION
    records = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=["s1"])
    )
    assert records[0].outcome == SessionOutcomeKind.SUCCESS
    assert records[0].outcome_revision == 1
