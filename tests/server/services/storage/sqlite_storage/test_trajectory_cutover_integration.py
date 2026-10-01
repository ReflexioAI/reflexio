"""Trajectory cutover on SQLite: prefix selection, prefix digests, verdict stamps.

The invariant under test: a cutover selects the session's requests whose
``(created_at, request_id)`` is ``<=`` the cutover's current row -- in SQL, with
``request_id`` breaking a same-second tie -- and the outcome digest, the
outcome precondition and ``get_requests_by_session`` all agree on that prefix.
"""

import sqlite3
from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    AgentSuccessEvaluationResult,
    GetSessionOutcomesRequest,
    Interaction,
    Request,
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.sqlite_storage._base import (
    _canonical_session_trajectory_snapshot,
)
from reflexio.server.services.storage.storage_base._session_outcomes import (
    OutcomePrefixPrecondition,
)

T0 = 1_760_000_000
SESSION = "cutover-session"
USER = "cutover-user"

_next_interaction_id = iter(range(1, 10_000))


@pytest.fixture
def storage(tmp_path) -> Generator[SQLiteStorage]:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id="cutover", db_path=str(tmp_path / "cutover.db"))


def _add_request(
    storage: SQLiteStorage,
    request_id: str,
    created_at: int,
    *,
    session_id: str = SESSION,
    interactions_at: tuple[int, ...] = (),
) -> None:
    storage.add_request(
        Request(
            request_id=request_id,
            user_id=USER,
            session_id=session_id,
            source="published",
            created_at=created_at,
        )
    )
    _add_interactions(storage, request_id, interactions_at)


def _add_interactions(
    storage: SQLiteStorage, request_id: str, created_ats: tuple[int, ...]
) -> None:
    if not created_ats:
        return
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                interaction_id=next(_next_interaction_id),
                user_id=USER,
                request_id=request_id,
                created_at=created_at,
                content=f"{request_id}@{created_at}",
            )
            for created_at in created_ats
        ],
        embeddings_prepared=True,
    )


def _seed_tied_session(storage: SQLiteStorage) -> None:
    """r1 < (a, b tied on created_at, inserted b first) < r3."""
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "b", T0 + 5, interactions_at=(T0 + 6,))
    _add_request(storage, "a", T0 + 5, interactions_at=(T0 + 6,))
    _add_request(storage, "r3", T0 + 10, interactions_at=(T0 + 11,))


def _ids(storage: SQLiteStorage, through: str | None) -> list[str]:
    requests = (
        storage.get_requests_by_session(USER, SESSION)
        if through is None
        else storage.get_requests_by_session(USER, SESSION, through_request_id=through)
    )
    return [r.request_id for r in requests]


def _digest(storage: SQLiteStorage, through: str | None = None) -> str:
    return _canonical_session_trajectory_snapshot(storage.conn, SESSION, through).digest


def _inferred(outcome: SessionOutcomeKind = SessionOutcomeKind.SUCCESS):
    return SetSessionOutcomeRequest(
        session_id=SESSION,
        outcome=outcome,
        occurred_at=T0 + 20,
        label="inferred_from_agent_success_evaluation",
    )


def _record(storage: SQLiteStorage, **kwargs):
    return storage.record_session_outcome(
        _inferred(),
        created_at=T0 + 30,
        expected_context=storage.get_session_outcome_context(SESSION),
        is_inferred=True,
        **kwargs,
    )


def _stored(storage: SQLiteStorage):
    [record] = storage.get_session_outcomes(
        GetSessionOutcomesRequest(session_ids=[SESSION])
    )
    return record


def test_cutover_selects_prefix_in_sql_order_with_request_id_tiebreak(
    storage: SQLiteStorage,
) -> None:
    _seed_tied_session(storage)

    assert _ids(storage, None) == ["r1", "a", "b", "r3"]
    # "a" and "b" share created_at; request_id decides, regardless of insert order.
    assert _ids(storage, "a") == ["r1", "a"]
    assert _ids(storage, "b") == ["r1", "a", "b"]
    assert _ids(storage, "r1") == ["r1"]
    assert _ids(storage, "r3") == ["r1", "a", "b", "r3"]


def test_cutover_outside_the_session_matches_nothing(storage: SQLiteStorage) -> None:
    _seed_tied_session(storage)
    _add_request(storage, "elsewhere", T0 + 100, session_id="other-session")

    assert _ids(storage, "elsewhere") == []
    assert _ids(storage, "no-such-request") == []


def test_prefix_through_last_request_digests_like_whole_session(
    storage: SQLiteStorage,
) -> None:
    _seed_tied_session(storage)

    whole = _digest(storage)
    assert _digest(storage, "r3") == whole
    assert _digest(storage, "b") != whole
    # Through the tie's FIRST member must exclude its second.
    assert _digest(storage, "a") != _digest(storage, "b")


def test_inferred_outcome_digest_is_prefix_scoped(storage: SQLiteStorage) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))

    first = _record(storage, trajectory_through_request_id="r2")
    assert first.recorded is True
    assert first.finalized_trajectory_digest == _digest(storage, "r2")
    stored = _stored(storage)
    assert stored.trajectory_through_request_id == "r2"

    # A later turn does not move the prefix's digest, so the same write is an
    # exact retry rather than a refresh.
    _add_request(storage, "r3", T0 + 40, interactions_at=(T0 + 41,))
    assert _digest(storage, "r2") == first.finalized_trajectory_digest
    retry = _record(storage, trajectory_through_request_id="r2")
    assert retry.recorded is False
    assert retry.reason is None
    assert retry.prefix_moved is False
    assert _stored(storage).finalized_trajectory_digest == (
        first.finalized_trajectory_digest
    )


def test_exact_retry_with_extended_cutover_refreshes_in_place(
    storage: SQLiteStorage,
) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))
    _add_request(storage, "r3", T0 + 10, interactions_at=(T0 + 11,))
    first = _record(storage, trajectory_through_request_id="r2")

    extended = _record(storage, trajectory_through_request_id="r3")

    assert extended.recorded is True
    assert extended.outcome_id == first.outcome_id
    assert extended.outcome_revision == first.outcome_revision == 1
    assert extended.finalized_trajectory_digest == _digest(storage, "r3")
    stored = _stored(storage)
    assert stored.trajectory_through_request_id == "r3"
    assert stored.is_inferred is True


def test_legacy_whole_session_retry_still_idempotent(storage: SQLiteStorage) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    first = _record(storage)
    retry = _record(storage)

    assert first.recorded is True
    assert first.finalized_trajectory_digest == _digest(storage)
    assert retry.recorded is False and retry.reason is None
    assert _stored(storage).trajectory_through_request_id is None


def test_legacy_row_gains_cutover_even_when_digest_is_unchanged(
    storage: SQLiteStorage,
) -> None:
    """A cutover through the LAST request hashes like the whole session, so
    only the cutover comparison tells this write apart from an exact retry."""
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))
    legacy = _record(storage)

    result = _record(storage, trajectory_through_request_id="r2")

    assert result.finalized_trajectory_digest == legacy.finalized_trajectory_digest
    assert result.recorded is True
    assert _stored(storage).trajectory_through_request_id == "r2"


def test_precondition_that_holds_writes(storage: SQLiteStorage) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))
    _add_request(storage, "r3", T0 + 50, interactions_at=(T0 + 51,))

    result = _record(
        storage,
        trajectory_through_request_id="r2",
        prefix_precondition=OutcomePrefixPrecondition(
            verdict_settled_at=T0 + 20, interaction_count=2
        ),
    )

    # r3 is newer than the verdict but outside the prefix, so it is not a move.
    assert result.recorded is True
    assert result.prefix_moved is False
    # And the stored digest describes the prefix, not the grown session.
    assert result.finalized_trajectory_digest == _digest(storage, "r2")
    assert result.finalized_trajectory_digest != _digest(storage)
    assert _stored(storage).finalized_trajectory_digest == _digest(storage, "r2")


def test_late_prefix_interaction_is_prefix_moved(storage: SQLiteStorage) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))
    _add_interactions(storage, "r1", (T0 + 25,))

    result = _record(
        storage,
        trajectory_through_request_id="r2",
        # The count matches; only the clock gives the late interaction away.
        prefix_precondition=OutcomePrefixPrecondition(
            verdict_settled_at=T0 + 20, interaction_count=3
        ),
    )

    assert result.prefix_moved is True
    assert result.recorded is False
    assert storage.get_session_outcomes(GetSessionOutcomesRequest()) == []


def test_prefix_interaction_count_mismatch_is_prefix_moved(
    storage: SQLiteStorage,
) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))
    _add_request(storage, "r2", T0 + 5, interactions_at=(T0 + 6,))
    # Backdated: its clock predates the verdict, only the count can catch it.
    _add_interactions(storage, "r1", (T0 + 2,))

    result = _record(
        storage,
        trajectory_through_request_id="r2",
        prefix_precondition=OutcomePrefixPrecondition(
            verdict_settled_at=T0 + 20, interaction_count=2
        ),
    )

    assert result.prefix_moved is True
    assert storage.get_session_outcomes(GetSessionOutcomesRequest()) == []


def test_missing_cutover_is_prefix_moved(storage: SQLiteStorage) -> None:
    _add_request(storage, "r1", T0, interactions_at=(T0 + 1,))

    result = _record(storage, trajectory_through_request_id="gone")

    assert result.prefix_moved is True
    assert storage.get_session_outcomes(GetSessionOutcomesRequest()) == []


def test_verdict_cutover_stamp_round_trips(storage: SQLiteStorage) -> None:
    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                user_id=USER,
                session_id=SESSION,
                agent_version="v1",
                evaluation_name="default",
                is_success=True,
                created_at=T0,
                trajectory_through_request_id="r2",
                trajectory_interaction_count=4,
                embedding=[0.0] * 512,
            )
        ]
    )

    [result] = storage.get_agent_success_evaluation_results(user_id=USER)

    assert result.trajectory_through_request_id == "r2"
    assert result.trajectory_interaction_count == 4


def test_upgraded_database_gains_cutover_columns(tmp_path) -> None:
    db_path = str(tmp_path / "upgrade.db")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        SQLiteStorage(org_id="upgrade", db_path=db_path).conn.close()
    conn = sqlite3.connect(db_path)
    for table, column in (
        ("session_outcomes", "trajectory_through_request_id"),
        ("agent_success_evaluation_result", "trajectory_through_request_id"),
        ("agent_success_evaluation_result", "trajectory_interaction_count"),
    ):
        conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.commit()
    conn.close()

    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        upgraded = SQLiteStorage(org_id="upgrade", db_path=db_path)

    def columns(table: str) -> set[str]:
        return {
            row["name"] for row in upgraded.conn.execute(f"PRAGMA table_info({table})")
        }

    assert "trajectory_through_request_id" in columns("session_outcomes")
    assert {
        "trajectory_through_request_id",
        "trajectory_interaction_count",
    } <= columns("agent_success_evaluation_result")
