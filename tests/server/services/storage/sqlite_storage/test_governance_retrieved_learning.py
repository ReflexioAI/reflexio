"""Governance erasure coverage for retrieved-learning evaluation data.

Erasing a user must delete their ``retrieved_learning_evaluation`` rows and
scrub all three evaluation ``_operation_state`` namespaces (retrieved-eval
state, the ``agent_success_group_eval`` marker, and ``grade_on_demand`` cache
rows) — while preserving every other user's rows and state.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    Request,
    RetrievedLearning,
    RetrievedLearningEvaluationResult,
    UserProfile,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base.evaluation_state_keys import (
    build_agent_success_marker_key,
    build_grade_on_demand_cache_key,
)
from reflexio.server.services.storage.storage_base.retrieved_learning_state import (
    build_retrieved_learning_state_key,
    session_fingerprint,
)

ORG = "org1"
SUBJECT_REF = "subref_v1_" + "a" * 32
REQUEST_REF = "reqref_v1_" + "b" * 32


@pytest.fixture
def storage(tmp_path, monkeypatch) -> Generator[SQLiteStorage]:
    monkeypatch.setenv("REFLEXIO_GOVERNANCE_REF_SECRET", "test-governance-secret")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        yield SQLiteStorage(org_id=ORG, db_path=str(tmp_path / "g.db"))


def _seed_user(storage: SQLiteStorage, user_id: str, session_id: str) -> None:
    storage.add_request(
        Request(request_id=f"r-{user_id}", user_id=user_id, session_id=session_id)
    )
    storage.add_user_profile(
        user_id,
        [
            UserProfile(
                profile_id=f"prof-{user_id}",
                user_id=user_id,
                content="c",
                last_modified_timestamp=1,
                generated_from_request_id=f"r-{user_id}",
            )
        ],
    )
    storage.add_user_interactions_bulk(
        user_id,
        [
            Interaction(
                user_id=user_id,
                request_id=f"r-{user_id}",
                content="hi",
                role="Assistant",
                retrieved_learnings=[
                    RetrievedLearning(kind="profile", learning_id=f"prof-{user_id}")
                ],
            )
        ],
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(user_id, session_id)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(user_id, session_id)
    commit = storage.replace_retrieved_learning_evaluation_results(
        user_id,
        session_id,
        generation,
        fingerprint,
        "complete",
        {},
        [
            RetrievedLearningEvaluationResult(
                user_id=user_id,
                session_id=session_id,
                kind="profile",
                learning_id=f"prof-{user_id}",
                is_relevant=True,
                relevance_reason="r",
                impact="positive",
                impact_reason="i",
                created_at=1,
            )
        ],
    )
    assert commit.disposition == "applied" and commit.committed_count == 1
    # Pre-existing-gap namespaces: agent-success marker + grade cache.
    storage.upsert_operation_state(
        build_agent_success_marker_key(ORG, user_id, session_id),
        {"evaluated": True, "evaluated_at": 1},
    )
    storage.upsert_operation_state(
        build_grade_on_demand_cache_key(ORG, session_id, "v1", "agent_success"),
        {"last_graded_at": 1, "result_id": 1},
    )


def _erase(storage: SQLiteStorage, user_id: str, purge_id: str) -> dict[str, int]:
    storage.begin_purge_operation(
        purge_id=purge_id,
        idempotency_key=f"idem_{purge_id}",
        operation_type="user_erasure",
        scope_type="user",
        subject_ref=SUBJECT_REF,
        request_ref=REQUEST_REF,
    )
    storage.prepare_governance_erase_targets(purge_id, user_id)
    return storage.apply_governance_user_data_delete(purge_id, user_id)


def test_erase_scrubs_rle_rows_and_all_state_namespaces(storage) -> None:
    _seed_user(storage, "erase-me", "sess-a")
    _seed_user(storage, "keep-me", "sess-b")

    counts = _erase(storage, "erase-me", "purge_rle_scrub")

    assert counts["retrieved_learning_evaluation_results"] == 1
    # 3 namespaces for the one session: retrieved-eval state, agent-success
    # marker, grade-cache row.
    assert counts["evaluation_operation_states"] == 3

    assert storage.get_retrieved_learning_evaluation_results(user_id="erase-me") == []
    for key in (
        build_retrieved_learning_state_key("erase-me", "sess-a"),
        build_agent_success_marker_key(ORG, "erase-me", "sess-a"),
        build_grade_on_demand_cache_key(ORG, "sess-a", "v1", "agent_success"),
    ):
        assert storage.get_operation_state(key) is None, key

    # The other user's rows and state are untouched.
    kept = storage.get_retrieved_learning_evaluation_results(user_id="keep-me")
    assert len(kept) == 1
    for key in (
        build_retrieved_learning_state_key("keep-me", "sess-b"),
        build_agent_success_marker_key(ORG, "keep-me", "sess-b"),
        build_grade_on_demand_cache_key(ORG, "sess-b", "v1", "agent_success"),
    ):
        assert storage.get_operation_state(key) is not None, key


def test_erase_is_idempotent_for_state_counts(storage) -> None:
    _seed_user(storage, "erase-me", "sess-a")
    first = _erase(storage, "erase-me", "purge_rle_retry_one")
    assert first["evaluation_operation_states"] == 3
    second = _erase(storage, "erase-me", "purge_rle_retry_two")
    assert second["retrieved_learning_evaluation_results"] == 0
    assert second["evaluation_operation_states"] == 0
