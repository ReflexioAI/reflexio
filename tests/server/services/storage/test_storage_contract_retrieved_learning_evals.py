"""Contract tests for retrieved-learning evaluation storage.

Covers the two core invariants:

1. Round-trip parity — ``retrieved_learnings`` survives publish/storage/read
   identically on every locally-testable backend.
2. Exact evaluation set — replacement persists exactly the attached AND
   retrieval-eligible identities, atomically, fenced by generation +
   session fingerprint.
"""

from __future__ import annotations

import threading

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    Request,
    RetrievedLearning,
    RetrievedLearningEvaluationResult,
    RetrievedLearningSnapshot,
    UserPlaybook,
    UserProfile,
)
from reflexio.server.services.storage.storage_base.retrieved_learning_state import (
    build_retrieved_learning_state_key,
    session_fingerprint,
)

pytestmark = pytest.mark.integration

USER = "rle-user"
SESSION = "rle-session"


def _seed_session(storage, *, refs: list[RetrievedLearning] | None = None) -> None:
    storage.add_request(Request(request_id="r1", user_id=USER, session_id=SESSION))
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(user_id=USER, request_id="r1", content="hi", role="User"),
            Interaction(
                user_id=USER,
                request_id="r1",
                content="hello!",
                role="Assistant",
                retrieved_learnings=refs or [],
            ),
        ],
    )


def _seed_eligible_learnings(storage) -> int:
    """Seed one live profile + one live user playbook; return the playbook id."""
    storage.add_user_profile(
        USER,
        [
            UserProfile(
                profile_id="prof-1",
                user_id=USER,
                content="prefers concise answers",
                last_modified_timestamp=1,
                generated_from_request_id="r1",
            )
        ],
    )
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id=USER,
                playbook_name="checklist",
                request_id="r1",
                agent_version="v1",
                content="always produce a checklist",
                trigger="when deploying",
            )
        ]
    )
    saved = storage.get_user_playbooks(user_id=USER, limit=10)
    return saved[0].user_playbook_id


def _result(kind: str, learning_id: str) -> RetrievedLearningEvaluationResult:
    return RetrievedLearningEvaluationResult(
        user_id=USER,
        session_id=SESSION,
        agent_version="v1",
        kind=kind,  # type: ignore[arg-type]
        learning_id=learning_id,
        is_relevant=True,
        relevance_reason="applies",
        impact="positive",
        impact_reason="helped",
        created_at=1_700_000_000,
    )


def test_retrieved_learnings_round_trip(storage) -> None:
    refs = [
        RetrievedLearning(
            kind="profile",
            learning_id="prof-1",
            snapshot=RetrievedLearningSnapshot(
                title="Preference at injection",
                content="Keep answers concise.",
                trigger="",
                rationale="",
            ),
        ),
        RetrievedLearning(kind="user_playbook", learning_id="42"),
        RetrievedLearning(kind="agent_playbook", learning_id="7"),
    ]
    _seed_session(storage, refs=refs)
    back = storage.get_user_interaction(USER)
    assistant = next(i for i in back if i.role == "Assistant")
    assert [(r.kind, r.learning_id) for r in assistant.retrieved_learnings] == [
        ("profile", "prof-1"),
        ("user_playbook", "42"),
        ("agent_playbook", "7"),
    ]
    assert assistant.retrieved_learnings[0].snapshot == RetrievedLearningSnapshot(
        title="Preference at injection",
        content="Keep answers concise.",
        trigger="",
        rationale="",
    )
    user_turn = next(i for i in back if i.role == "User")
    assert user_turn.retrieved_learnings == []


def test_snapshot_covers_every_interaction_and_ref(storage) -> None:
    refs = [RetrievedLearning(kind="profile", learning_id="prof-1")]
    _seed_session(storage, refs=refs)
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    assert len(snapshot.interactions) == 2
    assert snapshot.raw_attachment_count == 1
    assert not snapshot.attachment_limit_exceeded
    all_refs = [ref for i in snapshot.interactions for ref in i.refs]
    assert all_refs == [("profile", "prof-1")]
    assert snapshot.agent_version == ""
    assert snapshot.earliest_request_created_at is not None


def test_content_only_edit_invalidates_fingerprint(storage) -> None:
    """An in-place content edit (same id + refs) must invalidate the cache.

    The fingerprint used to cover only interaction ids + refs, so replacing an
    interaction's transcript while keeping its id and attachments left stale
    verdicts cached against an unchanged digest.
    """
    refs = [RetrievedLearning(kind="profile", learning_id="prof-1")]
    _seed_session(storage, refs=refs)
    before = session_fingerprint(
        storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    )
    assistant = next(
        i for i in storage.get_user_interaction(USER) if i.role == "Assistant"
    )

    # INSERT OR REPLACE the same interaction with different content.
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                interaction_id=assistant.interaction_id,
                user_id=USER,
                request_id="r1",
                content="a completely different assistant answer",
                role="Assistant",
                retrieved_learnings=refs,
            )
        ],
    )
    after = session_fingerprint(
        storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    )
    assert after != before

    # The commit-side recompute agrees: a run fenced by the pre-edit
    # fingerprint is now stale and cannot overwrite the verdicts.
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        before,
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert commit.disposition == "stale"


def test_replace_persists_exact_eligible_set(storage) -> None:
    playbook_id = _seed_eligible_learnings(storage)
    _seed_session(
        storage,
        refs=[
            RetrievedLearning(kind="profile", learning_id="prof-1"),
            RetrievedLearning(kind="user_playbook", learning_id=str(playbook_id)),
            RetrievedLearning(kind="user_playbook", learning_id="999999"),
        ],
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)

    proposed = [
        _result("profile", "prof-1"),
        _result("user_playbook", str(playbook_id)),
        # Attached but nonexistent — must be dropped at commit time.
        _result("user_playbook", "999999"),
    ]
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER, SESSION, generation, fingerprint, "complete", {}, proposed
    )
    assert commit.disposition == "applied"
    assert commit.status == "complete"
    assert commit.committed_count == 2

    rows = storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    assert {(r.kind, r.learning_id) for r in rows} == {
        ("profile", "prof-1"),
        ("user_playbook", str(playbook_id)),
    }
    # Terminal state matches at this exact fingerprint.
    state = storage.get_matching_retrieved_learning_terminal_state(
        USER, SESSION, fingerprint
    )
    assert state is not None
    assert state["status"] == "complete"
    assert state["committed_count"] == 2


def test_replace_drops_eligible_but_unattached(storage) -> None:
    """Commit keeps only records attached to the live session.

    Guards the ``attached AND retrieval-eligible`` contract at the storage
    layer: an otherwise-eligible ref that was never attached to the session
    must be dropped rather than persisted.
    """
    _seed_eligible_learnings(storage)
    # A second eligible profile that is NOT attached to the session.
    storage.add_user_profile(
        USER,
        [
            UserProfile(
                profile_id="prof-2",
                user_id=USER,
                content="eligible but not attached",
                last_modified_timestamp=1,
                generated_from_request_id="r1",
            )
        ],
    )
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [
            _result("profile", "prof-1"),  # attached + eligible
            _result("profile", "prof-2"),  # eligible but never attached
        ],
    )
    assert commit.disposition == "applied"
    assert commit.committed_count == 1
    rows = storage.get_retrieved_learning_evaluation_results(session_id=SESSION)
    assert [(r.kind, r.learning_id) for r in rows] == [("profile", "prof-1")]


def test_replace_with_no_eligible_rows_clears_and_records_not_applicable(
    storage,
) -> None:
    _seed_eligible_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert commit.disposition == "applied" and commit.committed_count == 1

    # A later run whose candidates all became ineligible clears prior rows.
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [_result("profile", "no-longer-exists")],
    )
    assert commit.disposition == "applied"
    assert commit.status == "not_applicable"
    assert commit.committed_count == 0
    assert storage.get_retrieved_learning_evaluation_results(session_id=SESSION) == []


def test_none_verdict_fields_round_trip(storage) -> None:
    _seed_eligible_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    half = RetrievedLearningEvaluationResult(
        user_id=USER,
        session_id=SESSION,
        kind="profile",
        learning_id="prof-1",
        is_relevant=None,
        impact="negative",
        impact_reason="hurt",
        created_at=1,
    )
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER, SESSION, generation, fingerprint, "degraded", {}, [half]
    )
    assert commit.status == "degraded"
    row = storage.get_retrieved_learning_evaluation_results(session_id=SESSION)[0]
    assert row.is_relevant is None
    assert row.relevance_reason == ""
    assert row.impact == "negative"


def test_stale_fingerprint_and_superseded_generation(storage) -> None:
    _seed_eligible_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    gen1 = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    gen2 = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    assert gen2 == gen1 + 1

    # Older generation cannot commit after a newer one started.
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER, SESSION, gen1, fingerprint, "complete", {}, [_result("profile", "prof-1")]
    )
    assert commit.disposition == "superseded"

    # Wrong fingerprint is stale and changes nothing.
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        gen2,
        "not-the-fingerprint",
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert commit.disposition == "stale"
    assert storage.get_retrieved_learning_evaluation_results(session_id=SESSION) == []


def test_publish_and_delete_invalidate_fingerprint(storage) -> None:
    _seed_eligible_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert commit.disposition == "applied"
    assert (
        storage.get_matching_retrieved_learning_terminal_state(
            USER, SESSION, fingerprint
        )
        is not None
    )

    # Transcript-only publish (no retrieved_learnings) changes the fingerprint
    # and defeats the terminal shortcut.
    storage.add_user_interactions_bulk(
        USER,
        [Interaction(user_id=USER, request_id="r1", content="follow-up", role="User")],
    )
    # Even if a caller presents the pre-publish fingerprint, the storage-side
    # atomic check must recompute live state and reject it.
    assert (
        storage.get_matching_retrieved_learning_terminal_state(
            USER, SESSION, fingerprint
        )
        is None
    )
    new_fingerprint = session_fingerprint(
        storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    )
    assert new_fingerprint != fingerprint
    assert (
        storage.get_matching_retrieved_learning_terminal_state(
            USER, SESSION, new_fingerprint
        )
        is None
    )

    # A late replace with the old fingerprint is stale.
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    commit = storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert commit.disposition == "stale"


def test_finish_run_is_generation_guarded(storage) -> None:
    _seed_session(storage)
    gen1 = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    gen2 = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    # An older run's failure cannot overwrite the newer run's state.
    storage.finish_retrieved_learning_evaluation_run(
        USER, SESSION, gen1, "failed", {"error_type": "all_judges_failed"}
    )
    state_key = build_retrieved_learning_state_key(USER, SESSION)
    state_key_state = storage.get_operation_state(state_key)
    assert state_key_state is not None
    assert state_key_state["operation_state"]["status"] == "pending"
    # The current generation may record its outcome.
    storage.finish_retrieved_learning_evaluation_run(
        USER, SESSION, gen2, "failed", {"error_type": "all_judges_failed"}
    )
    state_key_state = storage.get_operation_state(state_key)
    assert state_key_state["operation_state"]["status"] == "failed"


def test_attachment_limit_aborts_scan(storage) -> None:
    refs = [
        RetrievedLearning(kind="user_playbook", learning_id=str(n))
        for n in range(1, 30)
    ]
    storage.add_request(Request(request_id="r1", user_id=USER, session_id=SESSION))
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                user_id=USER,
                request_id="r1",
                content="x",
                role="Assistant",
                retrieved_learnings=refs,
            )
        ],
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(
        USER, SESSION, raw_ref_limit=10
    )
    assert snapshot.attachment_limit_exceeded
    assert snapshot.interactions == []
    fingerprint = session_fingerprint(snapshot)
    storage.add_user_interactions_bulk(
        USER,
        [
            Interaction(
                user_id=USER,
                request_id="r1",
                content="tail",
                role="User",
            )
        ],
    )
    rescanned = storage.load_bounded_retrieved_learning_snapshot(
        USER, SESSION, raw_ref_limit=10
    )
    assert session_fingerprint(rescanned) != fingerprint


def test_snapshot_bounds_transcript_but_retains_late_refs(storage) -> None:
    storage.add_request(Request(request_id="r1", user_id=USER, session_id=SESSION))
    interactions = [
        Interaction(
            user_id=USER,
            request_id="r1",
            content="x" * 200,
            role="User",
            created_at=1_700_000_000 + index,
        )
        for index in range(20)
    ]
    interactions.append(
        Interaction(
            user_id=USER,
            request_id="r1",
            content="late attachment",
            role="Assistant",
            created_at=1_700_000_100,
            retrieved_learnings=[
                RetrievedLearning(kind="profile", learning_id="late-profile")
            ],
        )
    )
    storage.add_user_interactions_bulk(USER, interactions)

    snapshot = storage.load_bounded_retrieved_learning_snapshot(
        USER, SESSION, transcript_char_limit=128
    )

    assert (
        sum(
            len(item.role) + len(item.content) + 3
            for item in snapshot.interactions
            if item.content
        )
        <= 128
    )
    assert any(
        ("profile", "late-profile") in item.refs for item in snapshot.interactions
    )
    assert snapshot.precomputed_fingerprint


def test_results_ordering_and_filters(storage) -> None:
    _seed_eligible_learnings(storage)
    _seed_session(
        storage, refs=[RetrievedLearning(kind="profile", learning_id="prof-1")]
    )
    snapshot = storage.load_bounded_retrieved_learning_snapshot(USER, SESSION)
    fingerprint = session_fingerprint(snapshot)
    generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
    storage.replace_retrieved_learning_evaluation_results(
        USER,
        SESSION,
        generation,
        fingerprint,
        "complete",
        {},
        [_result("profile", "prof-1")],
    )
    assert storage.get_retrieved_learning_evaluation_results(user_id="other") == []
    assert storage.get_retrieved_learning_evaluation_results(session_id="other") == []
    rows = storage.get_retrieved_learning_evaluation_results(
        user_id=USER, session_id=SESSION, limit=1
    )
    assert len(rows) == 1


def test_concurrent_begin_allocates_distinct_generations(storage) -> None:
    """Two real threads racing begin() must receive distinct generations."""
    _seed_session(storage)
    generations: list[int] = []
    lock = threading.Lock()

    def begin() -> None:
        generation = storage.begin_retrieved_learning_evaluation_run(USER, SESSION)
        with lock:
            generations.append(generation)

    threads = [threading.Thread(target=begin) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(generations) == [1, 2]
