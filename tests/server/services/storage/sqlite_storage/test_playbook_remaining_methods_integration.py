"""Characterization tests: remaining uncovered SQLite playbook methods (Tier-1 Task 2).

Behavioral / round-trip tests for the methods marked NONE or INDIRECT in the
Task-1 coverage audit.  These are simple CRUD / getter methods — no atomicity
crash-window testing needed unless the method emits a lineage event (checked
per method; none of these do).

Methods covered here:

NONE (no direct test existed):
  - ``count_user_playbooks_by_session``
  - ``has_user_playbooks_with_status``
  - ``get_user_playbooks_by_ids_any_user``
  - ``insert_playbook_optimization_event``
  - ``delete_all_agent_success_evaluation_results``

INDIRECT (service-level tests existed; direct storage tests added here):
  - ``update_playbook_optimization_job``
  - ``update_playbook_optimization_candidate``
  - ``insert_playbook_optimization_evaluation``
  - ``list_playbook_optimization_evaluations``
"""

from __future__ import annotations

import pytest

from reflexio.models.api_schema.domain.entities import (
    AgentSuccessEvaluationResult,
    PlaybookOptimizationCandidate,
    PlaybookOptimizationEvaluation,
    PlaybookOptimizationEvent,
    PlaybookOptimizationJob,
    Request,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _store(tmp_path, org_id: str = "org-char") -> SQLiteStorage:
    s = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / f"{org_id}.db"))
    s.migrate()
    return s


def _make_user_playbook(
    *,
    request_id: str = "r1",
    user_id: str = "u1",
    content: str = "content",
    agent_version: str = "v1",
    playbook_name: str | None = None,
    status: Status | None = None,
) -> UserPlaybook:
    return UserPlaybook(
        user_id=user_id,
        agent_version=agent_version,
        request_id=request_id,
        content=content,
        playbook_name=playbook_name or "pb",
        status=status,
    )


def _make_request(
    *, request_id: str = "r1", user_id: str = "u1", session_id: str = "sess1"
) -> Request:
    return Request(request_id=request_id, user_id=user_id, session_id=session_id)


def _make_eval_result(
    *, user_id: str = "u1", session_id: str = "s1"
) -> AgentSuccessEvaluationResult:
    return AgentSuccessEvaluationResult(
        user_id=user_id,
        agent_version="v1",
        session_id=session_id,
        is_success=True,
    )


def _make_job(*, target_id: int = 1) -> PlaybookOptimizationJob:
    return PlaybookOptimizationJob(
        optimizer_kind="gepa", target_kind="agent_playbook", target_id=target_id
    )


def _make_candidate(
    *, job_id: int, content: str = "c"
) -> PlaybookOptimizationCandidate:
    return PlaybookOptimizationCandidate(job_id=job_id, content=content)


def _make_evaluation(
    *, job_id: int, candidate_id: int
) -> PlaybookOptimizationEvaluation:
    return PlaybookOptimizationEvaluation(
        job_id=job_id,
        candidate_id=candidate_id,
        target_kind="agent_playbook",
        target_id=1,
    )


# ---------------------------------------------------------------------------
# get_user_playbooks
# ---------------------------------------------------------------------------


class TestGetUserPlaybooks:
    """Round-trip characterization for get_user_playbooks user_id filtering."""

    def test_user_id_filter_includes_synthetic_generation_request_id(self, tmp_path):
        """Synthetic manual/rerun request ids stay visible under user_id filtering."""
        s = _store(tmp_path)
        s.add_request(_make_request(request_id="req-real", user_id="user-1"))
        real = _make_user_playbook(
            user_id="user-1",
            request_id="req-real",
            content="real request playbook",
        )
        synthetic = _make_user_playbook(
            user_id="user-1",
            request_id="rerun_playbook_ab12cd34",
            content="synthetic rerun playbook",
        )
        other_user = _make_user_playbook(
            user_id="user-2",
            request_id="manual_cd34ef56",
            content="other user synthetic playbook",
        )
        s.save_user_playbooks([real, synthetic, other_user])

        rows = s.get_user_playbooks(user_id="user-1", limit=10)

        assert {row.request_id for row in rows} == {
            "req-real",
            "rerun_playbook_ab12cd34",
        }


# ---------------------------------------------------------------------------
# count_user_playbooks_by_session
# ---------------------------------------------------------------------------


class TestCountUserPlaybooksBySession:
    """Round-trip characterization for count_user_playbooks_by_session.

    The method JOINs user_playbooks with requests on request_id, filtering by
    session_id and excluding tombstone statuses.
    """

    def test_count_matches_seeded_rows_for_session(self, tmp_path):
        """Seeded playbooks linked via request to a session are counted."""
        s = _store(tmp_path)
        s.add_request(_make_request(request_id="req1", session_id="sess-a"))
        pb1 = _make_user_playbook(request_id="req1", content="a")
        pb2 = _make_user_playbook(request_id="req1", content="b")
        s.save_user_playbooks([pb1, pb2])

        assert s.count_user_playbooks_by_session("sess-a") == 2

    def test_unknown_session_returns_zero(self, tmp_path):
        s = _store(tmp_path)
        s.add_request(_make_request(request_id="req1", session_id="sess-a"))
        s.save_user_playbooks([_make_user_playbook(request_id="req1")])

        assert s.count_user_playbooks_by_session("sess-NOT-PRESENT") == 0

    def test_empty_storage_returns_zero(self, tmp_path):
        s = _store(tmp_path)
        assert s.count_user_playbooks_by_session("any-session") == 0

    def test_tombstone_playbooks_excluded_from_count(self, tmp_path):
        """Superseded/merged playbooks must not count against the session total."""
        s = _store(tmp_path)
        s.add_request(_make_request(request_id="req1", session_id="sess-x"))
        pb_current = _make_user_playbook(request_id="req1", content="live")
        pb_superseded = _make_user_playbook(
            request_id="req1", content="dead", status=Status.SUPERSEDED
        )
        s.save_user_playbooks([pb_current, pb_superseded])

        # Only the live (non-tombstone) row should count.
        assert s.count_user_playbooks_by_session("sess-x") == 1

    def test_multiple_sessions_counted_independently(self, tmp_path):
        """Counts are scoped to the requested session_id only."""
        s = _store(tmp_path)
        s.add_request(_make_request(request_id="req-a", session_id="sess-1"))
        s.add_request(_make_request(request_id="req-b", session_id="sess-2"))
        pb_a = _make_user_playbook(request_id="req-a", content="x")
        pb_b1 = _make_user_playbook(request_id="req-b", content="y1")
        pb_b2 = _make_user_playbook(request_id="req-b", content="y2")
        s.save_user_playbooks([pb_a, pb_b1, pb_b2])

        assert s.count_user_playbooks_by_session("sess-1") == 1
        assert s.count_user_playbooks_by_session("sess-2") == 2


# ---------------------------------------------------------------------------
# has_user_playbooks_with_status
# ---------------------------------------------------------------------------


class TestHasUserPlaybooksWithStatus:
    """Round-trip characterization for has_user_playbooks_with_status."""

    def test_none_status_finds_null_status_rows(self, tmp_path):
        """status=None means WHERE status IS NULL (default CURRENT rows)."""
        s = _store(tmp_path)
        s.save_user_playbooks([_make_user_playbook(status=None)])
        assert s.has_user_playbooks_with_status(None) is True

    def test_archived_status_found_after_save(self, tmp_path):
        s = _store(tmp_path)
        s.save_user_playbooks([_make_user_playbook(status=Status.ARCHIVED)])
        assert s.has_user_playbooks_with_status(Status.ARCHIVED) is True

    def test_status_not_present_returns_false(self, tmp_path):
        """No rows with ARCHIVED status → False even when CURRENT rows exist."""
        s = _store(tmp_path)
        s.save_user_playbooks([_make_user_playbook(status=None)])
        assert s.has_user_playbooks_with_status(Status.ARCHIVED) is False

    def test_empty_storage_returns_false(self, tmp_path):
        s = _store(tmp_path)
        assert s.has_user_playbooks_with_status(None) is False

    def test_agent_version_filter_narrows_result(self, tmp_path):
        """agent_version filter isolates rows to the matching version only."""
        s = _store(tmp_path)
        s.save_user_playbooks([_make_user_playbook(agent_version="v1")])
        assert s.has_user_playbooks_with_status(None, agent_version="v1") is True
        assert s.has_user_playbooks_with_status(None, agent_version="v99") is False

    def test_playbook_name_filter_narrows_result(self, tmp_path):
        """playbook_name filter isolates rows to the matching name only."""
        s = _store(tmp_path)
        s.save_user_playbooks([_make_user_playbook(playbook_name="alpha")])
        assert s.has_user_playbooks_with_status(None, playbook_name="alpha") is True
        assert s.has_user_playbooks_with_status(None, playbook_name="beta") is False


# ---------------------------------------------------------------------------
# get_user_playbooks_by_ids_any_user
# ---------------------------------------------------------------------------


class TestGetUserPlaybooksByIdsAnyUser:
    """Round-trip characterization for get_user_playbooks_by_ids_any_user.

    This variant skips the user_id scope check — it returns playbooks from any
    user. Contrast with get_user_playbooks_by_ids which requires user_id match.
    """

    def test_empty_list_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        assert s.get_user_playbooks_by_ids_any_user([]) == []

    def test_returns_playbook_regardless_of_user(self, tmp_path):
        """Can fetch playbooks owned by different users in one call."""
        s = _store(tmp_path)
        pb1 = _make_user_playbook(user_id="alice", content="a")
        pb2 = _make_user_playbook(user_id="bob", content="b")
        s.save_user_playbooks([pb1, pb2])

        results = s.get_user_playbooks_by_ids_any_user(
            [pb1.user_playbook_id, pb2.user_playbook_id]
        )
        ids_returned = {r.user_playbook_id for r in results}
        assert pb1.user_playbook_id in ids_returned
        assert pb2.user_playbook_id in ids_returned
        assert len(results) == 2

    def test_nonexistent_id_silently_omitted(self, tmp_path):
        s = _store(tmp_path)
        pb = _make_user_playbook()
        s.save_user_playbooks([pb])
        results = s.get_user_playbooks_by_ids_any_user([pb.user_playbook_id, 999999])
        assert len(results) == 1
        assert results[0].user_playbook_id == pb.user_playbook_id

    def test_preserves_input_id_order(self, tmp_path):
        """Returned list preserves the order of the input id list."""
        s = _store(tmp_path)
        pb1 = _make_user_playbook(content="first")
        pb2 = _make_user_playbook(content="second")
        pb3 = _make_user_playbook(content="third")
        s.save_user_playbooks([pb1, pb2, pb3])

        ids_reversed = [
            pb3.user_playbook_id,
            pb1.user_playbook_id,
            pb2.user_playbook_id,
        ]
        results = s.get_user_playbooks_by_ids_any_user(ids_reversed)
        assert [r.user_playbook_id for r in results] == ids_reversed

    def test_status_filter_none_means_no_filter(self, tmp_path):
        """status_filter=None returns rows regardless of status (including tombstones)."""
        s = _store(tmp_path)
        pb_current = _make_user_playbook(content="live", status=None)
        pb_superseded = _make_user_playbook(content="dead", status=Status.SUPERSEDED)
        s.save_user_playbooks([pb_current, pb_superseded])

        # status_filter=None → no WHERE clause on status, both rows returned
        results = s.get_user_playbooks_by_ids_any_user(
            [pb_current.user_playbook_id, pb_superseded.user_playbook_id],
            status_filter=None,
        )
        assert len(results) == 2

    def test_status_filter_restricts_to_matching_status(self, tmp_path):
        """status_filter=[Status.SUPERSEDED] returns only superseded rows."""
        s = _store(tmp_path)
        pb_current = _make_user_playbook(content="live", status=None)
        pb_superseded = _make_user_playbook(content="dead", status=Status.SUPERSEDED)
        s.save_user_playbooks([pb_current, pb_superseded])

        results = s.get_user_playbooks_by_ids_any_user(
            [pb_current.user_playbook_id, pb_superseded.user_playbook_id],
            status_filter=[Status.SUPERSEDED],
        )
        assert len(results) == 1
        assert results[0].user_playbook_id == pb_superseded.user_playbook_id


# ---------------------------------------------------------------------------
# insert_playbook_optimization_event
# ---------------------------------------------------------------------------


class TestInsertPlaybookOptimizationEvent:
    """Round-trip characterization for insert_playbook_optimization_event."""

    def test_event_id_assigned_after_insert(self, tmp_path):
        """Returned event has event_id > 0 (storage assigned PK)."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        event = PlaybookOptimizationEvent(
            job_id=job.job_id, event_type="candidate_generated", payload_json='{"n": 1}'
        )
        returned = s.insert_playbook_optimization_event(event)
        assert returned.event_id > 0

    def test_fields_round_trip_correctly(self, tmp_path):
        """All fields survive the INSERT → SELECT cycle intact."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        payload = '{"key": "value"}'
        event = PlaybookOptimizationEvent(
            job_id=job.job_id,
            event_type="evaluation_complete",
            payload_json=payload,
        )
        returned = s.insert_playbook_optimization_event(event)

        row = s.conn.execute(
            "SELECT * FROM playbook_optimization_events WHERE event_id = ?",
            (returned.event_id,),
        ).fetchone()
        assert row is not None
        assert row["job_id"] == job.job_id
        assert row["event_type"] == "evaluation_complete"
        assert row["payload_json"] == payload

    def test_multiple_events_stored_per_job(self, tmp_path):
        """Multiple events for the same job are all persisted."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        for i in range(3):
            s.insert_playbook_optimization_event(
                PlaybookOptimizationEvent(job_id=job.job_id, event_type=f"ev_{i}")
            )
        count = s.conn.execute(
            "SELECT COUNT(*) FROM playbook_optimization_events WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()[0]
        assert count == 3


# ---------------------------------------------------------------------------
# delete_all_agent_success_evaluation_results
# ---------------------------------------------------------------------------


class TestDeleteAllAgentSuccessEvaluationResults:
    """Round-trip characterization for delete_all_agent_success_evaluation_results."""

    def test_seeded_rows_are_gone_after_delete(self, tmp_path):
        s = _store(tmp_path)
        s.save_agent_success_evaluation_results(
            [
                _make_eval_result(user_id="u1", session_id="s1"),
                _make_eval_result(user_id="u2", session_id="s2"),
            ]
        )
        assert len(s.get_agent_success_evaluation_results(limit=100)) == 2

        s.delete_all_agent_success_evaluation_results()

        assert s.get_agent_success_evaluation_results(limit=100) == []

    def test_idempotent_on_empty_table(self, tmp_path):
        """Calling on an empty table raises no error and leaves the table empty."""
        s = _store(tmp_path)
        s.delete_all_agent_success_evaluation_results()
        assert s.get_agent_success_evaluation_results(limit=100) == []


# ---------------------------------------------------------------------------
# update_playbook_optimization_job (INDIRECT → direct storage test)
# ---------------------------------------------------------------------------


class TestUpdatePlaybookOptimizationJob:
    """Direct storage characterization for update_playbook_optimization_job."""

    def _read_job(self, s: SQLiteStorage, job_id: int) -> dict:
        row = s.conn.execute(
            "SELECT * FROM playbook_optimization_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        assert row is not None, f"job_id {job_id} not found"
        return dict(row)

    def test_status_field_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        assert self._read_job(s, job.job_id)["status"] == "pending"

        s.update_playbook_optimization_job(job.job_id, status="running")

        assert self._read_job(s, job.job_id)["status"] == "running"

    def test_best_candidate_id_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )

        s.update_playbook_optimization_job(
            job.job_id, best_candidate_id=cand.candidate_id
        )

        assert self._read_job(s, job.job_id)["best_candidate_id"] == cand.candidate_id

    def test_decision_reason_and_metadata_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())

        s.update_playbook_optimization_job(
            job.job_id,
            decision_reason="candidate beats incumbent",
            metadata_json='{"runs": 3}',
        )

        row = self._read_job(s, job.job_id)
        assert row["decision_reason"] == "candidate beats incumbent"
        assert row["metadata_json"] == '{"runs": 3}'

    def test_successor_target_id_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())

        s.update_playbook_optimization_job(job.job_id, successor_target_id=42)

        assert self._read_job(s, job.job_id)["successor_target_id"] == 42

    def test_nonexistent_job_id_is_noop(self, tmp_path):
        """Updating a job_id that does not exist raises no error."""
        s = _store(tmp_path)
        s.update_playbook_optimization_job(99999, status="completed")  # must not raise


# ---------------------------------------------------------------------------
# update_playbook_optimization_candidate (INDIRECT → direct storage test)
# ---------------------------------------------------------------------------


class TestUpdatePlaybookOptimizationCandidate:
    """Direct storage characterization for update_playbook_optimization_candidate."""

    def test_aggregate_score_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )
        assert cand.aggregate_score is None

        s.update_playbook_optimization_candidate(
            cand.candidate_id, aggregate_score=0.85
        )

        [updated] = s.list_playbook_optimization_candidates(job.job_id)
        assert updated.aggregate_score == pytest.approx(0.85)

    def test_is_winner_updated(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )
        assert cand.is_winner is False

        s.update_playbook_optimization_candidate(cand.candidate_id, is_winner=True)

        [updated] = s.list_playbook_optimization_candidates(job.job_id)
        assert updated.is_winner is True

    def test_both_fields_updated_together(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )

        s.update_playbook_optimization_candidate(
            cand.candidate_id, aggregate_score=0.72, is_winner=True
        )

        [updated] = s.list_playbook_optimization_candidates(job.job_id)
        assert updated.aggregate_score == pytest.approx(0.72)
        assert updated.is_winner is True

    def test_no_kwargs_is_noop(self, tmp_path):
        """Calling with no fields to update raises no error and leaves row unchanged."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id, content="unchanged")
        )

        s.update_playbook_optimization_candidate(cand.candidate_id)

        [still_there] = s.list_playbook_optimization_candidates(job.job_id)
        assert still_there.content == "unchanged"
        assert still_there.is_winner is False

    def test_nonexistent_candidate_id_is_noop(self, tmp_path):
        s = _store(tmp_path)
        s.update_playbook_optimization_candidate(99999, aggregate_score=1.0)


# ---------------------------------------------------------------------------
# insert_playbook_optimization_evaluation (INDIRECT → direct storage test)
# ---------------------------------------------------------------------------


class TestInsertPlaybookOptimizationEvaluation:
    """Direct storage characterization for insert_playbook_optimization_evaluation."""

    def test_evaluation_id_assigned_after_insert(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )

        ev = s.insert_playbook_optimization_evaluation(
            _make_evaluation(job_id=job.job_id, candidate_id=cand.candidate_id)
        )
        assert ev.evaluation_id > 0

    def test_fields_round_trip_correctly(self, tmp_path):
        """All key fields survive INSERT → list round-trip."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )
        ev_in = PlaybookOptimizationEvaluation(
            job_id=job.job_id,
            candidate_id=cand.candidate_id,
            target_kind="agent_playbook",
            target_id=1,
            score=0.9,
            verdict="candidate",
            likert=4,
            rationale="better output",
        )
        returned = s.insert_playbook_optimization_evaluation(ev_in)

        [ev_out] = s.list_playbook_optimization_evaluations(job.job_id)
        assert ev_out.evaluation_id == returned.evaluation_id
        assert ev_out.job_id == job.job_id
        assert ev_out.candidate_id == cand.candidate_id
        assert ev_out.score == pytest.approx(0.9)
        assert ev_out.verdict == "candidate"
        assert ev_out.likert == 4
        assert ev_out.rationale == "better output"

    def test_multiple_evaluations_linked_to_job(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )

        for _ in range(3):
            s.insert_playbook_optimization_evaluation(
                _make_evaluation(job_id=job.job_id, candidate_id=cand.candidate_id)
            )

        evs = s.list_playbook_optimization_evaluations(job.job_id)
        assert len(evs) == 3


# ---------------------------------------------------------------------------
# list_playbook_optimization_evaluations (INDIRECT → direct storage test)
# ---------------------------------------------------------------------------


class TestListPlaybookOptimizationEvaluations:
    """Direct storage characterization for list_playbook_optimization_evaluations."""

    def test_empty_when_no_evaluations(self, tmp_path):
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        assert s.list_playbook_optimization_evaluations(job.job_id) == []

    def test_evaluations_ordered_by_evaluation_id_asc(self, tmp_path):
        """Rows returned in ascending evaluation_id order (ORDER BY evaluation_id ASC)."""
        s = _store(tmp_path)
        job = s.create_playbook_optimization_job(_make_job())
        cand = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job.job_id)
        )

        ids_inserted: list[int] = []
        for _ in range(4):
            ev = s.insert_playbook_optimization_evaluation(
                _make_evaluation(job_id=job.job_id, candidate_id=cand.candidate_id)
            )
            ids_inserted.append(ev.evaluation_id)

        evs = s.list_playbook_optimization_evaluations(job.job_id)
        assert [e.evaluation_id for e in evs] == sorted(ids_inserted)

    def test_scoped_to_job_id(self, tmp_path):
        """list_playbook_optimization_evaluations filters by job_id only."""
        s = _store(tmp_path)
        job_a = s.create_playbook_optimization_job(_make_job(target_id=1))
        job_b = s.create_playbook_optimization_job(_make_job(target_id=2))
        cand_a = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job_a.job_id)
        )
        cand_b = s.insert_playbook_optimization_candidate(
            _make_candidate(job_id=job_b.job_id)
        )

        for _ in range(2):
            s.insert_playbook_optimization_evaluation(
                _make_evaluation(job_id=job_a.job_id, candidate_id=cand_a.candidate_id)
            )
        s.insert_playbook_optimization_evaluation(
            _make_evaluation(job_id=job_b.job_id, candidate_id=cand_b.candidate_id)
        )

        assert len(s.list_playbook_optimization_evaluations(job_a.job_id)) == 2
        assert len(s.list_playbook_optimization_evaluations(job_b.job_id)) == 1

    def test_unknown_job_id_returns_empty(self, tmp_path):
        s = _store(tmp_path)
        assert s.list_playbook_optimization_evaluations(99999) == []
