import time
import uuid

import pytest

pytestmark = pytest.mark.integration


class TestLearningJobsSchema:
    def test_table_exists_with_expected_columns(self, storage) -> None:
        cols = set(storage.learning_jobs_columns())
        expected = {
            "job_id",
            "org_id",
            "user_id",
            "job_type",
            "latest_request_id",
            "status",
            "attempts",
            "max_attempts",
            "claimed_by",
            "claim_token",
            "claim_expires_at",
            "covers_through",
            "created_at",
            "updated_at",
        }
        assert expected <= cols


def _enqueue(
    storage, user_id: str = "u1", request_id: str = "r1", covers_through: float = 1000.0
) -> str:
    with storage.commit_scope():
        return storage.enqueue_learning_job(
            org_id=storage.org_id,
            user_id=user_id,
            request_id=request_id,
            covers_through=covers_through,
        )


class TestCoalescing:
    def test_two_pending_publishes_collapse_to_one_job(self, storage) -> None:
        j1 = _enqueue(storage, "u-c", "r-1", 1000.0)
        j2 = _enqueue(storage, "u-c", "r-2", 2000.0)
        assert j1 == j2
        claimed = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-c"
        ]
        assert len(claimed) == 1
        assert claimed[0].covers_through == 2000.0  # max kept

    def test_different_users_get_separate_jobs(self, storage) -> None:
        j1 = _enqueue(storage, "u-x1", "r-a", 1000.0)
        j2 = _enqueue(storage, "u-x2", "r-b", 1000.0)
        assert j1 != j2

    def test_enqueue_inside_commit_scope_commits_with_scope(self, storage) -> None:
        """enqueue inside commit_scope does not issue its own BEGIN/COMMIT."""
        with storage.commit_scope():
            storage.enqueue_learning_job(
                org_id=storage.org_id,
                user_id="u-scope",
                request_id="r-scope",
                covers_through=500.0,
            )
        claimed = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-scope"
        ]
        assert len(claimed) == 1


class TestFencedCompletion:
    def test_stale_token_completion_is_noop(self, storage) -> None:
        _enqueue(storage, "u-f", "r-f", 1000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-f"
        ]
        # Wrong (random) token → rowcount 0
        assert (
            storage.complete_learning_job(
                job_id=job.job_id, claim_token=str(uuid.uuid4())
            )
            == 0
        )
        # Correct token → rowcount 1
        assert (
            storage.complete_learning_job(
                job_id=job.job_id, claim_token=job.claim_token
            )
            == 1
        )

    def test_reclaim_after_expiry_supersedes_prior_token(self, storage) -> None:
        _enqueue(storage, "u-r", "r-r", 1000.0)
        # lease_seconds=-1 sets claim_expires_at to now-1s, which is
        # deterministically < now() on the very next claim call.
        [first] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=-1
            )
            if j.user_id == "u-r"
        ]
        [second] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w2", limit=10, lease_seconds=300
            )
            if j.user_id == "u-r"
        ]
        assert second.claim_token != first.claim_token
        # First token is now stale
        assert (
            storage.complete_learning_job(
                job_id=first.job_id, claim_token=first.claim_token
            )
            == 0
        )
        # Second token succeeds
        assert (
            storage.complete_learning_job(
                job_id=second.job_id, claim_token=second.claim_token
            )
            == 1
        )

    def test_double_complete_second_is_noop(self, storage) -> None:
        """Second complete on the same job (already done) returns 0."""
        _enqueue(storage, "u-d", "r-d", 1000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-d"
        ]
        assert (
            storage.complete_learning_job(
                job_id=job.job_id, claim_token=job.claim_token
            )
            == 1
        )
        assert (
            storage.complete_learning_job(
                job_id=job.job_id, claim_token=job.claim_token
            )
            == 0
        )

    def test_fail_after_complete_is_noop(self, storage) -> None:
        """fail_learning_job on an already-done job is a no-op (status stays 'done')."""
        _enqueue(storage, "u-fc", "r-fc", 5000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-fc"
        ]
        assert (
            storage.complete_learning_job(
                job_id=job.job_id, claim_token=job.claim_token
            )
            == 1
        )
        # Attempt to fail the already-completed job — must be a no-op.
        storage.fail_learning_job(
            job_id=job.job_id, claim_token=job.claim_token, dead=False
        )
        # Status check: job is done and request is in the past → still "done".
        assert (
            storage.get_learning_status_for_request(
                user_id="u-fc", request_created_at=4000.0
            )
            == "done"
        )


class TestStatusCoverage:
    def test_zero_yield_window_reports_done(self, storage) -> None:
        _enqueue(storage, "u-z", "r-z", covers_through=5000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-z"
        ]
        storage.complete_learning_job(job_id=job.job_id, claim_token=job.claim_token)
        assert (
            storage.get_learning_status_for_request(
                user_id="u-z", request_created_at=4000.0
            )
            == "done"
        )

    def test_pending_job_reports_pending_or_processing(self, storage) -> None:
        _enqueue(storage, "u-p", "r-p", covers_through=5000.0)
        assert storage.get_learning_status_for_request(
            user_id="u-p", request_created_at=4000.0
        ) in {"pending", "processing"}

    def test_claimed_job_reports_processing(self, storage) -> None:
        _enqueue(storage, "u-cl", "r-cl", covers_through=5000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-cl"
        ]
        assert job.status == "claimed"
        assert (
            storage.get_learning_status_for_request(
                user_id="u-cl", request_created_at=4000.0
            )
            == "processing"
        )

    def test_no_rows_old_request_returns_done_absence_semantics(self, storage) -> None:
        # No rows, request far in the past (older than 72 h retention window) → "done"
        assert (
            storage.get_learning_status_for_request(
                user_id="u-absent-old", request_created_at=1000.0
            )
            == "done"
        )

    def test_no_rows_recent_request_returns_pending(self, storage) -> None:
        # No rows, but request is recent → absence-done guard fires → "pending"
        recent_ts = time.time() - 60  # 60 seconds ago — well within 72 h window
        assert (
            storage.get_learning_status_for_request(
                user_id="u-absent-recent", request_created_at=recent_ts
            )
            == "pending"
        )

    def test_no_rows_old_enough_request_returns_done(self, storage) -> None:
        # No rows, request older than the retention window (90 h) → "done"
        old_ts = time.time() - 90 * 3600
        assert (
            storage.get_learning_status_for_request(
                user_id="u-absent-old2", request_created_at=old_ts
            )
            == "done"
        )

    def test_failed_reclaimable_job_reports_pending(self, storage) -> None:
        # A failed (not dead) job is reclaimable — should surface as "pending".
        _enqueue(storage, "u-rfail", "r-rfail", covers_through=5000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-rfail"
        ]
        storage.fail_learning_job(
            job_id=job.job_id, claim_token=job.claim_token, dead=False
        )
        assert (
            storage.get_learning_status_for_request(
                user_id="u-rfail", request_created_at=4000.0
            )
            == "pending"
        )

    def test_failed_job_reports_failed(self, storage) -> None:
        _enqueue(storage, "u-fail", "r-fail", covers_through=5000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-fail"
        ]
        storage.fail_learning_job(
            job_id=job.job_id, claim_token=job.claim_token, dead=True
        )
        assert (
            storage.get_learning_status_for_request(
                user_id="u-fail", request_created_at=4000.0
            )
            == "failed"
        )


class TestHeartbeat:
    def test_heartbeat_extends_live_claim(self, storage) -> None:
        _enqueue(storage, "u-hb", "r-hb", 1000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-hb"
        ]
        assert storage.heartbeat_learning_job(
            job_id=job.job_id, claim_token=job.claim_token, lease_seconds=600
        )

    def test_heartbeat_wrong_token_returns_false(self, storage) -> None:
        _enqueue(storage, "u-hb2", "r-hb2", 1000.0)
        [job] = [
            j
            for j in storage.claim_learning_jobs(
                claimed_by="w1", limit=10, lease_seconds=300
            )
            if j.user_id == "u-hb2"
        ]
        assert not storage.heartbeat_learning_job(
            job_id=job.job_id, claim_token=str(uuid.uuid4()), lease_seconds=600
        )
