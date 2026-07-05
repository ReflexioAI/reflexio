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
