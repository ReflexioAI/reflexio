"""SQLite publication claim/stage fences reject every drifted identity.

Each case mutates exactly one durable fact behind a valid claim or stage
request and asserts that the storage call refuses, and that nothing became
visible: no staged row, no successor playbook, and no claim-fence advance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from tests.server.services.storage.test_user_playbook_publication_sqlite import (
    _request,
    _seed,
    _store,
)

pytestmark = pytest.mark.integration


def _count(storage: SQLiteStorage, table: str) -> int:
    return storage.conn.execute(
        f"SELECT COUNT(*) AS n FROM {table}"  # noqa: S608
    ).fetchone()["n"]


def _set_job(storage: SQLiteStorage, job_id: int, column: str, value) -> None:
    storage.conn.execute(
        f"UPDATE playbook_optimization_jobs SET {column} = ? WHERE job_id = ?",  # noqa: S608
        (value, job_id),
    )
    storage.conn.commit()


@pytest.mark.parametrize(
    ("job_id", "owner", "worker_fence"),
    [(0, "worker-a", 5), (1, "worker-a", 0), (1, "   ", 5)],
    ids=["job-id", "worker-fence", "owner"],
)
def test_claim_rejects_invalid_identity_without_writing(
    tmp_path: Path, job_id: int, owner: str, worker_fence: int
) -> None:
    storage = _store(tmp_path)
    _seed(storage)
    with pytest.raises(StorageError, match="claim identity is invalid"):
        storage.claim_user_playbook_publication(
            job_id=job_id, owner=owner, worker_fence=worker_fence
        )
    assert _count(storage, "user_playbook_publication_claims") == 0


def test_claim_rejects_a_missing_job(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    _seed(storage)
    with pytest.raises(StorageError, match="job does not exist"):
        storage.claim_user_playbook_publication(
            job_id=987_654, owner="worker-a", worker_fence=5
        )
    assert _count(storage, "user_playbook_publication_claims") == 0


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("target_kind", "agent_playbook", "not a user playbook"),
        ("status", "completed", "terminal"),
        ("stage", "candidate_generated", "publishing stage"),
        ("lease_owner", "worker-b", "worker owner changed"),
        ("lease_fence", 6, "worker fence changed"),
    ],
)
def test_claim_rejects_a_job_that_drifted_from_the_claimer(
    tmp_path: Path, column: str, value, message: str
) -> None:
    storage = _store(tmp_path)
    _, job = _seed(storage)
    _set_job(storage, job.job_id, column, value)
    with pytest.raises(StorageError, match=message):
        storage.claim_user_playbook_publication(
            job_id=job.job_id, owner="worker-a", worker_fence=5
        )
    assert _count(storage, "user_playbook_publication_claims") == 0


def test_claim_is_idempotent_for_its_holder_and_refuses_once_consumed(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    _, job = _seed(storage)
    first = storage.claim_user_playbook_publication(
        job_id=job.job_id, owner="worker-a", worker_fence=5
    )
    again = storage.claim_user_playbook_publication(
        job_id=job.job_id, owner="worker-a", worker_fence=5
    )
    assert again.fence == first.fence == 1
    storage.conn.execute(
        "UPDATE user_playbook_publication_claims SET consumed = 1 WHERE job_id = ?",
        (job.job_id,),
    )
    storage.conn.commit()
    with pytest.raises(StorageError, match="already consumed"):
        storage.claim_user_playbook_publication(
            job_id=job.job_id, owner="worker-a", worker_fence=5
        )
    row = storage.conn.execute(
        "SELECT publication_fence FROM user_playbook_publication_claims"
    ).fetchone()
    assert row["publication_fence"] == 1


def _claimed_request(storage: SQLiteStorage):
    incumbent, job = _seed(storage)
    claim = storage.claim_user_playbook_publication(
        job_id=job.job_id, owner="worker-a", worker_fence=5
    )
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    return incumbent, job, request


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("target_kind", "agent_playbook", "not a user playbook"),
        ("status", "failed", "terminal"),
        ("candidate_content_digest", "0" * 64, "content digest changed"),
        ("metadata_json", "{not json", "metadata is invalid"),
    ],
)
def test_stage_rejects_a_job_that_drifted_after_claim(
    tmp_path: Path, column: str, value, message: str
) -> None:
    storage = _store(tmp_path)
    _, job, request = _claimed_request(storage)
    _set_job(storage, job.job_id, column, value)
    with pytest.raises(StorageError, match=message):
        storage.stage_user_playbook_publication(request)
    assert _count(storage, "user_playbook_publication_staging") == 0


def test_stage_rejects_a_job_deleted_after_claim(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    _, job, request = _claimed_request(storage)
    storage.conn.execute(
        "DELETE FROM playbook_optimization_jobs WHERE job_id = ?", (job.job_id,)
    )
    storage.conn.commit()
    with pytest.raises(StorageError, match="job does not exist"):
        storage.stage_user_playbook_publication(request)
    assert _count(storage, "user_playbook_publication_staging") == 0


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        ("DELETE FROM user_playbook_publication_claims", "claim does not exist"),
        (
            "UPDATE user_playbook_publication_claims SET owner = 'worker-z'",
            "claim owner changed",
        ),
        (
            "UPDATE user_playbook_publication_claims SET worker_fence = 4",
            "worker fence changed",
        ),
        (
            "UPDATE user_playbook_publication_claims SET consumed = 1",
            "already consumed",
        ),
    ],
)
def test_stage_rejects_a_claim_row_that_no_longer_matches(
    tmp_path: Path, statement: str, message: str
) -> None:
    storage = _store(tmp_path)
    _, _, request = _claimed_request(storage)
    storage.conn.execute(statement)
    storage.conn.commit()
    with pytest.raises(StorageError, match=message):
        storage.stage_user_playbook_publication(request)
    assert _count(storage, "user_playbook_publication_staging") == 0


def test_stage_rejects_an_incumbent_deleted_after_claim(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, _, request = _claimed_request(storage)
    storage.conn.execute(
        "DELETE FROM user_playbooks WHERE user_playbook_id = ?",
        (incumbent.user_playbook_id,),
    )
    storage.conn.commit()
    with pytest.raises(StorageError, match="incumbent does not exist"):
        storage.stage_user_playbook_publication(request)
    assert _count(storage, "user_playbook_publication_staging") == 0


def test_staged_request_cannot_be_rebound_to_a_different_digest(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    _, _, request = _claimed_request(storage)
    storage.stage_user_playbook_publication(request)
    # Same binding fields, different request id: the staging digest differs.
    changed = replace(request, request_id="publish-request-2")
    with pytest.raises(StorageError, match="staged publication conflicts"):
        storage.stage_user_playbook_publication(changed)
    row = storage.conn.execute(
        "SELECT request_id FROM user_playbook_publication_staging"
    ).fetchone()
    assert row["request_id"] == "publish-request-1"


def test_subject_epochs_for_missing_incumbent_is_refused(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, _ = _seed(storage)
    epochs = storage.get_user_playbook_publication_subject_epochs(
        incumbent.user_playbook_id
    )
    assert '"epoch":0' in epochs
    with pytest.raises(StorageError, match="incumbent does not exist"):
        storage.get_user_playbook_publication_subject_epochs(987_654)
