from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import PlaybookOptimizationJob, UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    PublicationClaim,
    PublicationRequest,
    PublicationSearchProjection,
    UserPlaybookPublicationService,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _store(tmp_path: Path) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[9.0] * 512):
        store = SQLiteStorage(
            org_id="publication-sqlite", db_path=str(tmp_path / "r.db")
        )
    store._get_embedding = Mock(return_value=[9.0] * 512)  # noqa: SLF001
    store.llm_client.get_embeddings = Mock(return_value=[[9.0] * 512])
    return store


def _incumbent() -> UserPlaybook:
    return UserPlaybook(
        user_id="u1",
        agent_version="agent-v1",
        request_id="seed-request",
        playbook_name="refund",
        content="old content",
        trigger="refund trigger",
        rationale="keep customer calm",
        tags=["billing"],
        source_interaction_ids=[1, 2],
        source="seed",
    )


def _job(
    *,
    target_id: int,
    attempt_key: str = "attempt-1",
    worker_fence: int = 5,
    projection_digest: str,
    content_digest: str,
    proof_digest: str,
    stage: str | None = "publishing",
) -> PlaybookOptimizationJob:
    return PlaybookOptimizationJob(
        optimizer_kind="gepa",
        target_kind="user_playbook",
        target_id=target_id,
        status="running",
        metadata_json=_canonical({"publication_proof_digest": proof_digest}),
        attempt_key=attempt_key,
        lease_owner="worker-a",
        lease_fence=worker_fence,
        stage=stage,  # type: ignore[arg-type]
        candidate_content_digest=content_digest,
        search_projection_digest=projection_digest,
    )


def _projection(content: str = "new content") -> PublicationSearchProjection:
    embedding = ["0.25"] * 512
    payload = {
        "content_digest": _digest(content),
        "embedding": embedding,
        "embedding_model_id": "test-embedding-v1",
        "expanded_terms": ["exact-expanded", "projection-token"],
        "lexical_document": "exact lexical projection-token",
        "schema_version": "publication-search-projection-v1",
        "trigger": "refund trigger",
    }
    canonical = _canonical(payload)
    return PublicationSearchProjection(
        schema_version="publication-search-projection-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        content_digest=_digest(content),
        trigger="refund trigger",
        embedding_model_id="test-embedding-v1",
        embedding=tuple(embedding),
        expanded_terms=("exact-expanded", "projection-token"),
        lexical_document="exact lexical projection-token",
    )


def _proof(source: str = "playbook_optimizer") -> DecisionProofEnvelope:
    payload = {
        "adoption": {"min_commit_windows": 1, "score": 0.91},
        "decision": "apply",
        "optimizer_kind": "gepa",
        "schema_version": "gepa-publication-proof-v1",
        "source": source,
    }
    canonical = _canonical(payload)
    return DecisionProofEnvelope(
        optimizer_kind="gepa",
        schema_version="gepa-publication-proof-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        decision="apply",
    )


def _request(
    *,
    job_id: int,
    incumbent_id: int,
    claim: PublicationClaim,
    worker_fence: int = 5,
    content: str = "new content",
    projection: PublicationSearchProjection | None = None,
    proof: DecisionProofEnvelope | None = None,
    request_id: str = "publish-request-1",
    subject_epochs_json: str | None = None,
) -> PublicationRequest:
    return PublicationRequest(
        optimizer_kind="gepa",
        job_id=job_id,
        attempt_key="attempt-1",
        publication_claim=claim,
        worker_fence=worker_fence,
        incumbent_user_playbook_id=incumbent_id,
        revised_content=content,
        projection=projection or _projection(content),
        decision_proof=proof or _proof(),
        subject_epochs_json=subject_epochs_json
        or _canonical({"subjects": [{"epoch": 0, "ref": "user:u1"}]}),
        request_id=request_id,
    )


def _seed(storage: SQLiteStorage) -> tuple[UserPlaybook, PlaybookOptimizationJob]:
    incumbent = _incumbent()
    storage.save_user_playbooks([incumbent])
    projection = _projection()
    proof = _proof()
    job = storage.create_playbook_optimization_job(
        _job(
            target_id=incumbent.user_playbook_id,
            projection_digest=projection.digest,
            content_digest=projection.content_digest,
            proof_digest=proof.digest,
        )
    )
    return incumbent, job


def test_stage_is_hidden_from_user_playbook_reads_and_search(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    service.stage(request)

    assert storage.get_user_playbooks(query="new content") == []
    assert (
        storage.search_user_playbooks(
            SearchUserPlaybookRequest(user_id="u1", query="projection-token")
        )
        == []
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


def test_publish_commits_exact_staged_projection_and_terminal_result(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    service.stage(request)
    result = service.publish(request)

    assert result.outcome == "applied"
    assert result.successor_user_playbook_id is not None
    successors = storage.get_user_playbooks(
        user_playbook_id=result.successor_user_playbook_id,
        include_embedding=True,
    )
    assert len(successors) == 1
    successor = successors[0]
    assert successor.content == "new content"
    assert successor.embedding == [0.25] * 512
    assert successor.expanded_terms == "exact-expanded projection-token"
    tombstone = storage.get_user_playbook_by_id(
        incumbent.user_playbook_id, include_tombstones=True
    )
    assert tombstone is not None
    assert tombstone.status is Status.SUPERSEDED
    assert tombstone.superseded_by == result.successor_user_playbook_id

    fts = storage.conn.execute(
        "SELECT search_text FROM user_playbooks_fts WHERE rowid = ?",
        (result.successor_user_playbook_id,),
    ).fetchone()
    assert fts["search_text"] == "exact lexical projection-token"
    terminal = service.load_committed(job.job_id)
    assert terminal == result


def test_publish_lost_incumbent_cas_returns_incumbent_changed_without_orphan(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    storage.archive_user_playbook_by_id("u1", incumbent.user_playbook_id)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    result = service.publish(request)

    assert result.outcome == "incumbent_changed"
    assert result.successor_user_playbook_id is None
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("worker_fence", "worker fence"),
        ("publication_fence", "publication fence"),
        ("epochs", "subject epochs"),
        ("proof", "proof digest"),
        ("projection", "projection digest"),
    ],
)
def test_publish_rejects_changed_identity_fences_and_digests(
    tmp_path: Path, field: str, message: str
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    service.stage(request)

    if field == "worker_fence":
        bad = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
            worker_fence=4,
        )
    elif field == "publication_fence":
        bad = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=PublicationClaim(
                job_id=job.job_id, owner="worker-a", fence=claim.fence + 1
            ),
        )
    elif field == "epochs":
        bad = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
            subject_epochs_json=_canonical(
                {"subjects": [{"epoch": 1, "ref": "user:u1"}]}
            ),
        )
    elif field == "proof":
        proof = _proof(source="changed-source")
        bad = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
            proof=proof,
        )
    else:
        embedding = ["0.5"] * 512
        payload = {
            "content_digest": _digest("new content"),
            "embedding": embedding,
            "embedding_model_id": "test-embedding-v1",
            "expanded_terms": ["changed-expanded"],
            "lexical_document": "changed lexical document",
            "schema_version": "publication-search-projection-v1",
            "trigger": "refund trigger",
        }
        canonical = _canonical(payload)
        projection = PublicationSearchProjection(
            schema_version="publication-search-projection-v1",
            canonical_json=canonical,
            digest=_digest(canonical),
            content_digest=_digest("new content"),
            trigger="refund trigger",
            embedding_model_id="test-embedding-v1",
            embedding=tuple(embedding),
            expanded_terms=("changed-expanded",),
            lexical_document="changed lexical document",
        )
        bad = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
            projection=projection,
        )

    with pytest.raises(StorageError, match=message):
        service.publish(bad)
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


def test_stage_idempotent_for_identical_request_and_rejects_conflict(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    service.stage(request)
    service.stage(request)
    with pytest.raises(StorageError, match="staged publication conflicts"):
        service.stage(
            _request(
                job_id=job.job_id,
                incumbent_id=incumbent.user_playbook_id,
                claim=claim,
                content="different content",
            )
        )


def test_committed_response_loss_retry_returns_same_successor(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    first = service.publish(request)
    retry = service.publish(request)

    assert retry == first
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 1
    )
    assert len(storage.get_lineage_events(entity_type="user_playbook")) == 1
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM playbook_optimization_events WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()["count"]
        == 1
    )


def test_two_concurrent_publishers_have_one_successor(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    first_service = UserPlaybookPublicationService(storage)
    second_service = UserPlaybookPublicationService(_store(tmp_path))
    first_claim = first_service.claim(
        job_id=job.job_id, owner="worker-a", worker_fence=5
    )
    second_claim = second_service.claim(
        job_id=job.job_id, owner="worker-a", worker_fence=5
    )
    first_request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=first_claim
    )
    second_request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=second_claim
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: item[0].publish(item[1]),
                [(first_service, first_request), (second_service, second_request)],
            )
        )

    assert all(result.outcome == "applied" for result in results)
    assert results[0] == results[1]
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 1
    )


def test_erasure_barrier_added_after_staging_rejects_publication(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    service.stage(request)
    subject_ref = storage.conn.execute(
        "SELECT governance_subject_ref FROM user_playbooks WHERE user_playbook_id = ?",
        (incumbent.user_playbook_id,),
    ).fetchone()["governance_subject_ref"]
    storage.conn.execute(
        """INSERT INTO subject_write_barriers
           (org_id, subject_ref, purge_id, status, created_at, updated_at)
           VALUES (?, ?, 'purge-publication', 'erased', 1, 1)""",
        (storage.org_id, subject_ref),
    )
    storage.conn.commit()

    with pytest.raises(StorageError, match="blocked by erasure barrier"):
        service.publish(request)

    assert service.load_committed(job.job_id) is None
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


@pytest.mark.parametrize(
    ("job_change", "message"),
    [
        ({"stage": "replay_evaluated"}, "publishing"),
        ({"attempt_key": "changed-attempt"}, "attempt"),
        ({"optimizer_kind": "offline_tuner_replay"}, "optimizer"),
        ({"target_id": 999}, "incumbent"),
    ],
)
def test_publish_rejects_changed_durable_job_identity(
    tmp_path: Path, job_change: dict[str, object], message: str
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    service.stage(request)
    column, value = next(iter(job_change.items()))
    storage.conn.execute(
        f"UPDATE playbook_optimization_jobs SET {column} = ? WHERE job_id = ?",  # noqa: S608
        (value, job.job_id),
    )
    storage.conn.commit()

    with pytest.raises(StorageError, match=message):
        service.publish(request)
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


def test_failure_inside_atomic_commit_rolls_back_every_visible_effect(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = UserPlaybookPublicationService(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    service.stage(request)

    with (
        patch(
            "reflexio.server.services.storage.sqlite_storage.playbook._user._append_event_stmt",
            side_effect=RuntimeError("injected publication crash"),
        ),
        pytest.raises(StorageError, match="injected publication crash"),
    ):
        service.publish(request)

    current = storage.get_user_playbook_by_id(
        incumbent.user_playbook_id, include_tombstones=True
    )
    assert current is not None and current.status is None
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks_fts WHERE search_text = ?",
            (request.projection.lexical_document,),
        ).fetchone()["count"]
        == 0
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM playbook_optimization_events WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()["count"]
        == 0
    )
    assert service.load_committed(job.job_id) is None


def test_verifier_rejection_happens_before_hidden_staging(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    verifier = Mock()
    verifier.verify.side_effect = ValueError("proof rejected")
    service = UserPlaybookPublicationService(storage, verifier=verifier)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    with pytest.raises(ValueError, match="proof rejected"):
        service.publish(request)

    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbook_publication_staging"
        ).fetchone()["count"]
        == 0
    )
