from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import (
    PlaybookOptimizationCandidate,
    PlaybookOptimizationJob,
    UserPlaybook,
)
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    PublicationClaim,
    PublicationRequest,
    PublicationSearchProjection,
    UserPlaybookPublicationService,
    incumbent_user_playbook_semantic_digest,
    publication_source_for_optimizer,
)
from reflexio.server.services.playbook_optimizer.gepa_publication import (
    GEPA_PUBLICATION_AUTHORITY_METADATA_KEY,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_ORG_ID = "publication-sqlite"
_LIVE_LEASE_EXPIRY = 4_000_000_000
_EPOCH_NOW_PATCH = (
    "reflexio.server.services.storage.sqlite_storage.playbook._user._epoch_now"
)


@pytest.mark.parametrize(
    ("optimizer_kind", "expected_source"),
    [("gepa", "gepa"), ("offline_tuner_replay", "offline_optimizer")],
)
def test_publication_source_is_explicitly_mapped(
    optimizer_kind: str, expected_source: str
) -> None:
    assert publication_source_for_optimizer(optimizer_kind) == expected_source  # type: ignore[arg-type]


def _canonical(payload: dict[str, object]) -> str:
    return json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _store(tmp_path: Path) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[9.0] * 512):
        store = SQLiteStorage(org_id=_ORG_ID, db_path=str(tmp_path / "r.db"))
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
    subject_epochs_json: str,
    stage: str | None = "publishing",
) -> PlaybookOptimizationJob:
    return PlaybookOptimizationJob(
        optimizer_kind="gepa",
        target_kind="user_playbook",
        target_id=target_id,
        status="running",
        metadata_json=_canonical(
            {
                "publication_proof_digest": proof_digest,
                "publication_subject_epochs": json.loads(subject_epochs_json),
            }
        ),
        attempt_key=attempt_key,
        lease_owner="worker-a",
        lease_fence=worker_fence,
        lease_expires_at=_LIVE_LEASE_EXPIRY,
        stage=stage,  # type: ignore[arg-type]
        candidate_content_digest=content_digest,
        search_projection_digest=projection_digest,
    )


def _projection(
    content: str = "new content", *, trigger: str | None = "refund trigger"
) -> PublicationSearchProjection:
    embedding = ["0.25"] * 512
    payload = {
        "candidate_content_digest": _digest(content),
        "embedding": embedding,
        "embedding_model_id": "test-embedding-v1",
        "expanded_terms": ["exact-expanded", "projection-token"],
        "lexical_document": "exact lexical projection-token",
        "preserved_trigger": trigger,
        "projector_code_digest": "a" * 64,
        "projector_id": "reflexio.search.user-playbook",
        "projector_version": "1",
        "schema_version": "offline-tuner-candidate-search-projection-v1",
    }
    canonical = _canonical(payload)
    return PublicationSearchProjection(
        schema_version="offline-tuner-candidate-search-projection-v1",
        canonical_json=canonical,
        digest=_digest(canonical),
        projector_id="reflexio.search.user-playbook",
        projector_version="1",
        projector_code_digest="a" * 64,
        candidate_content_digest=_digest(content),
        preserved_trigger=trigger,
        embedding_model_id="test-embedding-v1",
        embedding=tuple(embedding),
        expanded_terms=("exact-expanded", "projection-token"),
        lexical_document="exact lexical projection-token",
    )


def _proof(source: str = "playbook_optimizer") -> DecisionProofEnvelope:
    payload = {
        "adoption": {"min_commit_windows": 1, "score": "0.91"},
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
        incumbent_content_digest=_digest("old content"),
        incumbent_trigger="refund trigger",
        incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
            content_digest=_digest("old content"), trigger="refund trigger"
        ),
        revised_content=content,
        projection=projection or _projection(content),
        decision_proof=proof or _proof(),
        subject_epochs_json=subject_epochs_json or _subject_epochs_json(),
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
            content_digest=projection.candidate_content_digest,
            proof_digest=proof.digest,
            subject_epochs_json=_subject_epochs_json(),
        )
    )
    return incumbent, job


def test_prepare_gepa_publication_rejects_authority_substitution_before_metadata_overwrite(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent = _incumbent()
    storage.save_user_playbooks([incumbent])
    durable_authority = {
        "adoption_policy": {"min_commit_score": "0.75"},
        "validation_manifest": {"windows": []},
    }
    substituted_authority = {
        "adoption_policy": {"min_commit_score": "0.25"},
        "validation_manifest": {"windows": []},
    }
    job = storage.create_playbook_optimization_job(
        PlaybookOptimizationJob(
            optimizer_kind="gepa",
            target_kind="user_playbook",
            target_id=incumbent.user_playbook_id,
            status="running",
            metadata_json=_canonical(
                {GEPA_PUBLICATION_AUTHORITY_METADATA_KEY: durable_authority}
            ),
            attempt_key="attempt-prepare",
        )
    )
    candidate = storage.insert_playbook_optimization_candidate(
        PlaybookOptimizationCandidate(
            job_id=job.job_id,
            content="new content",
            aggregate_score=0.9,
            is_winner=True,
        )
    )
    projection = _projection("new content")
    proof = _proof("substituted-proof")
    incoming_metadata = _canonical(
        {
            GEPA_PUBLICATION_AUTHORITY_METADATA_KEY: substituted_authority,
            "best_idx": 0,
        }
    )

    with pytest.raises(StorageError, match="authority"):
        storage.prepare_gepa_user_playbook_publication(
            job_id=job.job_id,
            owner="worker-a",
            lease_seconds=60,
            winner_candidate_id=candidate.candidate_id,
            candidate_content_digest=projection.candidate_content_digest,
            search_projection_digest=projection.digest,
            publication_proof_digest=proof.digest,
            projection_json=projection.canonical_json,
            decision_proof_json=proof.canonical_json,
            subject_epochs_json=storage.get_user_playbook_publication_subject_epochs(
                incumbent.user_playbook_id
            ),
            metadata_json=incoming_metadata,
        )

    row = storage.conn.execute(
        "SELECT metadata_json, stage FROM playbook_optimization_jobs WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert json.loads(row["metadata_json"])[
        GEPA_PUBLICATION_AUTHORITY_METADATA_KEY
    ] == (durable_authority)
    assert row["stage"] is None


def _subject_ref() -> str:
    return governance_subject_ref(_ORG_ID, "u1", get_governance_ref_secret())


def _subject_epochs_json(*, epoch: int = 0, subject_ref: str | None = None) -> str:
    return _canonical(
        {"subjects": [{"epoch": epoch, "ref": subject_ref or _subject_ref()}]}
    )


class _AcceptingVerifier:
    def verify(self, request: PublicationRequest) -> None:
        assert request.optimizer_kind == "gepa"


def _service(storage: SQLiteStorage) -> UserPlaybookPublicationService:
    return UserPlaybookPublicationService(storage, verifier=_AcceptingVerifier())


def test_stage_is_hidden_from_user_playbook_reads_and_search(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )

    service.stage(request)
    staged = storage.conn.execute(
        "SELECT * FROM user_playbook_publication_staging WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert staged["projection_json"] == request.projection.canonical_json
    assert staged["projection_digest"] == request.projection.digest
    assert staged["content_digest"] == request.projection.candidate_content_digest

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
    service = _service(storage)
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
    assert successor.source == "gepa"
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
    service = _service(storage)
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
    ("update"),
    [
        {"content": "human-edited content"},
        {"trigger": "human-edited trigger"},
    ],
    ids=["content", "trigger"],
)
def test_publish_rejects_in_place_incumbent_semantic_change_after_staging(
    tmp_path: Path, update: dict[str, str]
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    service.stage(request)

    storage.update_user_playbook(incumbent.user_playbook_id, **update)
    result = service.publish(request)

    assert result.outcome == "incumbent_changed"
    assert result.successor_user_playbook_id is None
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )


def test_publication_request_rejects_projector_trigger_mutation(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)

    with pytest.raises(ValueError, match="preserve incumbent trigger"):
        _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
            projection=_projection(trigger="malicious trigger"),
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
    service = _service(storage)
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
            "candidate_content_digest": _digest("new content"),
            "embedding": embedding,
            "embedding_model_id": "test-embedding-v1",
            "expanded_terms": ["changed-expanded"],
            "lexical_document": "changed lexical document",
            "preserved_trigger": "refund trigger",
            "projector_code_digest": "a" * 64,
            "projector_id": "reflexio.search.user-playbook",
            "projector_version": "1",
            "schema_version": "offline-tuner-candidate-search-projection-v1",
        }
        canonical = _canonical(payload)
        projection = PublicationSearchProjection(
            schema_version="offline-tuner-candidate-search-projection-v1",
            canonical_json=canonical,
            digest=_digest(canonical),
            projector_id="reflexio.search.user-playbook",
            projector_version="1",
            projector_code_digest="a" * 64,
            candidate_content_digest=_digest("new content"),
            preserved_trigger="refund trigger",
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
    service = _service(storage)
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


def test_publication_claim_rejects_lease_expired_before_claim_without_mutation(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (100, job.job_id),
    )
    storage.conn.commit()
    changes_before = storage.conn.total_changes

    with (
        patch(_EPOCH_NOW_PATCH, return_value=101),
        pytest.raises(StorageError, match="lease expired"),
    ):
        _service(storage).claim(job_id=job.job_id, owner="worker-a", worker_fence=5)

    assert storage.conn.total_changes == changes_before
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbook_publication_claims"
        ).fetchone()["count"]
        == 0
    )
    assert incumbent.status is None


def test_publication_stage_rejects_lease_expired_after_claim_without_mutation(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (101, job.job_id),
    )
    storage.conn.commit()
    service = _service(storage)
    with patch(_EPOCH_NOW_PATCH, return_value=100):
        claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id, incumbent_id=incumbent.user_playbook_id, claim=claim
    )
    changes_before = storage.conn.total_changes

    with (
        patch(_EPOCH_NOW_PATCH, return_value=102),
        pytest.raises(StorageError, match="lease expired"),
    ):
        service.stage(request)

    assert storage.conn.total_changes == changes_before
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbook_publication_staging"
        ).fetchone()["count"]
        == 0
    )


def test_publication_commit_rejects_lease_expired_after_stage_without_mutation(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (101, job.job_id),
    )
    storage.conn.commit()
    service = _service(storage)
    with patch(_EPOCH_NOW_PATCH, return_value=100):
        claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
        request = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
        )
        service.stage(request)
    changes_before = storage.conn.total_changes

    with (
        patch(_EPOCH_NOW_PATCH, return_value=102),
        pytest.raises(StorageError, match="lease expired"),
    ):
        storage.commit_user_playbook_publication(request)

    assert storage.conn.total_changes == changes_before
    assert service.load_committed(job.job_id) is None
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbooks WHERE content = 'new content'"
        ).fetchone()["count"]
        == 0
    )
    claim_row = storage.conn.execute(
        "SELECT consumed FROM user_playbook_publication_claims WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert claim_row["consumed"] == 0


def test_publication_lease_is_expired_at_exact_epoch_boundary(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    _, job = _seed(storage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (100, job.job_id),
    )
    storage.conn.commit()
    changes_before = storage.conn.total_changes

    with (
        patch(_EPOCH_NOW_PATCH, return_value=100),
        pytest.raises(StorageError, match="lease expired"),
    ):
        _service(storage).claim(job_id=job.job_id, owner="worker-a", worker_fence=5)

    assert storage.conn.total_changes == changes_before


def test_committed_response_loss_retry_returns_same_successor_after_lease_expiry(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_expires_at = ? WHERE job_id = ?",
        (101, job.job_id),
    )
    storage.conn.commit()

    with patch(_EPOCH_NOW_PATCH, return_value=100):
        claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
        request = _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=claim,
        )
        first = service.publish(request)
    changes_before = storage.conn.total_changes

    with patch(_EPOCH_NOW_PATCH, return_value=102):
        retry = service.publish(request)

    assert retry == first
    assert storage.conn.total_changes == changes_before
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
    first_service = _service(storage)
    second_service = _service(_store(tmp_path))
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
    service = _service(storage)
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
    service = _service(storage)
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
    service = _service(storage)
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
    claim = PublicationClaim(job_id=job.job_id, owner="worker-a", fence=1)
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
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbook_publication_claims"
        ).fetchone()["count"]
        == 0
    )
    assert (
        storage.conn.execute(
            "SELECT COUNT(*) AS count FROM user_playbook_publication_results"
        ).fetchone()["count"]
        == 0
    )


def test_reclaimed_worker_refreshes_stage_binding_and_old_worker_is_rejected(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    old_claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    old_request = _request(
        job_id=job.job_id,
        incumbent_id=incumbent.user_playbook_id,
        claim=old_claim,
    )
    service.stage(old_request)

    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_owner = ?, lease_fence = ? WHERE job_id = ?",
        ("worker-b", 6, job.job_id),
    )
    storage.conn.commit()
    new_claim = service.claim(job_id=job.job_id, owner="worker-b", worker_fence=6)
    new_request = _request(
        job_id=job.job_id,
        incumbent_id=incumbent.user_playbook_id,
        claim=new_claim,
        worker_fence=6,
    )

    service.stage(new_request)
    staged = storage.conn.execute(
        "SELECT * FROM user_playbook_publication_staging WHERE job_id = ?",
        (job.job_id,),
    ).fetchone()
    assert staged["claim_owner"] == "worker-b"
    assert staged["worker_fence"] == 6
    assert staged["publication_fence"] == new_claim.fence

    with pytest.raises(StorageError, match="worker owner|worker fence|publication"):
        service.publish(old_request)

    result = service.publish(new_request)
    assert result.outcome == "applied"


def test_reclaimed_worker_cannot_change_immutable_staging_identity(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    old_claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    service.stage(
        _request(
            job_id=job.job_id,
            incumbent_id=incumbent.user_playbook_id,
            claim=old_claim,
        )
    )
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET lease_owner = ?, lease_fence = ? WHERE job_id = ?",
        ("worker-b", 6, job.job_id),
    )
    storage.conn.commit()
    new_claim = service.claim(job_id=job.job_id, owner="worker-b", worker_fence=6)

    with pytest.raises(StorageError, match="request identity"):
        service.stage(
            _request(
                job_id=job.job_id,
                incumbent_id=incumbent.user_playbook_id,
                claim=new_claim,
                worker_fence=6,
                request_id="changed-request",
            )
        )


def test_publication_requires_incumbent_in_frozen_subject_vector(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    frozen = _subject_epochs_json(subject_ref="subject:other")
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET metadata_json = ? WHERE job_id = ?",
        (
            _canonical(
                {
                    "publication_proof_digest": _proof().digest,
                    "publication_subject_epochs": json.loads(frozen),
                }
            ),
            job.job_id,
        ),
    )
    storage.conn.commit()
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)

    with pytest.raises(StorageError, match="incumbent governance subject"):
        service.stage(
            _request(
                job_id=job.job_id,
                incumbent_id=incumbent.user_playbook_id,
                claim=claim,
                subject_epochs_json=frozen,
            )
        )


def test_publication_rejects_request_subject_vector_different_from_job(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)

    with pytest.raises(StorageError, match="subject epochs vector"):
        service.stage(
            _request(
                job_id=job.job_id,
                incumbent_id=incumbent.user_playbook_id,
                claim=claim,
                subject_epochs_json=_subject_epochs_json(epoch=1),
            )
        )


def test_publication_rechecks_frozen_subject_vector_at_commit(tmp_path: Path) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id,
        incumbent_id=incumbent.user_playbook_id,
        claim=claim,
    )
    service.stage(request)
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET metadata_json = ? WHERE job_id = ?",
        (
            _canonical(
                {
                    "publication_proof_digest": request.decision_proof.digest,
                    "publication_subject_epochs": json.loads(
                        _subject_epochs_json(epoch=1)
                    ),
                }
            ),
            job.job_id,
        ),
    )
    storage.conn.commit()

    with pytest.raises(StorageError, match="subject epochs vector"):
        storage.commit_user_playbook_publication(request)


def test_publication_checks_erasure_barrier_for_every_frozen_subject(
    tmp_path: Path,
) -> None:
    storage = _store(tmp_path)
    incumbent, job = _seed(storage)
    frozen = _canonical(
        {
            "subjects": [
                {"epoch": 0, "ref": _subject_ref()},
                {"epoch": 2, "ref": "subject:auxiliary"},
            ]
        }
    )
    storage.conn.execute(
        "UPDATE playbook_optimization_jobs SET metadata_json = ? WHERE job_id = ?",
        (
            _canonical(
                {
                    "publication_proof_digest": _proof().digest,
                    "publication_subject_epochs": json.loads(frozen),
                }
            ),
            job.job_id,
        ),
    )
    storage.conn.commit()
    service = _service(storage)
    claim = service.claim(job_id=job.job_id, owner="worker-a", worker_fence=5)
    request = _request(
        job_id=job.job_id,
        incumbent_id=incumbent.user_playbook_id,
        claim=claim,
        subject_epochs_json=frozen,
    )
    service.stage(request)
    storage.conn.execute(
        """INSERT INTO subject_write_barriers
           (org_id, subject_ref, purge_id, status, created_at, updated_at)
           VALUES (?, 'subject:auxiliary', 'purge-aux', 'erased', 1, 1)""",
        (storage.org_id,),
    )
    storage.conn.commit()

    with pytest.raises(StorageError, match="blocked by erasure barrier"):
        service.publish(request)
