"""One open-world attempt on SQLite, from served sessions to a published successor.

Everything below the analyst is the real local runtime: the playbook is
searched and served with exposure receipts, each session is published and
judged, the evidence snapshot is read from SQLite, selected and frozen, the
rule-evidence attempt runs under its lease, and the provisional publisher
commits the successor. The analyst transport is scripted
(``GroundedTransport``) and every network call is refused, so no model is
called.

All data is synthetic.
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import (
    Interaction,
    Request,
    RetrievedLearning,
    UserPlaybook,
)
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import (
    SessionOutcomeKind,
    SetSessionOutcomeRequest,
)
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldTargetIdentity,
    OpenWorldTaskContext,
)
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    PRODUCTION_OPEN_WORLD_READ_LIMITS,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    select_open_world_evidence,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    OpenWorldEvidenceBundleStore,
    incumbent_published_at,
)
from reflexio.server.services.offline_tuner.exposure_receipts import (
    ExposureReceiptIssuer,
    receipt_project_id,
)
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    AttemptTerminalFailure,
)
from reflexio.server.services.offline_tuner.open_world.publication import (
    OpenWorldProvisionalPublisher,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    qualify_rule_evidence_method,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    RuleEvidenceAnalysisStore,
    RuleEvidenceAttemptRunner,
)
from reflexio.server.services.offline_tuner.open_world.runner import AttemptLease
from reflexio.server.services.playbook.publication import (
    incumbent_user_playbook_semantic_digest,
)
from reflexio.server.services.search_exposure import (
    SearchExposureBatch,
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.test_support.open_world_attempt_contract import GroundedTransport
from reflexio.test_support.open_world_publication_contract import ContractProjector

_OWNER = "e2e-owner"
_SESSIONS = 5
_LEASE = 3_600
_EMBEDDING = [1.0] + [0.0] * 511


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SQLiteStorage]:
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("the end-to-end attempt opened a network connection")

    monkeypatch.setattr(socket, "create_connection", refuse)
    with patch.object(SQLiteStorage, "_get_embedding", return_value=list(_EMBEDDING)):
        store = SQLiteStorage(org_id="e2e_org", db_path=str(tmp_path / "e2e.db"))
        try:
            yield store
        finally:
            store.conn.close()


def _served(storage: SQLiteStorage, query: str) -> list[UserPlaybook]:
    return storage.search_user_playbooks(
        SearchUserPlaybookRequest(
            query=query, user_id=_OWNER, search_mode=SearchMode.FTS
        )
    )


def _serve_and_judge(storage: SQLiteStorage, incumbent: UserPlaybook) -> None:
    """Five owner sessions, each served the incumbent with a receipt, then failed."""
    issuer = ExposureReceiptIssuer(project_id=receipt_project_id)
    for index in range(1, _SESSIONS + 1):
        session_id, request_id = f"e2e-session-{index}", f"e2e-request-{index}"
        (served,) = _served(storage, "zqincumbent")
        receipt = issuer.issue(
            SearchExposureBatch(
                org_id=storage.org_id,
                request_id=None,
                session_id=session_id,
                interaction_id=None,
                user_id=_OWNER,
                user_playbooks=(served,),
            )
        )[served.user_playbook_id]
        with storage.commit_scope():
            storage.add_request(
                Request(request_id=request_id, user_id=_OWNER, session_id=session_id)
            )
            storage.add_user_interactions_bulk(
                _OWNER,
                [
                    Interaction(
                        user_id=_OWNER,
                        request_id=request_id,
                        role="User",
                        content="The service is unavailable. Please fix it.",
                        embedding=list(_EMBEDDING),
                    ),
                    Interaction(
                        user_id=_OWNER,
                        request_id=request_id,
                        role="Assistant",
                        content="I retried immediately without checking the state.",
                        retrieved_learnings=[
                            RetrievedLearning(
                                kind="user_playbook",
                                learning_id=str(served.user_playbook_id),
                                exposure_receipt=receipt,
                            )
                        ],
                        embedding=list(_EMBEDDING),
                    ),
                ],
                embeddings_prepared=True,
            )
        occurred_at = int(time.time())
        recorded = storage.record_session_outcome(
            SetSessionOutcomeRequest(
                session_id=session_id,
                outcome=SessionOutcomeKind.FAILURE,
                occurred_at=occurred_at,
            ),
            created_at=occurred_at,
            expected_context=storage.get_session_outcome_context(session_id),
        )
        assert recorded.recorded, recorded


def test_an_attempt_runs_from_served_sessions_to_a_served_successor(
    storage: SQLiteStorage,
) -> None:
    storage.save_user_playbooks(
        [
            UserPlaybook(
                user_id=_OWNER,
                playbook_name="e2e-playbook",
                agent_version="v1",
                request_id="e2e-seed",
                content="Retry the call at once zqincumbent.",
                trigger="service unavailable",
            )
        ]
    )
    (seeded,) = _served(storage, "zqincumbent")
    incumbent = storage.get_user_playbook_by_id(seeded.user_playbook_id)
    assert incumbent is not None
    _serve_and_judge(storage, incumbent)

    # Freeze: the snapshot SQLite reads, the shared selection, the atomic freeze.
    frozen_at = int(time.time()) + 1
    target = OpenWorldTargetIdentity(
        tenant_id=storage.org_id,
        user_playbook_id=incumbent.user_playbook_id,
        incumbent_semantic_digest=incumbent_user_playbook_semantic_digest(
            content_digest=sha256(incumbent.content.encode()).hexdigest(),
            trigger=incumbent.trigger,
        ),
        incumbent_full_version_fingerprint=user_playbook_full_version_fingerprint(
            incumbent
        ),
        success_definition_digest=OpenWorldTaskContext().digest,
        governance_epoch_vector=storage.list_open_world_governance_epochs(),
        frozen_at=frozen_at,
    )
    snapshot = storage.load_open_world_evidence_snapshot(
        target=target, frozen_at=frozen_at, limits=PRODUCTION_OPEN_WORLD_READ_LIMITS
    )
    selection = select_open_world_evidence(
        snapshot,
        target,
        published_at=incumbent_published_at(
            storage, user_playbook_id=incumbent.user_playbook_id
        ),
    )
    assert selection.bundle.eligible, selection.bundle.reason
    bundles = OpenWorldEvidenceBundleStore(storage)
    frozen = bundles.freeze(
        selection, owner="e2e-worker", lease_seconds=_LEASE, now=frozen_at
    )
    assert frozen.claim is not None
    claim = frozen.claim

    # Attempt: the rule-evidence runner under the lease, analysts scripted.
    transport = GroundedTransport(called_at=frozen_at)
    analysis = RuleEvidenceAnalysisStore(
        storage, qualify_rule_evidence_method(transport), transport, lambda: frozen_at
    )
    analysis.reconcile_authority()
    loaded = bundles.load(claim.job_id)
    outcome = RuleEvidenceAttemptRunner(analysis, AttemptLease(claim), _LEASE).run(
        frozen=loaded,
        incumbent=incumbent,
        evidence=rehydrate_frozen_evidence(loaded, snapshot),
    )
    assert not isinstance(outcome, AttemptTerminalFailure), outcome
    assert outcome.decision.decision == "accept"
    assert len(transport.calls) == _SESSIONS + 1

    # Publication: the provisional publisher over SQLite.
    result = OpenWorldProvisionalPublisher(
        analysis_store=analysis,
        publication_store=storage,
        playbook_loader=storage,
        projector=ContractProjector(),
        clock=lambda: frozen_at,
    ).publish(job_id=claim.job_id, claim=claim)

    assert result.outcome == "provisional"
    successor_id = result.successor_user_playbook_id
    assert successor_id is not None
    assert [playbook.user_playbook_id for playbook in _served(storage, "retry")] == [
        successor_id
    ], "search serves the successor, not the superseded incumbent"
    successor = storage.get_user_playbook_by_id(successor_id)
    assert successor is not None
    assert successor.content == "Check state first; retry only if needed."
    assert (
        storage.load_user_playbook_provisional_publication_result(claim.job_id)
        == result
    )
    (lifecycle,) = storage.list_playbook_deployment_lifecycles(
        user_playbook_ids=(successor_id,)
    )
    assert (lifecycle.successor_id, lifecycle.predecessor_id) == (
        successor_id,
        incumbent.user_playbook_id,
    )
    assert lifecycle.deployment_lifecycle_id == result.deployment_lifecycle_id
    assert incumbent_published_at(storage, user_playbook_id=successor_id) == (
        lifecycle.published_at
    )
    job = storage.get_playbook_optimization_job(claim.job_id)
    assert job is not None
    assert (job.status, job.stage, job.terminal_outcome, job.successor_target_id) == (
        "completed",
        "applied",
        "applied",
        successor_id,
    )
