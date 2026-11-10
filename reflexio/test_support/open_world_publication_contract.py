# ruff: noqa: S101 -- Test-only assertions.
"""Shared contract for open-world provisional publication, part of the port contract.

:class:`~reflexio.test_support.open_world_port_contract.OpenWorldPortContract`
inherits :class:`OpenWorldPublicationContract`, so every backend binding of the
port contract runs these cases too. Each case runs the real rule-evidence
attempt to an accepted decision (``open_world_attempt_contract._accepted``) and
then publishes it through the real ``OpenWorldProvisionalPublisher``, or calls
the ``OpenWorldProvisionalPublicationStore`` members directly with the request
that publisher built.

All data is synthetic.

Invariants covered
------------------
I14 -- the provisional commit is one transaction: the successor, the
superseded incumbent, the deployment lifecycle, the lineage and optimizer
events, the result and the job's ``applied`` transition are written together,
and a commit that fails at its last write leaves none of them. An incumbent
that drifted from the frozen version -- before the claim, or between stage and
commit -- gets no publication: no successor, lifecycle or result, and the
worker's uncommitted claim and staging are cleaned. A retry under the claim
that committed returns the committed result and writes nothing; under another
claim it is refused.

I15 -- the commit rechecks the decision-proof bindings: a proof that disagrees
with the durable attempt in any one binding (job, attempt, decision, bundle,
outcome contract, candidate, population, incumbent version, attempt context,
authority, governance subjects) is refused and writes nothing; a proof that
was valid at stage is checked again at commit, so an attempt context that
changed after staging, or a governance subject erased since, refuses the
commit.

The harness hooks these cases add (:class:`PublicationHarness`) read rows
directly; no backend difference shows through them, so there is no capability
flag here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any, NamedTuple, Protocol, cast

import pytest

from reflexio.models.api_schema.domain import OptimizationJobClaim
from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.config_schema import SearchMode
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    incumbent_published_at,
)
from reflexio.server.services.offline_tuner.open_world.publication import (
    OpenWorldProvisionalPublisher,
    OpenWorldPublicationError,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldPublicationIncumbentChangedError,
    OpenWorldPublicationRetryIdentityMismatchError,
)
from reflexio.server.services.offline_tuner.search_projection import (
    CandidateSearchProjection,
    CandidateSearchProjectionRequest,
    SearchProjectorKeyIdentity,
)
from reflexio.server.services.playbook.publication import (
    DecisionProofEnvelope,
    ProvisionalPublicationRequest,
    ProvisionalPublicationResult,
    canonical_json_bytes,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.error import StorageError
from reflexio.test_support.open_world_attempt_contract import (
    Attempt,
    AttemptHarness,
    _accepted,
)
from reflexio.test_support.open_world_evidence_contract import JobRow

#: Fourteen days: a provisional lifecycle's observation deadline after publish.
OBSERVATION_SECONDS = 1_209_600
#: Both backends store 512-dimension embeddings.
_DIMENSIONS = 512
_PROJECTOR = SearchProjectorKeyIdentity(
    projector_id="contract-projector",
    projector_version="contract-projector-v1",
    code_digest=sha256(b"contract-projector").hexdigest(),
)


class PublicationRows(NamedTuple):
    """How many rows of each publication table one job owns."""

    claims: int
    consumed_claims: int
    staging: int
    results: int
    lifecycles: int
    publication_events: int


class LifecycleRow(NamedTuple):
    """One stored deployment lifecycle, as the harness reads it."""

    state: str
    predecessor_id: int
    successor_id: int
    published_at: int
    observation_deadline: int
    terminal_reason: str | None
    terminal_at: int | None
    aggregation_obligation: str


class PublicationHarness(AttemptHarness, Protocol):
    """The publication half of the port harness."""

    def publication_rows(self, job_id: int) -> PublicationRows:
        """Count the job's claim, staging, result, lifecycle and event rows.

        ``publication_events`` counts ``provisional_publication`` events.
        """
        ...

    def lifecycle(self, job_id: int) -> LifecycleRow | None:
        """The job's deployment lifecycle row, if any."""
        ...

    def lineage_ops(self, request_id: str) -> list[tuple[str, str]]:
        """``(entity_id, op)`` of every lineage event with ``request_id``, sorted."""
        ...

    def playbook_ids(self, user_id: str) -> list[int]:
        """Every user playbook ``user_id`` owns, whatever its status, ascending."""
        ...

    def drift_playbook(self, user_playbook_id: int, *, column: str) -> None:
        """Append to a stored playbook's text ``column``, bypassing storage.

        ``column`` is ``"content"`` or ``"rationale"``.
        """
        ...

    def fail_result_writes(self) -> AbstractContextManager[None]:
        """While open, inserting a publication result raises inside the store."""
        ...

    def rewrite_attempt_context(self, job_id: int) -> None:
        """Add a key to the job's stored attempt context and re-digest it.

        Bypasses storage. The context stays self-consistent (its digest names
        its bytes) and keeps its manifest and authority, so only a check that
        binds the context digest itself can notice.
        """
        ...


@dataclass
class ContractProjector:
    """A deterministic search projector: no embedding provider is called."""

    key_identity: SearchProjectorKeyIdentity = _PROJECTOR
    calls: list[CandidateSearchProjectionRequest] = field(default_factory=list)

    def build(
        self, request: CandidateSearchProjectionRequest
    ) -> CandidateSearchProjection:
        self.calls.append(request)
        return CandidateSearchProjection(
            projector_id=self.key_identity.projector_id,
            projector_version=self.key_identity.projector_version,
            projector_code_digest=self.key_identity.code_digest,
            candidate_content_digest=request.candidate_content_digest,
            preserved_trigger=request.preserved_trigger,
            embedding_model_id="contract-embedding-v1",
            embedding=("1",) + ("0",) * (_DIMENSIONS - 1),
            expanded_terms=("state check", "safe retry"),
            lexical_document=(
                f"{request.preserved_trigger or ''} {request.candidate_content}".strip()
            ),
        )


class StopBeforeError(Exception):
    """Raised by :class:`RecordingStore` in place of the call it stops at."""


class RecordingStore:
    """The store under test, recording each request the publisher builds.

    ``stop_at`` (``"stage"`` or ``"commit"``) raises :class:`StopBeforeError` in place
    of that call, leaving the publication exactly as far as the calls before it
    took it. Every other member is the store's own.
    """

    def __init__(
        self,
        storage: Any,
        stop_at: str | None = None,
        before_commit: Callable[[], None] | None = None,
    ) -> None:
        self._storage = storage
        self.stop_at = stop_at
        self.before_commit = before_commit
        self.request: ProvisionalPublicationRequest | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._storage, name)

    def stage_user_playbook_provisional_publication(
        self, request: ProvisionalPublicationRequest
    ) -> bool:
        self.request = request
        if self.stop_at == "stage":
            raise StopBeforeError("stage")
        return self._storage.stage_user_playbook_provisional_publication(request)

    def commit_user_playbook_provisional_publication(
        self, request: ProvisionalPublicationRequest
    ) -> ProvisionalPublicationResult:
        self.request = request
        if self.stop_at == "commit":
            raise StopBeforeError("commit")
        if self.before_commit is not None:
            self.before_commit()
        return self._storage.commit_user_playbook_provisional_publication(request)


def publisher(
    attempt: Attempt, *, store: Any | None = None
) -> OpenWorldProvisionalPublisher:
    """The production publisher over ``attempt``'s store, on the attempt's clock."""
    return OpenWorldProvisionalPublisher(
        analysis_store=attempt.store,
        publication_store=cast(Any, store or attempt.storage),
        playbook_loader=attempt.storage,
        projector=ContractProjector(),
        clock=lambda: attempt.now,
    )


def publish(attempt: Attempt) -> ProvisionalPublicationResult:
    return publisher(attempt).publish(job_id=attempt.job_id, claim=attempt.claim)


def request_until(attempt: Attempt, stop_at: str) -> ProvisionalPublicationRequest:
    """Run the publisher up to ``stop_at`` and return the request it built.

    ``"stage"``: claimed, nothing staged. ``"commit"``: claimed and staged.
    """
    store = RecordingStore(attempt.storage, stop_at=stop_at)
    with pytest.raises(StopBeforeError):
        publisher(attempt, store=store).publish(
            job_id=attempt.job_id, claim=attempt.claim
        )
    assert store.request is not None
    return store.request


def _no_publication(harness: PublicationHarness, attempt: Attempt) -> None:
    """Nothing of a publication exists for ``attempt``: no successor or result."""
    rows = harness.publication_rows(attempt.job_id)
    assert (rows.results, rows.lifecycles, rows.publication_events) == (0, 0, 0)
    assert rows.consumed_claims == 0
    assert harness.playbook_ids(attempt.owner) == [attempt.incumbent.user_playbook_id]
    current = attempt.storage.get_user_playbook_by_id(
        attempt.incumbent.user_playbook_id
    )
    assert current is not None and current.status is None
    assert harness.lineage_ops(f"open_world_provisional:{attempt.job_id}") == []
    assert (
        attempt.storage.list_playbook_deployment_lifecycles(
            user_playbook_ids=(attempt.incumbent.user_playbook_id,)
        )
        == ()
    )


def _forged_proof(
    request: ProvisionalPublicationRequest, change: Callable[[dict[str, Any]], None]
) -> ProvisionalPublicationRequest:
    """``request`` with a self-consistent proof whose payload ``change`` edits."""
    payload = json.loads(request.decision_proof.canonical_json)
    change(payload)
    canonical = canonical_json_bytes(payload).decode("utf-8")
    return replace(
        request,
        decision_proof=DecisionProofEnvelope(
            optimizer_kind="offline_tuner_open_world",
            schema_version=request.decision_proof.schema_version,
            canonical_json=canonical,
            digest=sha256(canonical.encode("utf-8")).hexdigest(),
            decision="apply",
        ),
    )


def _set(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def change(payload: dict[str, Any]) -> None:
        assert key in payload, key
        payload[key] = value

    return change


def _bump(key: str) -> Callable[[dict[str, Any]], None]:
    def change(payload: dict[str, Any]) -> None:
        payload[key] = payload[key] + 1

    return change


def _authority_epoch(payload: dict[str, Any]) -> None:
    payload["qualification_authority"]["epoch"] += 1


def _governance_epoch(payload: dict[str, Any]) -> None:
    payload["governance_subjects"][0]["epoch"] += 1


def _extra_subject(payload: dict[str, Any]) -> None:
    payload["governance_subjects"].append({"ref": "subref_v1_" + "f" * 32, "epoch": 0})


_OTHER = "f" * 64

#: Each decision-proof binding the commit rechecks, forged alone.
_PROOF_FORGERIES = [
    ("job_id", _bump("job_id")),
    ("attempt_key", _set("attempt_key", "another-attempt")),
    ("attempt_decision_digest", _set("attempt_decision_digest", _OTHER)),
    ("attempt_decision_json", _set("attempt_decision_json", "{}")),
    ("evidence_bundle_digest", _set("evidence_bundle_digest", _OTHER)),
    ("outcome_contract_digest", _set("outcome_contract_digest", _OTHER)),
    ("candidate_digest", _set("candidate_digest", _OTHER)),
    ("candidate_content_digest", _set("candidate_content_digest", _OTHER)),
    ("diagnostic_population_digest", _set("diagnostic_population_digest", _OTHER)),
    (
        "incumbent_full_version_fingerprint",
        _set("incumbent_full_version_fingerprint", _OTHER),
    ),
    ("attempt_context_digest", _set("attempt_context_digest", _OTHER)),
    ("qualification_authority", _authority_epoch),
    ("governance_subject_epoch", _governance_epoch),
    ("governance_subject_added", _extra_subject),
]


class OpenWorldPublicationContract:
    """Provisional publication's behaviour, asserted identically on every backend."""

    # ----------------------------------------------------------------- I14

    def test_a_provisional_commit_writes_every_row_together(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        storage = attempt.storage
        incumbent_id = attempt.incumbent.user_playbook_id
        assert (
            storage.load_user_playbook_provisional_publication_result(attempt.job_id)
            is None
        )
        candidate = attempt.payload("open_world_candidate")["payload"]

        result = publish(attempt)

        assert result.outcome == "provisional"
        successor_id = result.successor_user_playbook_id
        assert successor_id is not None
        successor = storage.get_user_playbook_by_id(successor_id)
        assert successor is not None
        assert successor.status is None
        assert successor.content == candidate["revised_content"]
        assert (successor.user_id, successor.trigger, successor.playbook_name) == (
            attempt.incumbent.user_id,
            attempt.incumbent.trigger,
            attempt.incumbent.playbook_name,
        )
        assert successor.governance_subject_ref == (
            attempt.incumbent.governance_subject_ref
        )
        incumbent = storage.get_user_playbook_by_id(
            incumbent_id, include_tombstones=True
        )
        assert incumbent is not None
        assert incumbent.status is not None
        assert incumbent.status.value == "superseded"
        assert incumbent.superseded_by == successor_id
        assert port_harness.optimization_jobs(incumbent_id) == [
            JobRow(attempt.job_id, "completed", "applied", "applied")
        ]
        assert port_harness.publication_rows(attempt.job_id) == PublicationRows(
            claims=1,
            consumed_claims=1,
            staging=0,
            results=1,
            lifecycles=1,
            publication_events=1,
        )
        lifecycle = port_harness.lifecycle(attempt.job_id)
        assert lifecycle is not None
        assert lifecycle == LifecycleRow(
            state="confirmed",
            predecessor_id=incumbent_id,
            successor_id=successor_id,
            published_at=lifecycle.published_at,
            observation_deadline=lifecycle.published_at + OBSERVATION_SECONDS,
            terminal_reason="published_unconditional",
            terminal_at=lifecycle.published_at,
            aggregation_obligation="not_required",
        )
        assert result.observation_deadline == lifecycle.observation_deadline
        assert result.deployment_lifecycle_id is not None
        assert port_harness.lineage_ops(
            f"open_world_provisional:{attempt.job_id}"
        ) == sorted(
            [(str(incumbent_id), "status_change"), (str(successor_id), "revise")]
        )
        assert (
            storage.load_user_playbook_provisional_publication_result(attempt.job_id)
            == result
        )
        listed = storage.list_playbook_deployment_lifecycles(
            user_playbook_ids=(successor_id,)
        )
        assert [(row.successor_id, row.published_at) for row in listed] == [
            (successor_id, lifecycle.published_at)
        ]
        assert incumbent_published_at(storage, user_playbook_id=successor_id) == (
            lifecycle.published_at
        )
        assert incumbent_published_at(storage, user_playbook_id=incumbent_id) is None

    def test_the_successor_is_what_search_serves(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)

        result = publish(attempt)

        served = attempt.storage.search_user_playbooks(
            SearchUserPlaybookRequest(
                query="retry",
                user_id=attempt.owner,
                search_mode=SearchMode.FTS,
            )
        )
        assert [item.user_playbook_id for item in served] == [
            result.successor_user_playbook_id
        ]

    def test_a_commit_that_fails_at_its_last_write_writes_nothing(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        fingerprint = user_playbook_full_version_fingerprint(attempt.incumbent)

        with port_harness.fail_result_writes(), pytest.raises(Exception):  # noqa: B017
            publish(attempt)

        _no_publication(port_harness, attempt)
        current = attempt.storage.get_user_playbook_by_id(
            attempt.incumbent.user_playbook_id
        )
        assert current is not None
        assert user_playbook_full_version_fingerprint(current) == fingerprint
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "running", "publishing", None)
        ]
        assert port_harness.publication_rows(attempt.job_id).staging == 1, (
            "the stage committed on its own; the failed commit left it as it was"
        )

        result = publish(attempt)

        assert result.outcome == "provisional"
        assert port_harness.publication_rows(attempt.job_id).lifecycles == 1

    def test_drift_before_the_claim_writes_nothing(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        port_harness.drift_playbook(
            attempt.incumbent.user_playbook_id, column="content"
        )

        with pytest.raises(
            OpenWorldPublicationError,
            match="current incumbent does not match the frozen full version",
        ):
            publish(attempt)

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id) == PublicationRows(
            0, 0, 0, 0, 0, 0
        )
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "running", "held_out_analyzed", None)
        ]

    def test_a_claim_on_a_drifted_incumbent_writes_nothing(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        request = request_until(attempt, "stage")
        cleaned = attempt.storage.cleanup_user_playbook_provisional_stale_attempt(
            job_id=attempt.job_id,
            owner=attempt.claim.owner,
            worker_fence=attempt.claim.fence,
        )
        assert cleaned is True
        port_harness.drift_playbook(
            attempt.incumbent.user_playbook_id, column="rationale"
        )

        with pytest.raises(
            OpenWorldPublicationIncumbentChangedError,
            match="provisional publication incumbent fingerprint changed",
        ):
            attempt.storage.claim_user_playbook_provisional_publication(
                job_id=attempt.job_id,
                owner=attempt.claim.owner,
                worker_fence=attempt.claim.fence,
                incumbent_user_playbook_id=request.incumbent_user_playbook_id,
                incumbent_full_version_fingerprint=(
                    request.incumbent_full_version_fingerprint
                ),
                incumbent_snapshot_json=request.incumbent_snapshot_json,
            )

        assert port_harness.publication_rows(attempt.job_id) == PublicationRows(
            0, 0, 0, 0, 0, 0
        )
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "running", "held_out_analyzed", None)
        ]

    def test_drift_between_stage_and_commit_writes_nothing(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        request = request_until(attempt, "commit")
        assert port_harness.publication_rows(attempt.job_id).staging == 1
        # A field only the full version binds: the decision's authorization,
        # which rereads the incumbent's content, still passes.
        port_harness.drift_playbook(
            attempt.incumbent.user_playbook_id, column="rationale"
        )

        with pytest.raises(
            OpenWorldPublicationIncumbentChangedError,
            match="provisional publication incumbent fingerprint changed",
        ):
            attempt.storage.commit_user_playbook_provisional_publication(request)

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id) == PublicationRows(
            0, 0, 0, 0, 0, 0
        ), "the worker's own claim and staging are cleaned"
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "running", "held_out_analyzed", None)
        ]

    def test_content_drift_between_stage_and_commit_publishes_nothing(
        self, port_harness: PublicationHarness
    ) -> None:
        """Content drift fails the commit's authorization; the publisher cleans up."""
        attempt = _accepted(port_harness)
        store = RecordingStore(
            attempt.storage,
            before_commit=lambda: port_harness.drift_playbook(
                attempt.incumbent.user_playbook_id, column="content"
            ),
        )

        with pytest.raises(
            OpenWorldPublicationError,
            match="current incumbent does not match the frozen full version",
        ):
            publisher(attempt, store=store).publish(
                job_id=attempt.job_id, claim=attempt.claim
            )

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id) == PublicationRows(
            0, 0, 0, 0, 0, 0
        )

    def test_a_same_claim_retry_returns_the_committed_result(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        storage = attempt.storage
        assert (
            storage.load_open_world_provisional_publication_retry(
                job_id=attempt.job_id, claim=attempt.claim
            )
            is None
        )
        request = request_until(attempt, "commit")
        first = storage.commit_user_playbook_provisional_publication(request)
        rows = port_harness.publication_rows(attempt.job_id)
        playbooks = port_harness.playbook_ids(attempt.owner)
        lineage = port_harness.lineage_ops(f"open_world_provisional:{attempt.job_id}")

        assert storage.commit_user_playbook_provisional_publication(request) == first
        assert storage.stage_user_playbook_provisional_publication(request) is True
        assert (
            storage.load_open_world_provisional_publication_retry(
                job_id=attempt.job_id, claim=attempt.claim
            )
            == first
        )
        assert publish(attempt) == first, "a lost response is recovered, not redone"

        assert port_harness.publication_rows(attempt.job_id) == rows
        assert port_harness.playbook_ids(attempt.owner) == playbooks
        assert (
            port_harness.lineage_ops(f"open_world_provisional:{attempt.job_id}")
            == lineage
        )

    @pytest.mark.parametrize(
        ("owner", "fence_step"),
        [("worker-b", 0), (None, 1), ("worker-b", 1)],
        ids=["owner", "fence", "owner-and-fence"],
    )
    def test_a_retry_under_another_claim_is_refused(
        self, port_harness: PublicationHarness, owner: str | None, fence_step: int
    ) -> None:
        attempt = _accepted(port_harness)
        publish(attempt)
        other = OptimizationJobClaim(
            job_id=attempt.job_id,
            owner=owner or attempt.claim.owner,
            fence=attempt.claim.fence + fence_step,
            expires_at=attempt.claim.expires_at,
        )

        with pytest.raises(
            OpenWorldPublicationRetryIdentityMismatchError,
            match="committed provisional retry identity changed",
        ):
            attempt.storage.load_open_world_provisional_publication_retry(
                job_id=attempt.job_id, claim=other
            )
        with pytest.raises(
            OpenWorldPublicationError, match="committed provisional retry identity"
        ):
            publisher(attempt).publish(job_id=attempt.job_id, claim=other)

    def test_a_replay_with_other_bytes_is_refused(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        request = request_until(attempt, "commit")
        attempt.storage.commit_user_playbook_provisional_publication(request)
        rows = port_harness.publication_rows(attempt.job_id)

        with pytest.raises(
            StorageError, match="committed provisional publication identity changed"
        ):
            attempt.storage.commit_user_playbook_provisional_publication(
                replace(request, candidate_digest=_OTHER)
            )

        assert port_harness.publication_rows(attempt.job_id) == rows

    # ----------------------------------------------------------------- I15

    @pytest.mark.parametrize(
        "change",
        [change for _name, change in _PROOF_FORGERIES],
        ids=[name for name, _change in _PROOF_FORGERIES],
    )
    def test_the_commit_refuses_a_proof_the_attempt_does_not_imply(
        self,
        port_harness: PublicationHarness,
        change: Callable[[dict[str, Any]], None],
    ) -> None:
        attempt = _accepted(port_harness)
        request = request_until(attempt, "stage")
        forged = _forged_proof(request, change)

        with pytest.raises(
            StorageError,
            match="provisional publication proof does not match durable authority",
        ):
            attempt.storage.commit_user_playbook_provisional_publication(forged)

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id).staging == 0
        assert (
            attempt.storage.commit_user_playbook_provisional_publication(
                request
            ).outcome
            == "provisional"
        ), "the attempt's own proof still publishes"

    def test_the_commit_rechecks_the_proof_after_staging(
        self, port_harness: PublicationHarness
    ) -> None:
        """A staged proof the durable attempt no longer implies cannot commit.

        The attempt context changes after the stage accepted the proof; every
        other binding still holds, so the commit's own proof check refuses.
        """
        attempt = _accepted(port_harness)
        request = request_until(attempt, "commit")
        port_harness.rewrite_attempt_context(attempt.job_id)

        with pytest.raises(
            StorageError,
            match="provisional publication proof does not match durable authority",
        ):
            attempt.storage.commit_user_playbook_provisional_publication(request)

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id).staging == 1

    def test_a_subject_erased_after_staging_refuses_the_commit(
        self, port_harness: PublicationHarness
    ) -> None:
        attempt = _accepted(port_harness)
        request = request_until(attempt, "commit")
        port_harness.block_subject(attempt.owner)

        with pytest.raises(
            StorageError, match="provisional publication governance subject is erased"
        ):
            attempt.storage.commit_user_playbook_provisional_publication(request)

        _no_publication(port_harness, attempt)
        assert port_harness.publication_rows(attempt.job_id).staging == 1


__all__ = [
    "OBSERVATION_SECONDS",
    "ContractProjector",
    "LifecycleRow",
    "OpenWorldPublicationContract",
    "PublicationHarness",
    "PublicationRows",
    "RecordingStore",
    "StopBeforeError",
    "publish",
    "publisher",
    "request_until",
]
