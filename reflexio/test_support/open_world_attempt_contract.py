# ruff: noqa: S101 -- Test-only assertions.
"""Shared contract for the open-world attempt store, part of the port contract.

:class:`~reflexio.test_support.open_world_port_contract.OpenWorldPortContract`
inherits :class:`OpenWorldAttemptContract`, so every backend binding of the port
contract runs these cases too. They drive the real rule-evidence attempt
(``RuleEvidenceAttemptRunner``) over a bundle frozen through storage, with a
scripted transport that never reaches a provider, and then call the attempt
members of ``OpenWorldStoragePort`` directly with the bindings, receipts and
artifacts that run produced.

All data is synthetic.

Invariants covered
------------------
I3 -- every fenced read and write refuses a stale fence and an expired lease
with ``OpenWorldInvocationLeaseConflictError``, the replay of an already
stored row included, and writes nothing.

I4 -- a prepare replay returns the stored row; another identity in a held
question slot conflicts; the same identity with other bytes, or another
job's, is bound; a completion is written once and is immutable; storage
recomputes the analyst-input identity from its preimage before admitting an
artifact.

I5 -- a completed invocation is reused: a restarted attempt, or one whose
process died after the provider answered, makes no new provider call.

I6 -- the attempt context is a singleton; a replay returns it and any drift
(bundle, contract, authority) raises.

I7 -- one artifact per job and kind; a replay of its exact bytes returns it,
other bytes conflict; a read rechecks the stored digest; an abstained job's
decision stays readable while every other read needs the live lease.

I8 -- a discovery memo, candidate or decision that disagrees with its
completed invocations or the frozen bundle is refused and writes nothing.

I13 -- the compaction checkpoint: the shared protocol is compare-and-swap and
advances only forward; the rolling protocol never regresses and is immutable
once complete; a record above ``ROW_BYTES`` is refused; a complete rolling
checkpoint is reusable by another job.

I17 -- a terminal stage advance accepts exactly the outcomes its arm admits
(``OPEN_WORLD_FAILED_TERMINAL_OUTCOMES``, ``OPEN_WORLD_ABSTAINED_TERMINAL_
OUTCOMES``) out of every ``OptimizationTerminalOutcome``, and a job carrying any
admitted outcome reads back.

Clearing a user deletes the analyst calls about their sessions.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, NamedTuple, Protocol, get_args

import pytest

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    OptimizationTerminalOutcome,
    PlaybookOptimizationArtifact,
    UserPlaybook,
)
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
    OpenWorldTaskContext,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    select_open_world_evidence,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    FrozenOpenWorldEvidence,
    OpenWorldEvidenceBundleStore,
)
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    PROTOCOL,
    ROW_BYTES,
    Root,
    digest,
    receipt,
)
from reflexio.server.services.offline_tuner.open_world.evidence_documents import (
    rehydrate_frozen_evidence,
)
from reflexio.server.services.offline_tuner.open_world.models import (
    ObservedStructuredCompletion,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_authority import (
    qualify_rule_evidence_method,
)
from reflexio.server.services.offline_tuner.open_world.rule_evidence_runtime import (
    RuleEvidenceAnalysisStore,
    RuleEvidenceAttemptRunner,
)
from reflexio.server.services.offline_tuner.open_world.runner import AttemptLease
from reflexio.server.services.offline_tuner.open_world_invocations import (
    CompletionRecord,
    ExpectedProvenance,
    OpenWorldInvocationBinding,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES,
    OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND,
    OPEN_WORLD_CANDIDATE_ARTIFACT_KIND,
    OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND,
    OPEN_WORLD_FAILED_TERMINAL_OUTCOMES,
    OpenWorldInvocationConflictError,
    OpenWorldInvocationIdentityBoundError,
    OpenWorldInvocationIntegrityError,
    OpenWorldInvocationLeaseConflictError,
    OpenWorldInvocationRow,
    OpenWorldQualificationAuthorityStaleError,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.error import (
    OptimizationArtifactIntegrityError,
    StorageError,
)
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)
from reflexio.test_support.open_world_evidence_contract import (
    EvidenceHarness,
    JobRow,
    _freeze,
    _frozen_target,
)
from reflexio.test_support.open_world_exposure_contract import _owned_playbook, _uid
from reflexio.test_support.open_world_rule_evidence import (
    evidence_digest,
    evidence_event,
)

_LEASE = 3_600
_SESSIONS = 5
_MEMO = OPEN_WORLD_DISCOVERY_MEMO_ARTIFACT_KIND
_CANDIDATE = OPEN_WORLD_CANDIDATE_ARTIFACT_KIND
_DECISION = OPEN_WORLD_ATTEMPT_DECISION_ARTIFACT_KIND


class AttemptRows(NamedTuple):
    """How many rows of each attempt table one job owns."""

    invocations: int
    contexts: int
    artifacts: int
    compactions: int


class AttemptHarness(EvidenceHarness, Protocol):
    """The attempt half of the port harness."""

    def attempt_rows(self, job_id: int) -> AttemptRows:
        """Count the job's invocation, context, artifact and checkpoint rows."""
        ...

    def corrupt_artifact_digest(
        self, job_id: int, artifact_kind: str, content_digest: str
    ) -> None:
        """Overwrite a stored artifact's digest directly, bypassing storage."""
        ...


# ---------------------------------------------------------------------------
# A scripted transport and one frozen attempt
# ---------------------------------------------------------------------------


class GroundedTransport:
    """Return source-bound Diagnostic and Proposal outputs; never a provider.

    Every diagnostic cites the session's first agent turn and the incumbent's
    whole clause range; the proposal rewrites that range, or abstains.
    """

    def __init__(
        self,
        *,
        called_at: int,
        abstain: bool = False,
        model_revision: str = "contract-model-1",
    ) -> None:
        self.called_at = called_at
        self.abstain = abstain
        self.transport_identity = SimpleNamespace(
            provider="contract-provider", model_revision=model_revision
        )
        self.calls: list[dict[str, Any]] = []

    def complete(self, **kwargs: Any) -> ObservedStructuredCompletion:
        self.calls.append(kwargs)
        payload = json.loads(kwargs["messages"][-1].content)
        schema = kwargs["response_format"]
        if issubclass(schema, Diagnostic):
            turn = next(m["turn"] for m in payload["messages"] if m["role"] == "agent")
            clauses = payload["incumbent"]["content_clauses"]
            value: Any = {
                "user_id": payload["user_id"],
                "session_id": payload["session_id"],
                "observations": [
                    {
                        "dimension": "execution_effectiveness",
                        "kind": "problem",
                        "observation": "The agent retried without checking state.",
                        "consequence": "The available state check was skipped.",
                        "evidence_refs": {"r1": turn},
                    }
                ],
                "rule_assessment": {
                    "attribution": "missing_required_step",
                    "target_scope": "Retry state check",
                    "counterevidence_refs": {},
                    "clause_start": clauses[0]["id"],
                    "clause_end": clauses[-1]["id"],
                    "expected_if_followed": "Retry without checking state.",
                    "reason": "The guidance omits the state check.",
                    "evidence_refs": {"r1": turn},
                },
                "missing_information": [],
            }
        else:
            assert issubclass(schema, Proposal), "unexpected wire contract"
            clauses = payload["original_clauses"]
            value = (
                {"hypothesis": "", "edits": [], "abstention_reason": "No edit."}
                if self.abstain
                else {
                    "hypothesis": "Check state before retrying.",
                    "edits": [
                        {
                            "clause_start": clauses[0]["id"],
                            "clause_end": clauses[-1]["id"],
                            "replacement": "Check state first; retry only if needed.",
                            "defect": "Retrying omits the available state check.",
                            "evidence_refs": {"r1": "REF_1"},
                        }
                    ],
                    "abstention_reason": "",
                }
            )
        return ObservedStructuredCompletion(
            value=schema.model_validate_json(json.dumps(value)),
            observed_provider="contract-provider",
            observed_model_revision=self.transport_identity.model_revision,
            called_at=self.called_at,
            latency_ms=5,
            usage_json=None,
        )


@dataclass
class Attempt:
    """One frozen attempt on the store under test, and what runs it."""

    harness: AttemptHarness
    owner: str
    incumbent: UserPlaybook
    selection: Any
    snapshot: OpenWorldEvidenceSnapshot
    claim: OptimizationJobClaim
    transport: GroundedTransport
    store: RuleEvidenceAnalysisStore
    now: int
    frozen: FrozenOpenWorldEvidence

    @property
    def storage(self) -> Any:
        return self.harness.storage

    @property
    def job_id(self) -> int:
        return self.claim.job_id

    def runner(
        self, store: RuleEvidenceAnalysisStore | None = None
    ) -> RuleEvidenceAttemptRunner:
        return RuleEvidenceAttemptRunner(
            store or self.store, AttemptLease(self.claim), _LEASE
        )

    def run(self, store: RuleEvidenceAnalysisStore | None = None) -> Any:
        return self.runner(store)._run(
            frozen=self.frozen,
            incumbent=self.incumbent,
            evidence=rehydrate_frozen_evidence(self.frozen, self.snapshot),
        )

    def read(self, kind: str) -> PlaybookOptimizationArtifact | None:
        return self.storage.read_open_world_artifact(
            job_id=self.job_id, artifact_kind=kind, fence=self.claim.fence, now=self.now
        )

    def payload(self, kind: str) -> dict[str, Any]:
        artifact = self.read(kind)
        assert artifact is not None, kind
        return json.loads(artifact.content_json)

    def write(self, kind: str, content: dict[str, Any]) -> PlaybookOptimizationArtifact:
        text = canonical_json_bytes(content).decode()
        return self.storage.write_open_world_artifact(
            job_id=self.job_id,
            artifact_kind=kind,
            content_json=text,
            content_digest=sha256(text.encode()).hexdigest(),
            fence=self.claim.fence,
            now=self.now,
        )


def _events(
    storage: Any, incumbent: UserPlaybook, subject_ref: str, *, at: int
) -> OpenWorldEvidenceSnapshot:
    """Five owned sessions, each exposing the incumbent before ``at``."""
    assert incumbent.user_id is not None
    fingerprint = user_playbook_full_version_fingerprint(incumbent)
    events = []
    for index in range(1, _SESSIONS + 1):
        event = evidence_event(
            session_id=f"session-{index}",
            outcome="failure",
            content="The service is unavailable. Check its state before retrying.",
        )
        trajectory = json.loads(json.dumps(event.trajectory))
        for request in trajectory["requests"]:
            request["request"]["user_id"] = incumbent.user_id
            for interaction in request["interactions"]:
                interaction["user_id"] = incumbent.user_id
                if interaction["role"] == "assistant":
                    interaction["content"] = (
                        "I retried immediately without checking the state."
                    )
        canonical = canonical_trajectory_bytes(trajectory)
        events.append(
            replace(
                event,
                tenant_id=storage.org_id,
                user_id=incumbent.user_id,
                user_playbook_id=incumbent.user_playbook_id,
                served_full_version_fingerprint=fingerprint,
                governance_subject_refs=(subject_ref,),
                exposed_at=at - 100,
                ingested_at=at - 100,
                trajectory=trajectory,
                trajectory_digest=sha256(canonical).hexdigest(),
                trajectory_byte_count=len(canonical),
                trajectory_token_count=count_input_tokens(canonical.decode()),
                outcome=None,
                outcome_id=None,
                outcome_revision=None,
                outcome_contract_digest=None,
            )
        )
    return OpenWorldEvidenceSnapshot(tenant_id=storage.org_id, events=tuple(events))


def _selection(
    storage: Any, incumbent: UserPlaybook, subject_ref: str, snapshot: Any, *, at: int
) -> Any:
    target = OpenWorldTargetIdentity(
        tenant_id=storage.org_id,
        user_playbook_id=incumbent.user_playbook_id,
        # The digest every synthetic event says it served.
        incumbent_semantic_digest=evidence_digest("incumbent"),
        incumbent_full_version_fingerprint=user_playbook_full_version_fingerprint(
            incumbent
        ),
        success_definition_digest=OpenWorldTaskContext().digest,
        governance_epoch_vector=((subject_ref, 0),),
        frozen_at=at,
    )
    selection = select_open_world_evidence(snapshot, target)
    assert selection.bundle.eligible
    return selection


def _attempt(
    harness: AttemptHarness,
    *,
    transport: GroundedTransport | None = None,
    store_type: type[RuleEvidenceAnalysisStore] = RuleEvidenceAnalysisStore,
) -> Attempt:
    """A frozen attempt on a fresh owned playbook, with the authority reconciled."""
    storage = harness.storage
    owner = _uid("owner")
    playbook = _owned_playbook(harness, owner)
    incumbent = storage.get_user_playbook_by_id(playbook.user_playbook_id)
    assert incumbent is not None
    subject_ref = harness.register_subject(owner)
    now = harness.now()
    snapshot = _events(storage, incumbent, subject_ref, at=now)
    selection = _selection(storage, incumbent, subject_ref, snapshot, at=now)
    frozen = OpenWorldEvidenceBundleStore(storage).freeze(
        selection, owner="worker-a", lease_seconds=_LEASE, now=now
    )
    assert frozen.claim is not None
    transport = transport or GroundedTransport(called_at=now)
    store = store_type(
        storage, qualify_rule_evidence_method(transport), transport, lambda: now
    )
    store.reconcile_authority()
    return Attempt(
        harness,
        owner,
        incumbent,
        selection,
        snapshot,
        frozen.claim,
        transport,
        store,
        now,
        OpenWorldEvidenceBundleStore(storage).load(frozen.job.job_id),
    )


def _accepted(harness: AttemptHarness) -> Attempt:
    attempt = _attempt(harness)
    assert attempt.run().decision.decision == "accept"
    return attempt


def _first_session(attempt: Attempt) -> dict[str, Any]:
    return attempt.payload(_MEMO)["payload"]["sessions"][0]


def _binding(attempt: Attempt, **changes: Any) -> OpenWorldInvocationBinding:
    prepared = _first_session(attempt)["prepared"]
    return OpenWorldInvocationBinding(**{**prepared["binding"], **changes})


def _expected(attempt: Attempt, **changes: Any) -> ExpectedProvenance:
    prepared = _first_session(attempt)["prepared"]
    return ExpectedProvenance(**{**prepared["expected"], **changes})


def _load(
    attempt: Attempt, binding: OpenWorldInvocationBinding
) -> OpenWorldInvocationRow:
    return attempt.storage.load_open_world_invocation(
        binding=binding, fence=attempt.claim.fence, now=attempt.now
    )


def _completion(row: OpenWorldInvocationRow, **changes: Any) -> CompletionRecord:
    assert row.output_json is not None and row.output_digest is not None
    assert row.observed_provider is not None
    assert row.observed_model_revision is not None
    assert row.called_at is not None and row.latency_ms is not None
    record = CompletionRecord(
        output_json=row.output_json,
        output_digest=row.output_digest,
        observed_provider=row.observed_provider,
        observed_model_revision=row.observed_model_revision,
        called_at=row.called_at,
        latency_ms=row.latency_ms,
        usage_json=row.usage_json,
    )
    return replace(record, **changes)


def _rolling(
    identity: str, *, batches: int, complete: bool, word: str
) -> dict[str, Any]:
    return {
        "identity": identity,
        "batches": batches,
        "complete": complete,
        "compaction": {
            "points": [
                {
                    "category": "goal",
                    "detail": f"synthetic {word}",
                    "source_id": "source-1",
                    "exact_quote": "synthetic quote",
                }
            ]
        },
    }


def _root(identity: str, *, calls: int) -> dict[str, Any]:
    return receipt(
        identity,
        Root(
            source_digest="1" * 64,
            plan_witness="2" * 64,
            request_contract_digest="3" * 64,
            limit=8,
            calls=calls,
        ),
        complete=False,
    )


def _checkpoint(attempt_harness: AttemptHarness) -> tuple[int, int, int]:
    """A live open-world job and its lease, for the checkpoint cases."""
    _playbook, request, now = _frozen_target(attempt_harness, attempt=_uid("job"))
    record = _freeze(attempt_harness, request, now=now)
    assert record.claim is not None
    return record.claim.job_id, record.claim.fence, now


def _forge(content: dict[str, Any], change: Callable[[dict[str, Any]], None]) -> dict:
    forged = copy.deepcopy(content)
    change(forged["payload"])
    return forged


# ---------------------------------------------------------------------------
# The forgeries I8 refuses, each with the check that must catch it
# ---------------------------------------------------------------------------


def _memo_population(payload: dict[str, Any]) -> None:
    payload["diagnostic_population_digest"] = "f" * 64


def _memo_observed_output(payload: dict[str, Any]) -> None:
    payload["sessions"][0]["observed_output"]["observations"][0]["observation"] = (
        "A different observation than the analyst returned."
    )


def _memo_identity_preimage(payload: dict[str, Any]) -> None:
    payload["sessions"][0]["prepared"]["analyst_input_identity"]["attempt"] = 2


def _memo_trajectory(payload: dict[str, Any]) -> None:
    payload["sessions"][0]["trajectory_json"] += " "


def _candidate_content(payload: dict[str, Any]) -> None:
    payload["revised_content"] = "Invented replacement guidance."


def _candidate_citations(payload: dict[str, Any]) -> None:
    payload["verified_citations"] = []


def _candidate_diagnostics(payload: dict[str, Any]) -> None:
    payload["diagnostics_artifact_digest"] = "f" * 64


def _decision_candidate(payload: dict[str, Any]) -> None:
    payload["candidate_digest"] = "f" * 64


def _decision_abstains(payload: dict[str, Any]) -> None:
    payload["decision"] = {"decision": "abstain", "reason": "no_grounded_hypothesis"}
    payload["candidate_digest"] = None


def _decision_receipt(payload: dict[str, Any]) -> None:
    payload["proposal_receipt"] = {"prepared": {}}


_FORGERIES = [
    (_MEMO, _memo_population, OpenWorldInvocationConflictError, "population digest"),
    (
        _MEMO,
        _memo_observed_output,
        OpenWorldInvocationConflictError,
        "observed output changed",
    ),
    (
        _MEMO,
        _memo_identity_preimage,
        OpenWorldInvocationConflictError,
        "provenance changed",
    ),
    (
        _MEMO,
        _memo_trajectory,
        OpenWorldInvocationIntegrityError,
        "trajectory preimage changed",
    ),
    (
        _CANDIDATE,
        _candidate_content,
        OpenWorldInvocationIntegrityError,
        "source-linked atomic global edit",
    ),
    (
        _CANDIDATE,
        _candidate_citations,
        OpenWorldInvocationIntegrityError,
        "selected-gap source citations",
    ),
    (
        _CANDIDATE,
        _candidate_diagnostics,
        OpenWorldInvocationConflictError,
        "does not bind diagnostics",
    ),
    (
        _DECISION,
        _decision_candidate,
        OpenWorldInvocationConflictError,
        "does not bind exact candidate",
    ),
    (
        _DECISION,
        _decision_abstains,
        OpenWorldInvocationIntegrityError,
        "invalid rule-evidence abstention",
    ),
    (
        _DECISION,
        _decision_receipt,
        OpenWorldInvocationIntegrityError,
        "proposal receipt belongs in exact candidate",
    ),
]


class OpenWorldAttemptContract:
    """The attempt store's behaviour, asserted identically on every backend."""

    # ------------------------------------------------------------------ I5

    def test_a_completed_attempt_is_reused_without_new_calls(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _attempt(port_harness)
        first = attempt.run()
        calls = len(attempt.transport.calls)
        artifacts = {
            kind: attempt.read(kind) for kind in (_MEMO, _CANDIDATE, _DECISION)
        }

        restarted = RuleEvidenceAnalysisStore(
            attempt.storage,
            attempt.store.method,
            attempt.transport,
            lambda: attempt.now,
        )
        second = attempt.run(restarted)

        assert first.decision.decision == "accept"
        assert calls == _SESSIONS + 1, "one diagnostic per session, one proposal"
        assert dataclasses.asdict(second) == dataclasses.asdict(first)
        assert len(attempt.transport.calls) == calls, "no provider call is repeated"
        assert {
            kind: attempt.read(kind) for kind in (_MEMO, _CANDIDATE, _DECISION)
        } == artifacts
        assert port_harness.attempt_rows(attempt.job_id) == AttemptRows(
            invocations=calls, contexts=1, artifacts=4, compactions=0
        )
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "running", "held_out_analyzed", None)
        ]

    def test_a_crash_after_the_provider_answered_reuses_the_ledger(
        self, port_harness: AttemptHarness
    ) -> None:
        class CrashStore(RuleEvidenceAnalysisStore):
            crashed = False

            def write(self, kind: str, payload: dict[str, Any], claim: Any) -> Any:
                if kind == _CANDIDATE and not CrashStore.crashed:
                    CrashStore.crashed = True
                    raise KeyboardInterrupt("process exited after the proposal")
                return super().write(kind, payload, claim)

        attempt = _attempt(port_harness, store_type=CrashStore)
        with pytest.raises(KeyboardInterrupt):
            attempt.run()
        calls = len(attempt.transport.calls)
        assert attempt.read(_CANDIDATE) is None

        restarted = RuleEvidenceAnalysisStore(
            attempt.storage,
            attempt.store.method,
            attempt.transport,
            lambda: attempt.now,
        )

        assert attempt.run(restarted).decision.decision == "accept"
        assert len(attempt.transport.calls) == calls
        assert port_harness.attempt_rows(attempt.job_id).invocations == calls

    # ------------------------------------------------------------------ I3

    @pytest.mark.parametrize(
        "operation",
        [
            "prepare_replay",
            "prepare_new",
            "complete_replay",
            "load",
            "bind_replay",
            "write_replay",
            "read",
            "checkpoint_read",
            "checkpoint_write",
        ],
    )
    @pytest.mark.parametrize("stale", ["fence", "expiry"])
    def test_every_fenced_call_checks_fence_and_lease(
        self, port_harness: AttemptHarness, operation: str, stale: str
    ) -> None:
        attempt = _accepted(port_harness)
        storage = attempt.storage
        binding = _binding(attempt)
        row = _load(attempt, binding)
        memo = attempt.read(_MEMO)
        assert memo is not None
        context = attempt.store.bind_context(attempt.frozen.bundle, attempt.claim)[0]
        identity = sha256(b"contract-compaction").hexdigest()

        def call(fence: int, now: int) -> object:
            if operation == "prepare_replay":
                return storage.prepare_open_world_invocation(
                    binding=binding, expected=_expected(attempt), fence=fence, now=now
                )
            if operation == "prepare_new":
                return storage.prepare_open_world_invocation(
                    binding=replace(binding, analyst_input_identity="e" * 64),
                    expected=_expected(attempt),
                    fence=fence,
                    now=now,
                )
            if operation == "complete_replay":
                return storage.complete_open_world_invocation(
                    binding=binding, completion=_completion(row), fence=fence, now=now
                )
            if operation == "load":
                return storage.load_open_world_invocation(
                    binding=binding, fence=fence, now=now
                )
            if operation == "bind_replay":
                return storage.bind_open_world_attempt_context(
                    job_id=attempt.job_id,
                    bundle_digest=context.bundle_digest,
                    outcome_contract_digest=context.outcome_contract_digest,
                    authority_epoch=context.qualification_authority_epoch,
                    authority_digest=context.qualification_authority_digest,
                    fence=fence,
                    now=now,
                )
            if operation == "write_replay":
                return storage.write_open_world_artifact(
                    job_id=attempt.job_id,
                    artifact_kind=_MEMO,
                    content_json=memo.content_json,
                    content_digest=memo.content_digest,
                    fence=fence,
                    now=now,
                )
            if operation == "read":
                return storage.read_open_world_artifact(
                    job_id=attempt.job_id, artifact_kind=_MEMO, fence=fence, now=now
                )
            if operation == "checkpoint_read":
                return storage.session_compaction_checkpoint(
                    job_id=attempt.job_id, identity=identity, fence=fence, now=now
                )
            return storage.session_compaction_checkpoint(
                job_id=attempt.job_id,
                identity=identity,
                fence=fence,
                now=now,
                receipt=_rolling(identity, batches=1, complete=False, word="one"),
            )

        before = port_harness.attempt_rows(attempt.job_id)
        fence, now = (
            (attempt.claim.fence + 1, attempt.now)
            if stale == "fence"
            else (attempt.claim.fence, attempt.claim.expires_at)
        )

        with pytest.raises(OpenWorldInvocationLeaseConflictError):
            call(fence, now)

        assert port_harness.attempt_rows(attempt.job_id) == before, "nothing written"
        if operation != "prepare_new":
            call(attempt.claim.fence, attempt.now)  # the live lease is admitted

    # ------------------------------------------------------------------ I4

    def test_a_prepare_replay_returns_the_stored_row(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        binding = _binding(attempt)

        replayed = attempt.storage.prepare_open_world_invocation(
            binding=binding,
            expected=_expected(attempt),
            fence=attempt.claim.fence,
            now=attempt.now,
        )

        assert replayed == _load(attempt, binding)
        assert replayed.state == "completed", "a replay never resets a completion"

    def test_other_bytes_for_a_question_are_refused(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        binding = _binding(attempt)
        before = port_harness.attempt_rows(attempt.job_id)

        with pytest.raises(OpenWorldInvocationIdentityBoundError):
            attempt.storage.prepare_open_world_invocation(
                binding=binding,
                expected=_expected(attempt, input_digest="d" * 64),
                fence=attempt.claim.fence,
                now=attempt.now,
            )
        with pytest.raises(OpenWorldInvocationConflictError) as held:
            attempt.storage.prepare_open_world_invocation(
                binding=replace(binding, analyst_input_identity="e" * 64),
                expected=_expected(attempt),
                fence=attempt.claim.fence,
                now=attempt.now,
            )

        assert not isinstance(held.value, OpenWorldInvocationIdentityBoundError), (
            "another identity in a held slot conflicts; it is not bound elsewhere"
        )
        assert port_harness.attempt_rows(attempt.job_id) == before

    def test_another_jobs_identity_is_bound_and_the_runtime_moves_on(
        self, port_harness: AttemptHarness
    ) -> None:
        first = _accepted(port_harness)
        storage = first.storage
        bound, expected = _binding(first), _expected(first)
        assert storage.advance_playbook_optimization_stage(
            first.job_id,
            first.claim.fence,
            "failed",
            terminal_outcome="infrastructure_failure",
            now=first.now,
        )
        again = OpenWorldEvidenceBundleStore(storage).freeze(
            first.selection, owner="worker-b", lease_seconds=_LEASE, now=first.now
        )
        assert again.claim is not None and again.job.job_id != first.job_id
        assert again.bundle.digest == first.frozen.bundle.digest
        second = Attempt(
            port_harness,
            first.owner,
            first.incumbent,
            first.selection,
            first.snapshot,
            again.claim,
            first.transport,
            first.store,
            first.now,
            OpenWorldEvidenceBundleStore(storage).load(again.job.job_id),
        )
        first.store.bind_context(again.bundle, again.claim)

        with pytest.raises(OpenWorldInvocationIdentityBoundError):
            storage.prepare_open_world_invocation(
                binding=replace(bound, job_id=second.job_id),
                expected=expected,
                fence=second.claim.fence,
                now=second.now,
            )

        calls = len(first.transport.calls)
        assert second.run().decision.decision == "accept"
        assert len(first.transport.calls) == calls + _SESSIONS + 1, (
            "the second job asks again under its next attempt identity"
        )

    def test_a_completion_is_written_once_and_is_immutable(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        binding = _binding(attempt)
        row = _load(attempt, binding)
        storage = attempt.storage

        def complete(completion: CompletionRecord, **changes: Any) -> Any:
            return storage.complete_open_world_invocation(
                binding=replace(binding, **changes),
                completion=completion,
                fence=attempt.claim.fence,
                now=attempt.now,
            )

        assert complete(_completion(row)) == row, "the same completion replays"
        other = '{"payload":{},"contract_type":"x","schema_version":"v1"}'
        with pytest.raises(OpenWorldInvocationConflictError, match="output bytes"):
            complete(
                _completion(
                    row,
                    output_json=other,
                    output_digest=sha256(other.encode()).hexdigest(),
                )
            )
        with pytest.raises(OpenWorldInvocationConflictError, match="provenance"):
            complete(_completion(row, called_at=(row.called_at or 0) + 1))
        with pytest.raises(OpenWorldInvocationLeaseConflictError):
            complete(_completion(row), analyst_input_identity="c" * 64)

        assert _load(attempt, binding) == row

    # ------------------------------------------------------------------ I6

    def test_the_attempt_context_is_a_singleton_and_drift_raises(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        storage = attempt.storage
        bundle = attempt.frozen.bundle
        context, authority = attempt.store.bind_context(bundle, attempt.claim)

        def bind(**changes: Any) -> Any:
            arguments = {
                "job_id": attempt.job_id,
                "bundle_digest": context.bundle_digest,
                "outcome_contract_digest": context.outcome_contract_digest,
                "authority_epoch": authority.epoch,
                "authority_digest": authority.authority_digest,
                "fence": attempt.claim.fence,
                "now": attempt.now,
            }
            return storage.bind_open_world_attempt_context(**{**arguments, **changes})

        assert bind() == context
        manifest = json.loads(context.context_json)["manifest"]
        assert sorted(
            (entry["user_id"], entry["session_id"], entry["trajectory_digest"])
            for entry in manifest
            if entry["manifest_role"] == "discovery"
        ) == sorted(
            (session.user_id, session.session_id, session.trajectory_digest)
            for session in bundle.discovery.sessions
        ), "the context binds every frozen discovery session"
        assert (
            sha256(context.context_json.encode()).hexdigest() == context.context_digest
        )
        with pytest.raises(OpenWorldInvocationConflictError, match="evidence bundle"):
            bind(bundle_digest="a" * 64)
        with pytest.raises(OpenWorldInvocationConflictError, match="outcome contract"):
            bind(outcome_contract_digest="b" * 64)
        with pytest.raises(OpenWorldQualificationAuthorityStaleError):
            bind(authority_epoch=authority.epoch + 1)

        moved = GroundedTransport(
            called_at=attempt.now, model_revision="contract-model-2"
        )
        newer = RuleEvidenceAnalysisStore(
            storage, qualify_rule_evidence_method(moved), moved, lambda: attempt.now
        ).reconcile_authority()
        assert newer.epoch == authority.epoch + 1

        with pytest.raises(OpenWorldInvocationConflictError, match="conflicts"):
            bind(authority_epoch=newer.epoch, authority_digest=newer.authority_digest)
        with pytest.raises(OpenWorldQualificationAuthorityStaleError):
            _load(attempt, _binding(attempt))
        assert port_harness.attempt_rows(attempt.job_id).contexts == 1

    # ------------------------------------------------------------------ I7

    def test_one_artifact_per_job_and_kind(self, port_harness: AttemptHarness) -> None:
        attempt = _accepted(port_harness)
        memo = attempt.read(_MEMO)
        assert memo is not None
        content = json.loads(memo.content_json)

        assert attempt.write(_MEMO, content) == memo, "an exact replay returns it"
        content["payload"]["note"] = "other bytes the checks do not read"
        with pytest.raises(OpenWorldInvocationConflictError):
            attempt.write(_MEMO, content)

        assert attempt.read(_MEMO) == memo
        assert port_harness.attempt_rows(attempt.job_id).artifacts == 4

    def test_a_read_rechecks_the_stored_digest(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        port_harness.corrupt_artifact_digest(attempt.job_id, _CANDIDATE, "0" * 64)

        with pytest.raises(OptimizationArtifactIntegrityError):
            attempt.read(_CANDIDATE)

    def test_the_decision_stays_readable_after_abstention(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _attempt(
            port_harness, transport=GroundedTransport(called_at=1, abstain=True)
        )
        result = attempt.run()

        assert result.decision.decision == "abstain"
        assert port_harness.optimization_jobs(attempt.incumbent.user_playbook_id) == [
            JobRow(attempt.job_id, "skipped", "abstained", "no_grounded_hypothesis")
        ]
        decision = attempt.payload(_DECISION)["payload"]
        assert decision["decision"]["decision"] == "abstain"
        with pytest.raises(OpenWorldInvocationLeaseConflictError):
            attempt.read(_MEMO)
        with pytest.raises(OpenWorldInvocationLeaseConflictError):
            attempt.write(_DECISION, attempt.payload(_DECISION))

    def test_a_failed_jobs_decision_is_not_readable(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        assert attempt.storage.advance_playbook_optimization_stage(
            attempt.job_id,
            attempt.claim.fence,
            "failed",
            terminal_outcome="infrastructure_failure",
            now=attempt.now,
        )

        with pytest.raises(OpenWorldInvocationLeaseConflictError):
            attempt.read(_DECISION)

    # ------------------------------------------------------------------ I8

    @pytest.mark.parametrize(
        ("kind", "change", "error", "message"),
        _FORGERIES,
        ids=[change.__name__.lstrip("_") for _kind, change, _e, _m in _FORGERIES],
    )
    def test_an_artifact_disagreeing_with_its_sources_is_refused(
        self,
        port_harness: AttemptHarness,
        kind: str,
        change: Callable[[dict[str, Any]], None],
        error: type[StorageError],
        message: str,
    ) -> None:
        attempt = _accepted(port_harness)
        stored = attempt.read(kind)
        assert stored is not None

        with pytest.raises(error, match=message):
            attempt.write(kind, _forge(json.loads(stored.content_json), change))

        assert attempt.read(kind) == stored, "the stored artifact is untouched"
        assert port_harness.attempt_rows(attempt.job_id).artifacts == 4

    # ----------------------------------------------------------------- I13

    def test_the_shared_checkpoint_is_compare_and_swap(
        self, port_harness: AttemptHarness
    ) -> None:
        job_id, fence, now = _checkpoint(port_harness)
        storage = port_harness.storage
        identity = sha256(b"contract-shared").hexdigest()

        def write(record: dict[str, Any], expected: str | None) -> Any:
            return storage.session_compaction_checkpoint(
                job_id=job_id,
                identity=identity,
                fence=fence,
                now=now,
                protocol=PROTOCOL,
                receipt=record,
                expected_digest=expected,
            )

        first = write(_root(identity, calls=1), None)
        assert (
            storage.session_compaction_checkpoint(
                job_id=job_id,
                identity=identity,
                fence=fence,
                now=now,
                protocol=PROTOCOL,
            )
            == first
        )
        with pytest.raises(OpenWorldInvocationConflictError, match="compare-and-swap"):
            write(_root(identity, calls=2), None)
        with pytest.raises(OpenWorldInvocationConflictError, match="transition"):
            write(_root(identity, calls=3), digest(first))
        with pytest.raises(OpenWorldInvocationConflictError, match="transition"):
            write(_root(identity, calls=1), digest(first))

        second = write(_root(identity, calls=2), digest(first))

        assert second["payload"]["calls"] == 2
        with pytest.raises(OpenWorldInvocationConflictError, match="compare-and-swap"):
            write(_root(identity, calls=3), digest(first))
        assert port_harness.attempt_rows(job_id).compactions == 1

    def test_the_rolling_checkpoint_never_regresses(
        self, port_harness: AttemptHarness
    ) -> None:
        job_id, fence, now = _checkpoint(port_harness)
        storage = port_harness.storage
        identity = sha256(b"contract-rolling").hexdigest()

        def write(record: dict[str, Any]) -> Any:
            return storage.session_compaction_checkpoint(
                job_id=job_id, identity=identity, fence=fence, now=now, receipt=record
            )

        def read() -> Any:
            return storage.session_compaction_checkpoint(
                job_id=job_id, identity=identity, fence=fence, now=now
            )

        assert read() is None
        one = write(_rolling(identity, batches=1, complete=False, word="one"))
        assert read() == one
        two = write(_rolling(identity, batches=2, complete=False, word="two"))
        assert read() == two
        assert write(two) == two, "an exact replay is admitted"
        with pytest.raises(OpenWorldInvocationConflictError, match="regress"):
            write(one)
        with pytest.raises(OpenWorldInvocationConflictError, match="finalization"):
            write(_rolling(identity, batches=2, complete=True, word="other"))
        done = write(_rolling(identity, batches=3, complete=True, word="three"))
        with pytest.raises(OpenWorldInvocationConflictError, match="after completion"):
            write(_rolling(identity, batches=4, complete=True, word="four"))

        assert read() == done
        assert port_harness.attempt_rows(job_id).compactions == 1

    def test_a_checkpoint_above_the_row_bound_is_refused(
        self, port_harness: AttemptHarness
    ) -> None:
        job_id, fence, now = _checkpoint(port_harness)
        identity = sha256(b"contract-bound").hexdigest()
        record = _rolling(identity, batches=1, complete=False, word="x")
        record["compaction"]["points"] = [
            {
                "category": "goal",
                # Two UTF-8 bytes per character: the bound is on bytes.
                "detail": "\u00e9" * 512,
                "source_id": f"s{index:02d}" + "x" * 77,
                "exact_quote": "q" * 240,
            }
            for index in range(24)
        ]
        assert len(canonical_json_bytes(record)) > ROW_BYTES

        with pytest.raises(OptimizationArtifactIntegrityError, match="bound"):
            port_harness.storage.session_compaction_checkpoint(
                job_id=job_id, identity=identity, fence=fence, now=now, receipt=record
            )
        with pytest.raises(OptimizationArtifactIntegrityError, match="identity"):
            port_harness.storage.session_compaction_checkpoint(
                job_id=job_id,
                identity="NOT-HEX",
                fence=fence,
                now=now,
            )

        assert port_harness.attempt_rows(job_id).compactions == 0

    def test_a_complete_rolling_checkpoint_is_reusable_by_another_job(
        self, port_harness: AttemptHarness
    ) -> None:
        first_job, first_fence, now = _checkpoint(port_harness)
        second_job, second_fence, _now = _checkpoint(port_harness)
        identity = sha256(b"contract-reuse").hexdigest()
        done = port_harness.storage.session_compaction_checkpoint(
            job_id=first_job,
            identity=identity,
            fence=first_fence,
            now=now,
            receipt=_rolling(identity, batches=1, complete=True, word="done"),
        )

        assert (
            port_harness.storage.session_compaction_checkpoint(
                job_id=second_job, identity=identity, fence=second_fence, now=now
            )
            == done
        )

    # ----------------------------------------------------------------- I17

    @pytest.mark.parametrize(
        ("stage", "admitted", "status"),
        [
            ("failed", OPEN_WORLD_FAILED_TERMINAL_OUTCOMES, "failed"),
            ("abstained", OPEN_WORLD_ABSTAINED_TERMINAL_OUTCOMES, "skipped"),
        ],
    )
    def test_a_terminal_advance_accepts_exactly_its_arms_outcomes(
        self,
        port_harness: AttemptHarness,
        stage: str,
        admitted: frozenset[str],
        status: str,
    ) -> None:
        storage = port_harness.storage
        accepted: set[str] = set()
        job_id, fence, now = _checkpoint(port_harness)
        for outcome in get_args(OptimizationTerminalOutcome):
            if not storage.advance_playbook_optimization_stage(
                job_id, fence, stage, terminal_outcome=outcome, now=now
            ):
                job = storage.get_playbook_optimization_job(job_id)
                assert job is not None and job.status == "running", outcome
                continue
            accepted.add(outcome)
            job = storage.get_playbook_optimization_job(job_id)
            assert job is not None
            assert (job.status, job.stage, job.terminal_outcome) == (
                status,
                stage,
                outcome,
            )
            job_id, fence, now = _checkpoint(port_harness)

        assert accepted == admitted

    # ------------------------------------------------------------- clearing

    def test_clearing_a_user_deletes_their_analyst_calls(
        self, port_harness: AttemptHarness
    ) -> None:
        attempt = _accepted(port_harness)
        assert port_harness.attempt_rows(attempt.job_id).invocations == _SESSIONS + 1

        attempt.storage.clear_user_data(attempt.owner)

        assert port_harness.attempt_rows(attempt.job_id).invocations == 1, (
            "the per-session diagnostics go; the session-less proposal stays"
        )


__all__ = [
    "AttemptHarness",
    "AttemptRows",
    "GroundedTransport",
    "OpenWorldAttemptContract",
]
