# ruff: noqa: S101,ANN001,ANN201,ANN202,ARG001,ARG002 -- Test doubles: assertions and duck-typed fakes of the storage port.
"""Shared synthetic fixtures for the open-world rule-evidence tests.

One definition, consumed by the OSS unit tests and by hosts that run the same
runtime against their own storage. It ships in the package (like
``owned_window_recovery``) because host test trees cannot import another
repository's ``tests`` package.

Contents:

* ``evidence_event`` / ``frozen_evidence``: a five-session frozen evidence
  bundle and the snapshot it was selected from.
* ``source``: the frozen native source, one diagnostic and one exact-edit
  proposal (a pytest fixture).
* ``MemoryStore`` / ``runtime``: an in-memory attempt store and a scripted
  transport around ``RuleEvidenceAttemptRunner`` (``runtime`` is a fixture that
  depends on ``source``).
* ``no_network`` (autouse where imported) and ``MemoryCheckpoints``: the shared
  compaction tests' network denial and transactional checkpoint port.

All data is synthetic.
"""

from __future__ import annotations

import dataclasses
import socket
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from threading import RLock
from types import SimpleNamespace
from typing import Any, cast

import pytest

from reflexio.models.api_schema.domain import (
    OptimizationJobClaim,
    PlaybookOptimizationArtifact,
    PlaybookOptimizationJob,
    UserPlaybook,
)
from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    OpenWorldEvidenceEvent,
    OpenWorldEvidenceSnapshot,
    OpenWorldTargetIdentity,
    OpenWorldTaskContext,
)
from reflexio.server.services.offline_tuner.evidence.open_world_selection import (
    select_open_world_evidence,
)
from reflexio.server.services.offline_tuner.evidence.open_world_store import (
    FrozenOpenWorldEvidence,
)
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    PROTOCOL,
    SharedReceipt,
    digest,
    validate_advance,
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
    RuleEvidenceStorage,
)
from reflexio.server.services.offline_tuner.open_world.runner import AttemptLease
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldAttemptContextRow,
    OpenWorldInvocationRow,
    OpenWorldQualificationAuthorityRow,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedSlotAssessment as RuleAssessment,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    CounterTurns,
    SupportTurns,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    SlotObservation as Observation,
)
from reflexio.server.services.search_exposure import (
    user_playbook_full_version_fingerprint,
)
from reflexio.server.services.storage.session_outcome_identity import (
    canonical_trajectory_bytes,
)


def evidence_digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def evidence_event(
    *, session_id: str, outcome: str, content: str, count_drift: int = 0
) -> OpenWorldEvidenceEvent:
    request_id = f"request-{session_id}"
    trajectory = {
        "session_id": session_id,
        "requests": [
            {
                "request": {
                    "request_id": request_id,
                    "user_id": "user-1",
                    "created_at": "2026-08-17T00:00:00+00:00",
                    "source": "test",
                    "agent_version": "agent-v1",
                    "session_id": session_id,
                    "evaluation_only": False,
                    "retrieval_experiment_id": None,
                    "retrieval_experiment_arm": None,
                },
                "interactions": [
                    {
                        "interaction_id": int(session_id.rsplit("-", 1)[1]),
                        "user_id": "user-1",
                        "request_id": request_id,
                        "created_at": "2026-08-17T00:00:01+00:00",
                        "content": content,
                        "role": "user",
                        "token_count": 1,
                        "user_action": "",
                        "user_action_description": "",
                        "interacted_image_url": "",
                        "image_encoding": "",
                        "shadow_content": "",
                        "expert_content": "",
                        "tools_used": [],
                        "citations": [],
                        "retrieved_learnings": [],
                    },
                    # The agent's reply. Discovery requires an agent turn -- a
                    # session that only records what the user asked carries no
                    # agent behaviour to contrast -- so a user-only fixture
                    # would leave discovery empty and every consumer of this
                    # builder asserting against an abstained bundle.
                    {
                        "interaction_id": (int(session_id.rsplit("-", 1)[1]) + 1_000),
                        "user_id": "user-1",
                        "request_id": request_id,
                        "created_at": "2026-08-17T00:00:02+00:00",
                        # Deliberately shares NO substring with the user turn.
                        # `_resolve_by_exact_quote` matches by containment, so
                        # an agent turn quoting the user's text back would make
                        # every user quote resolve two ways and raise
                        # `ambiguous_quote` -- turning a citation-repair test
                        # into an ambiguity test without saying so.
                        "content": f"agent resolved case {session_id.rsplit('-', 1)[1]}",
                        "role": "assistant",
                        "token_count": 1,
                        "user_action": "",
                        "user_action_description": "",
                        "interacted_image_url": "",
                        "image_encoding": "",
                        "shadow_content": "",
                        "expert_content": "",
                        "tools_used": [],
                        "citations": [],
                        "retrieved_learnings": [],
                    },
                ],
            }
        ],
    }
    trajectory_bytes = canonical_trajectory_bytes(trajectory)
    return OpenWorldEvidenceEvent(
        exposure_event_id=f"exposure-{session_id}",
        tenant_id="tenant-1",
        user_id="user-1",
        session_id=session_id,
        user_playbook_id=7,
        served_semantic_digest=evidence_digest("incumbent"),
        served_full_version_fingerprint="a" * 64,
        request_id=request_id,
        correlation_reason=None,
        exposed_at=100,
        ingested_at=100,
        trajectory=cast(Any, trajectory),
        trajectory_digest=sha256(trajectory_bytes).hexdigest(),
        trajectory_digest_matches=True,
        # ``count_drift`` shifts BOTH stored counts away from what recomputing
        # them over the very same bytes would yield, while leaving the digest
        # -- and therefore the bytes -- untouched. That is the only shape a
        # tokenizer version bump can take, and it is applied here rather than
        # to the frozen snapshot so the manifest freezes the same drifted
        # counts the event carries.
        trajectory_byte_count=len(trajectory_bytes) + count_drift,
        trajectory_token_count=(
            count_input_tokens(trajectory_bytes.decode()) + count_drift
        ),
        outcome=outcome,  # type: ignore[arg-type]
        outcome_id=f"outcome-{session_id}",
        outcome_revision=1,
        outcome_contract_digest=evidence_digest("contract"),
        governance_subject_refs=("subject-1",),
        governance_valid=True,
    )


def frozen_evidence(
    *,
    duplicate_first_exposure: bool = False,
    count_drift: int = 0,
    extra_events: tuple[OpenWorldEvidenceEvent, ...] = (),
) -> tuple[FrozenOpenWorldEvidence, OpenWorldEvidenceSnapshot]:
    target = OpenWorldTargetIdentity(
        tenant_id="tenant-1",
        user_playbook_id=7,
        incumbent_semantic_digest=evidence_digest("incumbent"),
        incumbent_full_version_fingerprint="a" * 64,
        success_definition_digest=OpenWorldTaskContext().digest,
        governance_epoch_vector=(("subject-1", 0),),
        frozen_at=200,
    )
    events = tuple(
        evidence_event(
            session_id=f"session-{index}",
            outcome=outcome,
            content=f"quote {index}",
            count_drift=count_drift,
        )
        for index, outcome in enumerate(
            ("failure", "failure", "failure", "success", "success"), start=1
        )
    )
    if duplicate_first_exposure:
        events = (*events, replace(events[0], exposure_event_id="exposure-session-1-b"))
    # Appended AFTER the five-session eligibility floor, so anything a caller
    # adds is attributable to the added session rather than to eligibility.
    events = (*events, *extra_events)
    snapshot = OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=events)
    return (
        FrozenOpenWorldEvidence(
            job=None,  # type: ignore[arg-type]
            artifact=None,  # type: ignore[arg-type]
            bundle=select_open_world_evidence(snapshot, target).bundle,
        ),
        snapshot,
    )


@pytest.fixture
def source():
    incumbent = UserPlaybook(
        user_playbook_id=7,
        user_id="user-1",
        agent_version="v1",
        request_id="request-session-1",
        content="Send once. Report queued as delivered. Keep the recipient.",
        trigger="Sending a report",
    )
    version = user_playbook_full_version_fingerprint(incumbent)
    original, snapshot = frozen_evidence()
    event = replace(
        snapshot.events[0],
        served_full_version_fingerprint=version,
        outcome=None,
        outcome_id=None,
        outcome_revision=None,
        outcome_contract_digest=None,
    )
    snapshot = OpenWorldEvidenceSnapshot(tenant_id="tenant-1", events=(event,))
    target = replace(original.bundle.target, incumbent_full_version_fingerprint=version)
    bundle = select_open_world_evidence(snapshot, target).bundle
    rehydrated = rehydrate_frozen_evidence(replace(original, bundle=bundle), snapshot)
    documents = tuple(rehydrated.documents_by_session.values())
    turn = next(
        index
        for index, span in enumerate(documents[0].spans)
        if span.speaker == "agent"
    )
    reference = SupportTurns(r1=turn)
    diagnostic = Diagnostic(
        session_id="session-1",
        user_id="user-1",
        observations=[
            Observation(
                dimension="evidence_grounding",
                kind="problem",
                observation="Reported a stronger status than the evidence supports",
                consequence="User receives an unsupported delivery claim",
                evidence_refs=reference,
            )
        ],
        rule_assessment=RuleAssessment(
            attribution="instruction_conflicts_requirement",
            target_scope="Delivery-state reporting",
            counterevidence_refs=CounterTurns.model_validate({}),
            clause_start="CLAUSE_2",
            clause_end="CLAUSE_2",
            expected_if_followed="Report queue acceptance as delivery",
            reason="Literal compliance causes the unsupported claim",
            evidence_refs=reference,
        ),
        missing_information=[],
    )
    proposal = Proposal.model_validate(
        {
            "hypothesis": "Keep the report aligned with the service status",
            "edits": [
                {
                    "clause_start": "CLAUSE_2",
                    "clause_end": "CLAUSE_2",
                    "replacement": "Report queued as queued.",
                    "defect": "Queue acceptance does not establish delivery",
                    "evidence_refs": {"r1": f"REF_{turn + 1}"},
                }
            ],
            "abstention_reason": "",
        }
    )
    return {
        "bundle": bundle,
        "incumbent": incumbent,
        "documents": documents,
        "trajectory_json_by_session": rehydrated.trajectory_json_by_session,
        "diagnostics": [diagnostic],
        "proposal": proposal,
    }


class MemoryStore:
    org_id = "tenant-1"

    def __init__(self, method, incumbent):
        self.rows = {}
        self.artifacts = {}
        self.incumbent = incumbent
        self.crash_on_candidate = False
        self.job = SimpleNamespace(
            job_id=1,
            status="running",
            stage="evidence_frozen",
            lease_owner="worker",
            lease_fence=1,
            lease_expires_at=4000,
            terminal_outcome=None,
            metadata_json="{}",
        )
        self.authority = OpenWorldQualificationAuthorityRow(
            epoch=1,
            state="active",
            authority_digest=method.digest,
            discovery_component_identity_digest=method.diagnostic.digest,
            discovery_qualification_suite_digest=method.suite_digest,
            discovery_qualification_result_digest=method.result_digest,
            held_out_component_identity_digest=None,
            held_out_qualification_suite_digest=None,
            held_out_qualification_result_digest=None,
            candidate_generator_identity_digest=method.proposal.digest,
            candidate_generator_authorization_digest=method.proposal_authorization_digest,
            unavailable_reason=None,
            updated_at=200,
            method_authority_json=method.canonical_json,
        )

    def load_open_world_qualification_authority(self):
        return self.authority

    def bind_open_world_attempt_context(self, **kwargs):
        text = canonical_json_bytes(kwargs).decode()
        return OpenWorldAttemptContextRow(
            job_id=kwargs["job_id"],
            bundle_digest=kwargs["bundle_digest"],
            outcome_contract_digest=kwargs["outcome_contract_digest"],
            context_json=text,
            context_digest=sha256(text.encode()).hexdigest(),
            created_at=200,
            qualification_authority_epoch=kwargs["authority_epoch"],
            qualification_authority_digest=kwargs["authority_digest"],
        )

    def prepare_open_world_invocation(self, *, binding, expected, **kwargs):
        if binding.analyst_input_identity not in self.rows:
            self.rows[binding.analyst_input_identity] = OpenWorldInvocationRow(
                job_id=binding.job_id,
                analyst_input_identity=binding.analyst_input_identity,
                state="prepared",
                component_identity_digest=binding.component_identity_digest,
                input_digest=expected.input_digest,
                component_role=binding.component_role,
                manifest_role=binding.manifest_role,
                user_id=binding.user_id,
                session_id=binding.session_id,
                qualification_result_digest=binding.qualification_result_digest,
                trajectory_digest=binding.trajectory_digest,
                candidate_artifact_digest=None,
                candidate_analyst_input_identity=None,
                output_json=None,
                output_digest=None,
                observed_provider=None,
                observed_model_revision=None,
                called_at=None,
                latency_ms=None,
                usage_json=None,
            )
        return self.rows[binding.analyst_input_identity]

    def complete_open_world_invocation(self, *, binding, completion, **kwargs):
        row = replace(
            self.rows[binding.analyst_input_identity],
            state="completed",
            **dataclasses.asdict(completion),
        )
        self.rows[binding.analyst_input_identity] = row
        return row

    def load_open_world_invocation(self, *, binding, **kwargs):
        return self.rows[binding.analyst_input_identity]

    def write_open_world_artifact(self, **kwargs):
        if (
            kwargs["artifact_kind"] == "open_world_candidate"
            and self.crash_on_candidate
        ):
            raise KeyboardInterrupt("process exited after completed provider call")
        artifact = PlaybookOptimizationArtifact(
            artifact_id=len(self.artifacts) + 1,
            created_at=200,
            updated_at=200,
            **{
                name: kwargs[name]
                for name in (
                    "job_id",
                    "artifact_kind",
                    "content_json",
                    "content_digest",
                )
            },
        )
        self.artifacts[artifact.artifact_kind] = artifact
        return artifact

    def read_open_world_artifact(self, *, artifact_kind, **kwargs):
        return self.artifacts.get(artifact_kind)

    def advance_playbook_optimization_stage(self, job_id, fence, stage, **kwargs):
        self.job.stage = stage
        self.job.terminal_outcome = kwargs.get("terminal_outcome")
        if stage == "failed":
            self.job.status = "failed"
        elif stage == "abstained":
            self.job.status = "skipped"
        return True

    def get_playbook_optimization_job(self, job_id):
        return self.job

    def merge_playbook_optimization_job_metadata(self, job_id, patch):
        self.failure = patch


@pytest.fixture
def runtime(source):

    class Transport:
        transport_identity = SimpleNamespace(
            provider="test", model_revision="frozen-model"
        )

        def __init__(self):
            self.calls = []

        def complete(self, **kwargs):
            self.calls.append(kwargs)
            schema = kwargs["response_format"]
            if issubclass(schema, Diagnostic):
                value = source["diagnostics"][0]
            else:
                assert issubclass(schema, Proposal), "unexpected provider wire contract"
                value = source["proposal"]
            return ObservedStructuredCompletion(
                value=value,
                observed_provider="test",
                observed_model_revision="frozen-model",
                called_at=200,
                latency_ms=10,
                usage_json=None,
            )

    transport = Transport()
    method = qualify_rule_evidence_method(transport)
    storage = MemoryStore(method, source["incumbent"])
    # This in-memory double implements the promoted runner's exercised storage
    # operations; unrelated evidence-loading/publication methods are intentional
    # omissions. Keep the protocol cast at this single fake-to-runtime boundary.
    store = RuleEvidenceAnalysisStore(
        cast(RuleEvidenceStorage, storage), method, transport, lambda: 200
    )
    runner = RuleEvidenceAttemptRunner(
        store,
        AttemptLease(
            OptimizationJobClaim(job_id=1, owner="worker", fence=1, expires_at=4000)
        ),
        3600,
    )
    bundle = source["bundle"]
    frozen = FrozenOpenWorldEvidence(
        job=PlaybookOptimizationJob(
            optimizer_kind="offline_tuner_open_world",
            target_kind="user_playbook",
            target_id=source["incumbent"].user_playbook_id,
            **vars(storage.job),
        ),
        artifact=PlaybookOptimizationArtifact(
            job_id=storage.job.job_id,
            artifact_kind="open_world_evidence_bundle",
            content_json=bundle.canonical_bytes().decode(),
            content_digest=bundle.digest,
            created_at=200,
            updated_at=200,
        ),
        bundle=bundle,
    )
    evidence = SimpleNamespace(
        bundle_digest=source["bundle"].digest,
        trajectory_json_by_session=source["trajectory_json_by_session"],
        documents_by_session={
            (doc.user_id, doc.session_id): doc for doc in source["documents"]
        },
    )
    return (
        runner,
        transport,
        storage,
        {"frozen": frozen, "incumbent": source["incumbent"], "evidence": evidence},
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("This test must never access the network")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setenv("MINIMAX_API_KEY", "scripted-only")
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")


class MemoryCheckpoints:
    """Transactional test port; the storage backends' own tests verify the real owner."""

    def __init__(self):
        self.rows = {}
        self.lock = RLock()

    @contextmanager
    def commit_scope(self):
        with self.lock:
            original = deepcopy(self.rows)
            try:
                yield
            except BaseException:
                self.rows = original
                raise

    def session_compaction_checkpoint(
        self,
        *,
        job_id,
        identity,
        fence,
        now,
        protocol=None,
        receipt=None,
        expected_digest=None,
    ):
        assert protocol == PROTOCOL and fence == 1 and now == 1
        key = job_id, identity
        old = self.rows.get(key)
        if receipt is None:
            return deepcopy(old)
        assert expected_digest == (digest(old) if old is not None else None), (
            "stale CAS"
        )
        current = SharedReceipt.model_validate(receipt)
        if old is not None:
            validate_advance(SharedReceipt.model_validate(old), current)
        self.rows[key] = deepcopy(receipt)
        return deepcopy(receipt)
