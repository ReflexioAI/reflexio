"""Shared preparation for newly admitted jobs; legacy jobs retain their protocol."""

import json
from collections.abc import Callable
from dataclasses import replace
from types import MappingProxyType
from typing import Any, cast

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm.litellm_client import LiteLLMConfig
from reflexio.server.services.input_compression import (
    CompressionBudget,
    CompressionError,
)
from reflexio.server.services.offline_tuner.evidence.native_capacity import (
    verified_native_context,
)
from reflexio.server.services.offline_tuner.evidence.shared_compaction_checkpoint import (
    PROTOCOL,
    digest,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    _LiteLLMOpenWorldModelClient,
)
from reflexio.server.services.offline_tuner.open_world.models import EvidenceDocument
from reflexio.server.services.offline_tuner.open_world.shared_compaction import (
    JournaledCompressionClient,
    SharedCompactionJournal,
)
from reflexio.server.services.playbook.rule_evidence import (
    PAYLOAD_PROXY_TOKEN_LIMIT,
    proposal_request,
    request_input_tokens,
)
from reflexio.server.services.playbook.rule_evidence.models import (
    RuleEvidenceInput,
    Session,
    Turn,
)
from reflexio.server.services.playbook.rule_evidence.wire import diagnostic_request
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
)

from .evidence_documents import RehydratedEvidence
from .hierarchical_compaction import compress_session_evidence
from .rule_evidence import native_rule_evidence_input


def _compression_budget(output_tokens: int, context_tokens: int) -> CompressionBudget:
    # Large frozen sessions need wider leaves than the shared 32K default.
    # This task ceiling is deliberately below verified context; the native
    # request guard still prices messages/schema and reserves model output.
    input_tokens = min(131_072, context_tokens // 2)
    return CompressionBudget(
        output_tokens=output_tokens,
        input_tokens=input_tokens,
        section_tokens=input_tokens,
    )


class ConsumerEnvelope:
    """Price the existing consumer's complete envelopes without calling a model.

    Proposal diagnostics/audits do not exist yet. This prices its fixed source
    envelope only; normal consumer and provider guards still check actual calls.
    """

    def __init__(
        self, data: RuleEvidenceInput, transport: _LiteLLMOpenWorldModelClient
    ):
        self.data, self.transport = data, transport

    @staticmethod
    def requests(
        data: RuleEvidenceInput,
    ) -> list[tuple[str, dict[str, object], type[StrictStructuredOutput]]]:
        requests: list[tuple[str, dict[str, object], type[StrictStructuredOutput]]] = [
            diagnostic_request(data, session) for session in data.sessions
        ]
        requests.append(proposal_request(data, [], []))
        return requests

    def contract(self, data: RuleEvidenceInput) -> str:
        return digest(
            [
                self.transport.preview_request(*request)[2]
                for request in self.requests(data)
            ]
        )

    def measure(self, data: RuleEvidenceInput) -> int:
        costs = []
        for system, payload, schema in self.requests(data):
            tokens, capacity, _contract = self.transport.preview_request(
                system, payload, schema
            )
            if capacity <= 0:
                raise CompressionError("Native consumer has no input capacity")
            # Normalize native and portable ceilings to the same task units.
            costs.append(
                max(
                    request_input_tokens(system, payload, schema),
                    (tokens * PAYLOAD_PROXY_TOKEN_LIMIT + capacity - 1) // capacity,
                )
            )
        return max(costs)

    @staticmethod
    def session(original: Session, document: EvidenceDocument) -> Session:
        return original.model_copy(
            update={
                "source_limitations": document.compaction_summary or "",
                "messages": [
                    Turn(turn=index, role=span.speaker, content=span.immutable_text)
                    for index, span in enumerate(document.spans)
                ]
                or [
                    original.messages[0].model_copy(
                        update={"content": original.messages[0].content.strip()[0]}
                    )
                ],
            }
        )

    def frame(self, sessions: list[Session]) -> RuleEvidenceInput:
        return self.data.model_copy(update={"sessions": sessions})


def prepare_shared_evidence(
    *,
    pipeline: Any,
    storage: Any,
    freeze: Any,
    evidence: RehydratedEvidence,
    before_call: Callable[[], None],
    now: Callable[[], int],
    model: str,
    project_id: str,
) -> RehydratedEvidence:
    """Prepare frozen originals with job-local, non-refundable dispatch receipts."""
    before_call()
    frozen = freeze.frozen
    if frozen.claim is None or pipeline.compaction_generation_client is None:
        raise CompressionError(
            "Shared compaction requires a live claim and generation client"
        )
    if not isinstance(pipeline.transport, _LiteLLMOpenWorldModelClient):
        raise CompressionError(
            "Shared compaction requires the qualified native consumer"
        )
    documents = dict(evidence.documents_by_session)
    data = native_rule_evidence_input(
        bundle=frozen.bundle,
        incumbent=freeze.incumbent,
        documents=list(documents.values()),
        trajectory_json_by_session=evidence.trajectory_json_by_session,
    )
    consumer = ConsumerEnvelope(data, pipeline.transport)
    if consumer.measure(data) <= PAYLOAD_PROXY_TOKEN_LIMIT:
        return evidence

    reserved = {
        (event.user_id, event.session_id)
        for event in frozen.bundle.expected.events
        if event.compaction_required
    }
    pending = [
        index
        for index, session in enumerate(data.sessions)
        if (session.user_id, session.session_id) in reserved
    ]
    if not pending:
        raise CompressionError(
            "Native consumer exceeds capacity without reserved compaction"
        )

    # Keep unreserved sessions unchanged and price their complete payload first.
    # Only reserved, not-yet-prepared sessions use minimal pricing frames. These
    # frames never become evidence; final selections still bind to the original
    # sources through the adapter and native consumer.
    sessions = [
        session.model_copy(
            update={
                "source_limitations": "",
                "messages": [
                    session.messages[0].model_copy(
                        update={"content": session.messages[0].content.strip()[0]}
                    )
                ],
            }
        )
        if index in pending
        else session
        for index, session in enumerate(data.sessions)
    ]
    context = verified_native_context(model)
    for position, index in enumerate(pending):
        original = data.sessions[index]
        before_call()
        key = original.user_id, original.session_id
        document = documents[key]
        baseline = consumer.measure(consumer.frame(sessions))
        if baseline >= PAYLOAD_PROXY_TOKEN_LIMIT:
            raise CompressionError(
                "Native consumer fixed envelope exceeds input capacity"
            )
        target = baseline + (PAYLOAD_PROXY_TOKEN_LIMIT - baseline) // (
            len(pending) - position
        )

        def candidate_cost(
            candidate: EvidenceDocument,
            index: int = index,
            original: Session = original,
        ) -> int:
            candidate_sessions = list(sessions)
            candidate_sessions[index] = consumer.session(original, candidate)
            return consumer.measure(consumer.frame(candidate_sessions))

        contract = digest(
            {
                "protocol": PROTOCOL,
                "consumer": consumer.contract(consumer.frame(sessions)),
                "index": index,
                "target": target,
            }
        )
        # Runtime model/plan changes must fail at bind(), not create a new root
        # and silently refund a job's previously spent physical attempts.
        identity = digest(
            {
                "protocol": PROTOCOL,
                "job_id": freeze.job_id,
                "user_id": document.user_id,
                "session_id": document.session_id,
                "trajectory_digest": document.trajectory_digest,
            }
        )
        journal = SharedCompactionJournal(
            storage,
            job_id=freeze.job_id,
            fence=frozen.claim.fence,
            identity=identity,
            before_call=before_call,
            now=now,
            limit=128,
        )
        client = JournaledCompressionClient(
            LiteLLMConfig(
                model=model,
                api_key_config=pipeline.compaction_generation_client.config.api_key_config,
                max_tokens=8192,
                fallback_models=[],
            ),
            journal,
        )
        events = [
            event
            for event in frozen.bundle.expected.events
            if (event.user_id, event.session_id) == key
        ]
        trajectory = cast(
            CanonicalSessionTrajectory,
            json.loads(evidence.trajectory_json_by_session[key]),
        )
        through_request_id = events[0].trajectory_through_request_id
        # None means the whole frozen session, not missing source authority.
        # Derive its physical terminal only from the verified preimage; keep
        # explicit cutoffs (including invalid blanks) for the adapter to check.
        if through_request_id is None and trajectory["requests"]:
            through_request_id = trajectory["requests"][-1]["request"]["request_id"]
        prepared, _ = compress_session_evidence(
            document,
            project_id=project_id,
            through_request_id=through_request_id,
            frozen_trajectory=trajectory,
            governance_epochs=frozen.bundle.target.governance_epoch_vector,
            budget=_compression_budget(target, context),
            client=client,
            context_limits={model: context},
            before_call=before_call,
            journal=journal,
            measure_document=candidate_cost,
            consumer_contract=contract,
        )
        documents[key] = prepared
        sessions[index] = consumer.session(original, prepared)
    before_call()
    final = native_rule_evidence_input(
        bundle=frozen.bundle,
        incumbent=freeze.incumbent,
        documents=list(documents.values()),
        trajectory_json_by_session=evidence.trajectory_json_by_session,
    )
    if consumer.measure(final) > PAYLOAD_PROXY_TOKEN_LIMIT:
        raise CompressionError("Prepared evidence exceeds the native consumer envelope")
    return replace(evidence, documents_by_session=MappingProxyType(documents))
