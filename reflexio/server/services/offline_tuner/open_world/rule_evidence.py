"""Native source and exact-edit checks for the promoted rule-evidence method.

These pure functions establish content/source integrity, not permission to write.
The existing invocation ledger, qualification authority and atomic publication
transaction must additionally authorize every completed call and publication.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from reflexio.models.api_schema.domain import UserPlaybook
from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.offline_tuner.evidence.manifests import (
    canonical_json_bytes,
)
from reflexio.server.services.offline_tuner.evidence.open_world_models import (
    RULE_EVIDENCE_POLICY_ID,
    OpenWorldEvidenceBundle,
    validate_open_world_evidence_bundle,
)
from reflexio.server.services.offline_tuner.open_world.citations import verify_citations
from reflexio.server.services.offline_tuner.open_world.models import (
    EvidenceCitation,
    EvidenceDocument,
    VerifiedCitation,
)
from reflexio.server.services.offline_tuner.open_world_records import (
    OpenWorldEvidenceTargetStaleError,
)
from reflexio.server.services.playbook.rule_evidence import run_rule_evidence
from reflexio.server.services.playbook.rule_evidence.models import (
    Incumbent,
    RuleEvidenceInput,
    RuleEvidenceResult,
    Session,
    ToolContract,
    Turn,
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
from reflexio.server.services.storage.session_outcome_identity import (
    CanonicalSessionTrajectory,
)

from .evidence_documents import project_trajectory_spans


def native_rule_evidence_input(
    *,
    bundle: OpenWorldEvidenceBundle,
    incumbent: UserPlaybook,
    documents: Sequence[EvidenceDocument],
    trajectory_json_by_session: Mapping[tuple[str, str], str],
) -> RuleEvidenceInput:
    """Build the exact portable input from a verified frozen native source set."""
    validate_open_world_evidence_bundle(bundle)
    if not bundle.eligible:
        raise ValueError("rule evidence needs an eligible source bundle")
    version = user_playbook_full_version_fingerprint(incumbent)
    if (
        not incumbent.user_id
        or version != bundle.target.incumbent_full_version_fingerprint
        or incumbent.user_playbook_id != bundle.target.user_playbook_id
        or incumbent.status is not None
        or incumbent.superseded_by is not None
        or incumbent.retired_at is not None
    ):
        raise OpenWorldEvidenceTargetStaleError(
            "incumbent does not match the frozen current version"
        )
    expected = {
        (session.user_id, session.session_id): session
        for session in bundle.discovery.sessions
    }
    keys = [(document.user_id, document.session_id) for document in documents]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("documents must cover the exact frozen source population")
    if set(trajectory_json_by_session) != set(expected):
        raise ValueError("trajectory preimages must cover the exact frozen population")
    by_key = dict(zip(keys, documents, strict=True))
    sessions = []
    seen_spans: set[str] = set()
    for source in bundle.discovery.sessions:
        document = by_key[(source.user_id, source.session_id)]
        trajectory_json = trajectory_json_by_session[
            (source.user_id, source.session_id)
        ]
        if (
            not isinstance(trajectory_json, str)
            or sha256(trajectory_json.encode()).hexdigest() != source.trajectory_digest
        ):
            raise ValueError("trajectory preimage digest differs from frozen source")
        originals = project_trajectory_spans(
            cast(CanonicalSessionTrajectory, json.loads(trajectory_json)),
            trajectory_digest=source.trajectory_digest,
            tenant_id=bundle.target.tenant_id,
            role="discovery",
            user_id=source.user_id,
            session_id=source.session_id,
            include_tools=True,
        )
        original_by_path = {span.field_path: span for span in originals}
        for span in document.spans:
            match = re.fullmatch(r"(.+)\[(\d+):(\d+)\]", span.field_path)
            original = original_by_path.get(match[1] if match else span.field_path)
            if original is None or (
                span.request_id,
                span.interaction_id,
                span.speaker,
            ) != (original.request_id, original.interaction_id, original.speaker):
                raise ValueError(
                    "source span coordinates differ from original trajectory"
                )
            start, end = (
                (int(match[2]), int(match[3]))
                if match
                else (0, len(original.immutable_text))
            )
            if (
                not (0 <= start < end <= len(original.immutable_text))
                or span.immutable_text != original.immutable_text[start:end]
            ):
                raise ValueError("source span text differs from original trajectory")
        if (
            document.manifest_role != "discovery"
            or document.user_id != incumbent.user_id
            or document.trajectory_digest != source.trajectory_digest
            or not any(span.speaker == "agent" for span in document.spans)
        ):
            raise ValueError("source document must bind the owned frozen trajectory")
        for span in document.spans:
            if (
                span.tenant_id != bundle.target.tenant_id
                or span.span_id in seen_spans
                or not span.span_id.startswith(document.trajectory_digest + ":")
            ):
                raise ValueError("source span tenant or immutable identity is invalid")
            seen_spans.add(span.span_id)
        sessions.append(
            Session(
                session_id=source.session_id,
                user_id=source.user_id,
                exposed_version=version,
                source_limitations=document.compaction_summary or "",
                messages=[
                    Turn(turn=index, role=span.speaker, content=span.immutable_text)
                    for index, span in enumerate(document.spans)
                ],
            )
        )
    context = bundle.task_context
    return RuleEvidenceInput(
        incumbent=Incumbent(
            id=incumbent.user_playbook_id,
            user_id=incumbent.user_id,
            version=version,
            trigger=incumbent.trigger,
            content=incumbent.content,
        ),
        sessions=sessions,
        agent_context=context.agent_context,
        success_definition=context.success_definition,
        available_tools=[
            ToolContract(name=tool.name, description=tool.description)
            for tool in context.available_tools
        ],
        tool_context_status=context.tool_context_status,
    )


def observed_output_bytes(value: Diagnostic | Proposal) -> bytes:
    """Canonical actual model output, never an invented legacy memo/revision."""
    if not isinstance(value, (Diagnostic, Proposal)):
        raise TypeError("rule-evidence output must be Diagnostic or Proposal")
    return canonical_json_bytes(
        {
            "schema_version": "v1",
            "contract_type": (
                "rule_evidence_diagnostic"
                if isinstance(value, Diagnostic)
                else "rule_evidence_proposal"
            ),
            "payload": value.model_dump(mode="json"),
        }
    )


def diagnostic_population_digest(diagnostics: Sequence[Diagnostic]) -> str:
    """Bind each session's exact observed model output in stable source order."""
    keys = [(diagnostic.user_id, diagnostic.session_id) for diagnostic in diagnostics]
    if len(set(keys)) != len(keys):
        raise ValueError("diagnostic population must not duplicate a session")
    return sha256(
        canonical_json_bytes(
            {
                "contract_type": "rule_evidence_diagnostic_population",
                "schema_version": "v1",
                "payload": [
                    sha256(observed_output_bytes(diagnostic)).hexdigest()
                    for diagnostic in diagnostics
                ],
            }
        )
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceEditValidation:
    """Structural/source checks; publication still needs durable authority."""

    policy_id: str
    result: RuleEvidenceResult
    diagnostic_population_digest: str
    proposal_output_digest: str | None
    verified_citations: tuple[VerifiedCitation, ...]


def validate_evidence_edit(
    *,
    bundle: OpenWorldEvidenceBundle,
    incumbent: UserPlaybook,
    documents: Sequence[EvidenceDocument],
    trajectory_json_by_session: Mapping[tuple[str, str], str],
    diagnostics: Sequence[Diagnostic],
    proposal: Proposal | None,
) -> EvidenceEditValidation:
    """Re-run the frozen method's checks over durable observations, without an LLM.

    Reusing the method itself prevents native edit validation from drifting away
    from its gap admission, source-coordinate and exact-replacement semantics.
    It does not generate replacement observations or infer semantic truth.
    """
    data = native_rule_evidence_input(
        bundle=bundle,
        incumbent=incumbent,
        documents=documents,
        trajectory_json_by_session=trajectory_json_by_session,
    )
    if len(diagnostics) != len(data.sessions):
        raise ValueError("diagnostics must cover every frozen input session")
    observed = iter([*diagnostics, *([] if proposal is None else [proposal])])

    def replay(
        _system: str,
        _payload: dict[str, object],
        schema: type[StrictStructuredOutput],
    ) -> StrictStructuredOutput:
        try:
            value = next(observed)
        except StopIteration as exc:
            raise ValueError("a required durable model output is missing") from exc
        if not isinstance(
            value, Diagnostic if issubclass(schema, Diagnostic) else Proposal
        ):
            raise ValueError("durable output schema/order does not match the method")
        return schema.model_validate_json(value.model_dump_json())

    result = run_rule_evidence(data, replay)
    if next(observed, None) is not None:
        raise ValueError("durable outputs include a call the method did not request")
    by_coordinate = {
        (document.session_id, index): span
        for document in documents
        for index, span in enumerate(document.spans)
    }
    spans = tuple(span for document in documents for span in document.spans)
    selected = {
        (reference.session_id, reference.turn): by_coordinate[
            (reference.session_id, reference.turn)
        ]
        for reference in result.evidence_refs
    }
    citations = tuple(
        verify_citations(
            tuple(
                EvidenceCitation(span_id=span.span_id, exact_quote=span.immutable_text)
                for span in selected.values()
            ),
            allowed_spans=spans,
        )
    )
    return EvidenceEditValidation(
        policy_id=RULE_EVIDENCE_POLICY_ID,
        result=result,
        diagnostic_population_digest=diagnostic_population_digest(diagnostics),
        proposal_output_digest=(
            sha256(observed_output_bytes(proposal)).hexdigest()
            if proposal is not None
            else None
        ),
        verified_citations=citations,
    )
