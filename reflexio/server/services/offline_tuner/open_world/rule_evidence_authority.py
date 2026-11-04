"""Qualification and immutable identities for the single evidence-driven method."""

import dataclasses
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.services.playbook.publication import QualificationAuthorityRef
from reflexio.server.services.playbook.rule_evidence import (
    input_budget_policy,
    prompt,
    run_rule_evidence,
)
from reflexio.server.services.playbook.rule_evidence.models import (
    RuleEvidenceInput,
    RuleEvidenceResult,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    WIRE_NORMALIZATION_VERSION,
    SupportTurns,
    contract_schema,
    generation_profile,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    GlobalRuleAudit as Proposal,
)

from ..evidence.manifests import canonical_json_bytes
from ..evidence.open_world_models import RULE_EVIDENCE_POLICY_ID
from .analyst import generation_settings_digest
from .identity import pinned_generator_authorization_digest
from .models import ComponentIdentity


def _digest(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _structural_qualification() -> dict[str, object]:
    """Execute native contract fixtures; semantic method selection happens offline."""
    data = RuleEvidenceInput.model_validate_json(
        '{"incumbent":{"id":1,"user_id":"u","version":"v",'
        '"trigger":null,"content":"Send once. Say delivered. Keep recipient."},'
        '"sessions":[{"session_id":"s","user_id":"u","exposed_version":"v",'
        '"source_limitations":"","messages":[{"turn":0,"role":"agent",'
        '"content":"Service returned queued; reported delivered."}]}],'
        '"agent_context":"","success_definition":"","available_tools":[], '
        '"tool_context_status":"missing"}'
    )
    diagnostic = Diagnostic.model_validate(
        {
            "session_id": "s",
            "user_id": "u",
            "observations": [
                {
                    "dimension": "evidence_grounding",
                    "kind": "problem",
                    "observation": "Unsupported delivery claim",
                    "consequence": "False status",
                    "evidence_refs": {"r1": 0},
                }
            ],
            "rule_assessment": {
                "attribution": "instruction_conflicts_requirement",
                "target_scope": "Delivery-state reporting",
                "counterevidence_refs": {},
                "clause_start": "CLAUSE_2",
                "clause_end": "CLAUSE_2",
                "expected_if_followed": "Report delivery",
                "reason": "Rule conflicts with actual status",
                "evidence_refs": {"r1": 0},
            },
            "missing_information": [],
        }
    )
    proposal = Proposal.model_validate(
        {
            "hypothesis": "Report actual status",
            "edits": [
                {
                    "clause_start": "CLAUSE_2",
                    "clause_end": "CLAUSE_2",
                    "replacement": "Report queued as queued.",
                    "defect": "Queued is not delivered",
                    "evidence_refs": {"r1": "REF_1"},
                }
            ],
            "abstention_reason": "",
        }
    )

    def run(source: RuleEvidenceInput, observation: Diagnostic) -> RuleEvidenceResult:
        def invoke(
            _system: str,
            _payload: dict[str, object],
            schema: type[StrictStructuredOutput],
        ) -> StrictStructuredOutput:
            if issubclass(schema, Diagnostic):
                # Exercise the input-bound schema in the provider's decoded-JSON
                # mode; returning a generic model bypasses source-ID validation.
                return schema.model_validate(observation.model_dump(mode="json"))
            if observation.rule_assessment.attribution == "no_supported_target_defect":
                return schema.model_validate_json(
                    proposal.model_copy(
                        update={
                            "edits": [],
                            "hypothesis": "",
                            "abstention_reason": "No supported defect",
                        }
                    ).model_dump_json()
                )
            return schema.model_validate_json(proposal.model_dump_json())

        return run_rule_evidence(source, invoke)

    repaired = run(data, diagnostic)
    adequate = diagnostic.model_copy(
        update={
            "rule_assessment": diagnostic.rule_assessment.model_copy(
                update={"attribution": "no_supported_target_defect"}
            )
        }
    )
    ignored = diagnostic.model_copy(
        update={
            "rule_assessment": diagnostic.rule_assessment.model_copy(
                update={"attribution": "adequate_guidance_ignored"}
            )
        }
    )
    independent = run(data, ignored)
    checks = {
        "exact_edit": repaired.revised_content
        == ("Send once. Report queued as queued. Keep recipient."),
        "abstention": run(data, adequate).decision == "abstain",
        "global_edit_without_local_gap": independent.decision == "proposal"
        and independent.diagnostics[0].rule_assessment.disposition
        == "adherence_failure",
    }
    invalid = diagnostic.model_copy(
        update={
            "rule_assessment": diagnostic.rule_assessment.model_copy(
                update={"evidence_refs": SupportTurns(r1=999)}
            )
        }
    )
    for name, source, observation in (
        ("invalid_reference_refused", data, invalid),
        (
            "invalid_clause_refused",
            data,
            diagnostic.model_copy(
                update={
                    "rule_assessment": diagnostic.rule_assessment.model_copy(
                        update={"clause_start": "UNKNOWN"}
                    )
                }
            ),
        ),
    ):
        try:
            run(source, observation)
        except ValueError:
            checks[name] = True
        else:
            checks[name] = False
    if not all(checks.values()):
        raise ValueError("native rule-evidence structural qualification failed")
    suite_digest = _digest(
        {
            "suite": "native-global-rule-audit-contract-v7",
            "input": data.model_dump(mode="json"),
            "diagnostic": diagnostic.model_dump(mode="json"),
            "proposal": proposal.model_dump(mode="json"),
            "checks": tuple(checks),
        }
    )
    return {"suite_digest": suite_digest, "checks": checks, "passed": True}


@dataclass(frozen=True)
class RuleEvidenceMethod:
    """The two real components and actual structural-qualification receipt."""

    diagnostic: ComponentIdentity
    proposal: ComponentIdentity
    qualification_receipt: dict[str, object]

    @property
    def suite_digest(self) -> str:
        return str(self.qualification_receipt["suite_digest"])

    @property
    def result_digest(self) -> str:
        return _digest(self.qualification_receipt)

    @property
    def proposal_authorization_digest(self) -> str:
        return pinned_generator_authorization_digest(self.proposal)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "offline-tuner-rule-evidence-authority-v1",
            "policy_id": RULE_EVIDENCE_POLICY_ID,
            "diagnostic": {
                "component_identity": dataclasses.asdict(self.diagnostic),
                "suite_digest": self.suite_digest,
                "result_digest": self.result_digest,
                "qualification_receipt": self.qualification_receipt,
            },
            "proposal": {
                "component_identity": dataclasses.asdict(self.proposal),
                "authorization_digest": self.proposal_authorization_digest,
            },
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json_bytes(self.payload).decode()

    @property
    def digest(self) -> str:
        return sha256(self.canonical_json.encode()).hexdigest()

    def authority_ref(self, *, epoch: int) -> QualificationAuthorityRef:
        return QualificationAuthorityRef(
            epoch=epoch,
            authority_digest=self.digest,
            discovery_component_identity_digest=self.diagnostic.digest,
            discovery_qualification_suite_digest=self.suite_digest,
            discovery_qualification_result_digest=self.result_digest,
            held_out_component_identity_digest=None,
            held_out_qualification_suite_digest=None,
            held_out_qualification_result_digest=None,
            candidate_generator_identity_digest=self.proposal.digest,
            candidate_generator_authorization_digest=self.proposal_authorization_digest,
            policy_id=RULE_EVIDENCE_POLICY_ID,
        )


def qualify_rule_evidence_method(transport: Any) -> RuleEvidenceMethod:
    receipt = _structural_qualification()
    components = []
    for role, stage, schema in (
        ("discovery", "judge", Diagnostic),
        ("candidate", "tune", contract_schema("tune")),
    ):
        components.append(
            ComponentIdentity(
                component_role=cast(Literal["discovery", "candidate"], role),
                provider=transport.transport_identity.provider,
                immutable_model_revision=transport.transport_identity.model_revision,
                prompt_id=f"playbook.rule_evidence.{stage}",
                prompt_version="2",
                prompt_template_digest=sha256(prompt(stage).encode()).hexdigest(),
                output_schema_id=f"rule_evidence_{schema.__name__.lower()}",
                output_schema_version="v7",
                output_schema_digest=_digest(schema.model_json_schema()),
                generation_settings_digest=_digest(
                    {
                        "transport": generation_settings_digest(),
                        "input_budget_policy": input_budget_policy(),
                        "wire_normalization": WIRE_NORMALIZATION_VERSION,
                        "stage": generation_profile(
                            cast(Literal["judge", "tune"], stage),
                            provider=transport.transport_identity.provider,
                        ),
                    }
                ),
                qualification_suite_version="native-global-rule-audit-contract-v7",
            )
        )
    return RuleEvidenceMethod(components[0], components[1], receipt)
