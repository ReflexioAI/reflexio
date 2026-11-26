"""Qualification for the two-component database authority contract."""

from hashlib import sha256
from typing import Any, Literal, cast

from reflexio.server.services.playbook.rule_evidence import (
    input_budget_policy,
    prompt,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    WIRE_NORMALIZATION_VERSION,
    contract_schema,
    generation_profile,
)
from reflexio.server.services.playbook.rule_evidence.wire import (
    BoundedTargetDiagnostic as Diagnostic,
)

from .analyst import generation_settings_digest
from .models import ComponentIdentity
from .rule_evidence_authority import (
    RuleEvidenceMethod,
    _digest,
    _structural_qualification,
)


def qualify_rule_evidence_v1(transport: Any) -> RuleEvidenceMethod:
    """Qualify Diagnostic and Proposal without preparing Trigger authority."""
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
