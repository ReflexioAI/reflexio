"""Portable V5 Trigger maintenance; no storage or publication authority.

The native source-only gate profile is explicit and remains unqualified for live
quality. This component is not yet wired into the durable offline-tuner runner.
"""

from hashlib import sha256
from importlib.resources import files
from typing import Literal, cast

from pydantic import Field

from reflexio.server.services.playbook.publication import canonical_json_bytes
from reflexio.server.services.playbook.revision import FinalPlaybookRevision

from . import PAYLOAD_PROXY_TOKEN_LIMIT, Invoke, request_input_tokens
from .models import EvidenceModel, Text

PROFILE = "trigger-maintenance-native-source-only-v1"
REWRITE_LIMIT = 320
SupportedGap = Literal["opposed_consultation_state", "promised_help_absent"]


class TriggerSourceFrame(EvidenceModel):
    """Exact source/version bindings supplied by the caller, not write authority.

    ``final_content_artifact_digest`` names the accepted Content result, even if
    unchanged. Quotes against final Content must not masquerade as historical
    conversation evidence. Durable admission must verify these caller claims.
    """

    incumbent_id: int = Field(gt=0)
    owner_id: Text
    incumbent_version: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_content_artifact_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    original_content: Text
    original_trigger: str | None
    final_content: Text

    @property
    def digest(self) -> str:
        return sha256(canonical_json_bytes(self.model_dump(mode="json"))).hexdigest()


class TriggerSupport(EvidenceModel):
    field: Literal["trigger", "content"]
    quote: Text


class TriggerRevision(EvidenceModel):
    action: Literal["keep", "rewrite"]
    trigger: str
    reason: str = Field(min_length=1, max_length=1200)
    support: list[TriggerSupport]


class TriggerContrast(EvidenceModel):
    gap: Literal[
        "none", "opposed_consultation_state", "promised_help_absent", "uncertain"
    ]
    missed_task: str
    original_excludes_because: str
    content_actually_provides: str
    revision_admits_because: str
    support: list[TriggerSupport]
    missing_facts: list[str]


class TriggerGate(EvidenceModel):
    contrast: TriggerContrast
    revision: TriggerRevision


class BuiltTrigger(EvidenceModel):
    index: int
    revision: TriggerRevision


class TriggerBatch(EvidenceModel):
    # Keep the studied batch wire format, but native maintenance uses one source.
    revisions: list[BuiltTrigger] = Field(min_length=1, max_length=4)


class TriggerMaintenanceResult(EvidenceModel):
    profile: Literal["trigger-maintenance-native-source-only-v1"]
    source_digest: str
    status: Literal[
        "preserved_without_inference", "gate_keep", "builder_keep", "repaired"
    ]
    revision: FinalPlaybookRevision
    gate: TriggerGate | None
    builder: TriggerBatch | None


def trigger_prompt(stage: Literal["gate", "build"]) -> str:
    """Return packaged bytes to include in native component identity."""
    if stage not in {"gate", "build"}:
        raise ValueError("unknown Trigger stage")
    return (
        files(__package__).joinpath(f"trigger_{stage}.md").read_text(encoding="utf-8")
    )


def _validate_support(frame: TriggerSourceFrame, support: list[TriggerSupport]) -> None:
    source = {"trigger": frame.original_trigger or "", "content": frame.final_content}
    for citation in support:
        if citation.quote not in source[citation.field]:
            raise ValueError("Trigger support quote does not match its source frame")


def _validate_revision(frame: TriggerSourceFrame, revision: TriggerRevision) -> None:
    _validate_support(frame, revision.support)
    if revision.action == "keep":
        if revision.trigger != frame.original_trigger:
            raise ValueError("KEEP must preserve exact original Trigger bytes")
    elif (
        not revision.trigger.strip()
        or len(revision.trigger) > REWRITE_LIMIT
        or revision.trigger == frame.original_trigger
        or not revision.support
    ):
        raise ValueError(
            "REWRITE needs changed nonblank text <=320 characters and source support"
        )


def _check_request(
    system: str, payload: dict[str, object], schema: type[EvidenceModel]
) -> None:
    if request_input_tokens(system, payload, schema) > PAYLOAD_PROXY_TOKEN_LIMIT:
        raise ValueError(
            "Trigger source exceeds input budget; no evidence was truncated"
        )


def trigger_gate_request(
    frame: TriggerSourceFrame,
) -> tuple[str, dict[str, object], type[TriggerGate]]:
    """One exact native source-only envelope for sizing and durable dispatch."""
    return (
        trigger_prompt("gate"),
        {
            "source": {
                "trigger": frame.original_trigger,
                "content": frame.final_content,
            },
        },
        TriggerGate,
    )


def trigger_builder_request(
    frame: TriggerSourceFrame,
    gap: SupportedGap,
) -> tuple[str, dict[str, object], type[TriggerBatch]]:
    """Build only from source plus the bounded gap enum, never gate reasoning."""
    if gap not in {"opposed_consultation_state", "promised_help_absent"}:
        raise ValueError("Trigger builder needs a resolved supported gap")
    return (
        trigger_prompt("build"),
        {
            "sources": [
                {
                    "index": 0,
                    "original_trigger": frame.original_trigger,
                    "content": frame.final_content,
                    "gap": gap,
                }
            ],
        },
        TriggerBatch,
    )


def run_trigger_maintenance(
    frame: TriggerSourceFrame, invoke: Invoke
) -> TriggerMaintenanceResult:
    """Use one gate and at most one source-only builder; execution failures raise.

    Does not infer missing task probes, require a session pair, change Content,
    retry an adverse answer, admit invocations or publish. The caller owns those
    native boundaries and must bind these exact requests to its durable receipts.
    """
    frame = TriggerSourceFrame.model_validate_json(frame.model_dump_json())

    def result(
        status: Literal[
            "preserved_without_inference", "gate_keep", "builder_keep", "repaired"
        ],
        trigger: str | None,
        gate: TriggerGate | None = None,
        builder: TriggerBatch | None = None,
    ) -> TriggerMaintenanceResult:
        return TriggerMaintenanceResult(
            profile=PROFILE,
            source_digest=frame.digest,
            status=status,
            revision=FinalPlaybookRevision(
                revised_content=frame.final_content, revised_trigger=trigger
            ),
            gate=gate,
            builder=builder,
        )

    if (
        not frame.original_trigger
        or not frame.original_trigger.strip()
        or len(frame.original_trigger) > REWRITE_LIMIT
    ):
        return result("preserved_without_inference", frame.original_trigger)

    gate_system, gate_payload, _ = trigger_gate_request(frame)
    build_system, build_payload, _ = trigger_builder_request(
        frame, "opposed_consultation_state"
    )
    # Bound both fixed source envelopes before spending any inference. The real
    # request is checked again after the gate selects its bounded enum value.
    _check_request(gate_system, gate_payload, TriggerGate)
    _check_request(build_system, build_payload, TriggerBatch)
    gate = TriggerGate.model_validate_json(
        invoke(gate_system, gate_payload, TriggerGate).model_dump_json()
    )
    contrast = gate.contrast
    gap = contrast.gap
    _validate_support(frame, contrast.support)
    _validate_revision(frame, gate.revision)
    if gap in {"none", "uncertain"} or contrast.missing_facts:
        return result("gate_keep", frame.original_trigger, gate)
    if gate.revision.action == "keep":
        return result("gate_keep", frame.original_trigger, gate)
    if not all(
        value.strip()
        for value in (
            contrast.missed_task,
            contrast.original_excludes_because,
            contrast.content_actually_provides,
            contrast.revision_admits_because,
        )
    ) or {citation.field for citation in contrast.support} != {"trigger", "content"}:
        raise ValueError(
            "Trigger rewrite needs concrete contrast and both source fields"
        )

    build_system, build_payload, _ = trigger_builder_request(
        frame, cast(SupportedGap, gap)
    )
    _check_request(build_system, build_payload, TriggerBatch)
    builder = TriggerBatch.model_validate_json(
        invoke(build_system, build_payload, TriggerBatch).model_dump_json()
    )
    if len(builder.revisions) != 1 or builder.revisions[0].index != 0:
        raise ValueError("Native Trigger builder must return exactly source index 0")
    revision = builder.revisions[0].revision
    _validate_revision(frame, revision)
    return result(
        "repaired" if revision.action == "rewrite" else "builder_keep",
        revision.trigger,
        gate,
        builder,
    )
