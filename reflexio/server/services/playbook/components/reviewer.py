"""Evidence-preserving second-pass review for normal user playbook candidates."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.domain.enums import PlaybookReviewReasonCode
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.structured_output import normalize_provider_value
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_text_generation import ProviderRequestGuard
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.extraction.input_preparation import resolve_evidence_span
from reflexio.server.services.extraction.overflow_coordinator import (
    current_overflow_coordinator,
)
from reflexio.server.services.extraction.overflow_presentation import (
    protected_source_origins,
)
from reflexio.server.services.extraction.protected_record import (
    ProtectedRecord,
    ProtectedRecordUpdate,
)
from reflexio.server.services.playbook.playbook_evidence import (
    contains_call_local_turn_ref,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookEvidenceSource,
    PlaybookPromptContext,
    build_playbook_prompt_context,
)

logger = logging.getLogger(__name__)


# Decision-verb variants providers emit instead of the declared literals. Keys
# are in canonical ``normalize_provider_token`` form (casefolded, "_"-joined).
_DECISION_ALIASES = {
    "accepted": "accept",
    "approve": "accept",
    "keep": "accept",
    "retain": "accept",
    "revised": "revise",
    "edit": "revise",
    "edited": "revise",
    "rejected": "reject",
    "drop": "reject",
    "remove": "reject",
}


class PlaybookCandidateEvidenceError(ValueError):
    """A persisted candidate's cited evidence cannot be reconstructed."""


class CandidateRevision(BaseModel):
    """Complete replacement fields for one narrowly revised candidate."""

    content: str = Field(min_length=1)
    trigger: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")


#: Reason codes naming a defect with no salvageable core, so the only coherent
#: decision is ``reject``. Enforced structurally by
#: :meth:`CandidateReviewDecision.fatal_reason_codes_are_rejects`, and applied
#: by the coercion in `normalize_known_provider_shape`.
_FATAL_REASON_CODES = frozenset({"absence_inference", "not_agent_decision"})


class CandidateReviewDecision(BaseModel):
    """One exactly-accounted-for candidate decision."""

    candidate_id: str = Field(alias="id", min_length=1)
    decision: Literal["accept", "revise", "reject"]
    # Adding a code here is not self-contained: it must also appear in the
    # active prompt's "reason_code is exactly one of" list, or the model is
    # never told it exists. If it names an unsalvageable defect, add it to
    # _FATAL_REASON_CODES and to that prompt's "Always reject" line too.
    # test_prompt_reason_codes_match_the_schema_literal enforces the first.
    reason_code: PlaybookReviewReasonCode
    evidence_ids: list[str] = Field(default_factory=list)
    revision: CandidateRevision | None = None
    reason: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def fatal_reason_codes_are_rejects(self) -> CandidateReviewDecision:
        """A defect with no salvageable core can never end up a survivor.

        The prompt instructs the reviewer to reject these outright, but a prompt
        instruction is not an invariant: a response pairing one of these codes
        with ``accept`` or ``revise`` otherwise validates cleanly, and revision
        would launder the defect into tidier prose rather than removing it. That
        pairing has been observed in practice, so it is enforced structurally.

        Enforcement is the coercion in ``normalize_known_provider_shape``; this
        is the backstop that makes the invariant total. It is reachable only via
        a construction path that skips that coercion, so it raises rather than
        coercing a second time -- at that point the caller is our own code with a
        bug, not a provider, and no batch is at stake.

        Deliberately narrow. ``internal_status`` is NOT listed: an entry that
        merely rests on an internal event as evidence usually has a grounded
        core that revision can keep, and forcing it to reject was measured to
        destroy healthy entries.
        """
        if self.reason_code in _FATAL_REASON_CODES and self.decision != "reject":
            raise ValueError(
                f"{self.reason_code} requires decision='reject'; it has no grounded "
                f"core to narrow, so decision={self.decision!r} is invalid"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_known_provider_shape(cls, value: Any) -> Any:
        """Normalize narrow field-name variants without changing semantics."""
        if not isinstance(value, dict):
            return value
        data = dict(value)

        if "id" not in data and "candidate_id" not in data:
            for alias in ("candidate", "candidate_ref"):
                if alias in data:
                    data["id"] = data.pop(alias)
                    break
        if "decision" not in data and "action" in data:
            data["decision"] = data.pop("action")
        if "decision" in data:
            data["decision"] = normalize_provider_value(
                data["decision"], _DECISION_ALIASES
            )

        if "revision" not in data and "revised" in data:
            data["revision"] = data.pop("revised")
        revision = data.get("revision")
        if isinstance(revision, dict):
            normalized_revision = dict(revision)
            retained = normalized_revision.pop("retained_evidence_ids", None)
            if retained is None:
                retained = normalized_revision.pop("evidence_ids", None)
            if "evidence_ids" not in data and retained is not None:
                data["evidence_ids"] = retained
            data["revision"] = normalized_revision

        if "evidence_ids" not in data and "evidence" in data:
            data["evidence_ids"] = data.pop("evidence")
        elif "evidence" in data:
            data.pop("evidence")
        if "evidence_ids" not in data and "retained_evidence_ids" in data:
            data["evidence_ids"] = data.pop("retained_evidence_ids")
        elif "retained_evidence_ids" in data:
            data.pop("retained_evidence_ids")

        if data.get("decision") == "revise" and "revision" not in data:
            revision_fields = {
                field: data.pop(field)
                for field in ("content", "trigger", "rationale")
                if field in data
            }
            if revision_fields:
                data["revision"] = revision_fields

        if "reason_code" not in data and data.get("decision") in {
            "accept",
            "revise",
            "reject",
        }:
            data["reason_code"] = (
                "unsupported_evidence"
                if data["decision"] == "reject"
                else "grounded_useful"
            )

        # A fatal code paired with accept/revise is self-contradictory: the code
        # asserts nothing salvageable remains, and the decision tries to keep it.
        # Coerce to the decision the code already implies, rather than raising.
        #
        # Raising here was worse than it looked. This runs during PARSE of the
        # whole PlaybookCandidateReviewOutput, so one contradictory decision
        # invalidated every decision in the batch -- and the repair turn could
        # not fix it, because `_safe_validation_errors` deliberately drops the
        # message (Customer Content) and forwards only `decisions.N:
        # value_error`. The constraint is also absent from the JSON schema, since
        # a model_validator does not serialize, so the model was never told the
        # rule it had just broken. One slip therefore cost the whole review, with
        # a repair round-trip spent on an uninformative message.
        if data.get("reason_code") in _FATAL_REASON_CODES and data.get("decision") in {
            "accept",
            "revise",
        }:
            logger.warning(
                "event=playbook_review_fatal_code_coerced reason_code=%s from=%s",
                data.get("reason_code"),
                data.get("decision"),
            )
            data["decision"] = "reject"
            # A reject must retain no evidence and carry no revision, so clear
            # both -- otherwise the coerced object fails semantic validation and
            # reintroduces the batch loss by another route.
            data["revision"] = None
            data["evidence_ids"] = []
        return data


class PlaybookCandidateReviewOutput(BaseModel):
    """Structured output for one normal candidate review call."""

    decisions: list[CandidateReviewDecision]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_single_decision_wrapper(cls, value: Any) -> Any:
        """Move provider-duplicated single-decision fields into the decision.

        Some providers repeat ``candidate_id`` beside the ``decisions`` array
        even though the response schema places it on the sole array item. This
        normalization accepts only the unambiguous one-candidate equivalent;
        conflicting or multi-candidate wrapper fields remain invalid.
        """
        if not isinstance(value, dict):
            return value
        data = dict(value)
        decisions = data.get("decisions")
        if not isinstance(decisions, list) or len(decisions) != 1:
            return data
        decision = decisions[0]
        if not isinstance(decision, dict):
            return data
        normalized_decision = dict(decision)
        wrapper_fields = {
            "candidate_id": "id",
            "id": "id",
            "retained_evidence_ids": "evidence_ids",
            "evidence_ids": "evidence_ids",
        }
        for wrapper_field, decision_field in wrapper_fields.items():
            if wrapper_field not in data:
                continue
            wrapper_value = data[wrapper_field]
            existing = normalized_decision.get(decision_field)
            if existing is None and decision_field == "id":
                existing = normalized_decision.get("candidate_id")
            if existing is not None and existing != wrapper_value:
                return data
            normalized_decision.setdefault(decision_field, wrapper_value)
            data.pop(wrapper_field)
        data["decisions"] = [normalized_decision]
        return data


class SupportingExcerpt(BaseModel):
    """Transient verbatim evidence; provenance is not semantic approval."""

    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=600)

    model_config = ConfigDict(extra="forbid")


class CandidateSupportingEvidence(BaseModel):
    candidate_id: str = Field(alias="id", min_length=1)
    excerpts: list[SupportingExcerpt] = Field(min_length=1, max_length=16)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class EvidenceFirstReviewOutput(BaseModel):
    """Experimental internal response; evidence precedes the decisions."""

    supporting_evidence: list[CandidateSupportingEvidence]
    decisions: list[CandidateReviewDecision]

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class CandidateEvidenceUnit:
    """One validated evidence span exposed through a call-local unit id."""

    evidence_id: str
    turn_ref: str
    source_span: str
    interaction_id: int
    role: str = ""
    request_source: str = ""


@dataclass(frozen=True)
class PlaybookReviewOutcome:
    """A validated review result plus the evidence units it was decided against.

    Carrying the units keeps ``decide`` -> ``apply_decisions`` a single pass:
    rebuilding them means re-walking the whole interaction window.
    """

    output: PlaybookCandidateReviewOutput
    units_by_candidate: dict[str, list[CandidateEvidenceUnit]]
    # Only private evaluation artifacts may retain these. apply_decisions never
    # copies them into saved playbooks, notes, or public response models.
    supporting_evidence: list[CandidateSupportingEvidence] = field(default_factory=list)


def _append_review_note(
    existing: str | None,
    decision: str,
    reason_code: str,
    reason: str | None,
) -> str:
    normalized_reason = " ".join((reason or reason_code).split())[:500]
    review_note = (
        f"First-pass reviewer: {decision} ({reason_code}) — {normalized_reason}"
    )
    return f"{existing.rstrip()}\n{review_note}" if existing else review_note


class PlaybookCandidateReviewer:
    """Review complete first-pass candidates without creating new lessons."""

    PROMPT_ID = "playbook_candidate_review"

    def __init__(
        self,
        *,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        max_tokens: int | None = None,
        reasoning_effort: Literal["high", "none"] | None = None,
        provider_request_guard: ProviderRequestGuard | None = None,
    ) -> None:
        self.request_context = request_context
        self.client = llm_client
        # Internal qualification overrides; normal service callers supply none.
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.provider_request_guard = provider_request_guard

    def is_enabled(self) -> bool:
        """Return whether a reviewer prompt version is active."""
        return (
            self.request_context.prompt_manager.get_active_version(self.PROMPT_ID)
            is not None
        )

    @staticmethod
    def _candidate_evidence_units(
        candidates: list[UserPlaybook],
        prompt_context: PlaybookPromptContext,
    ) -> dict[str, list[CandidateEvidenceUnit]]:
        coordinator = current_overflow_coordinator.get()
        if coordinator is not None:
            spans = tuple(
                span
                for chunk in coordinator.require_prepared().chunks
                for span in (*chunk.neighbors, *chunk.spans)
            )
            result: dict[str, list[CandidateEvidenceUnit]] = {}
            for index, candidate in enumerate(candidates, 1):
                candidate_id = f"C{index}"
                record = coordinator.candidate_records.get(
                    coordinator.playbook_key(candidate)
                )
                if record is None:
                    raise PlaybookCandidateEvidenceError(
                        "Candidate lacks verified quotation context"
                    )
                checked = ProtectedRecord().merge(
                    ProtectedRecordUpdate(evidence=list(record.evidence)),
                    visible_spans=spans,
                )
                units = []
                for item in checked.evidence:
                    first = resolve_evidence_span(item.evidence_refs[0], spans)
                    # Additional contradictions/scope context is shown separately;
                    # it cannot silently become this candidate's historical source.
                    if first.interaction_id not in candidate.source_interaction_ids:
                        continue
                    source = prompt_context.evidence_sources.get(
                        f"T{first.source_order}"
                    )
                    units.append(
                        CandidateEvidenceUnit(
                            evidence_id=f"{candidate_id}-E{len(units) + 1}",
                            turn_ref=f"T{first.source_order}",
                            source_span=item.quote,
                            interaction_id=first.interaction_id,
                            role=first.role,
                            request_source=source.request_source if source else "",
                        )
                    )
                if not units:
                    raise PlaybookCandidateEvidenceError(
                        "Candidate lacks quotations from its original source mapping"
                    )
                result[candidate_id] = units
            return result
        by_interaction_id: dict[int, list[Any]] = {}
        for unit in prompt_context.evidence_units.values():
            if unit.interaction_id:
                by_interaction_id.setdefault(unit.interaction_id, []).append(unit)
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]] = {}
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{candidate_index}"
            units: list[CandidateEvidenceUnit] = []
            for interaction_id in candidate.source_interaction_ids:
                mapped_units = by_interaction_id.get(interaction_id)
                if not mapped_units:
                    raise PlaybookCandidateEvidenceError(
                        f"Reviewer cannot map {candidate_id} evidence to a local turn"
                    )
                # Exact persisted interaction IDs are the durable evidence
                # identity. Rebuild their bounded visible spans from storage;
                # do not require every span to survive in the playbook's
                # combined ``source_span``, which may be truncated or may have
                # been assembled from multiple rows during consolidation.
                for mapped_unit in mapped_units:
                    if not mapped_unit.source_span:
                        continue
                    if any(
                        unit.turn_ref == mapped_unit.turn_ref
                        and unit.source_span == mapped_unit.source_span
                        for unit in units
                    ):
                        continue
                    units.append(
                        CandidateEvidenceUnit(
                            evidence_id=f"{candidate_id}-E{len(units) + 1}",
                            turn_ref=mapped_unit.turn_ref,
                            source_span=mapped_unit.source_span,
                            interaction_id=interaction_id,
                            role=mapped_unit.role,
                            request_source=mapped_unit.request_source,
                        )
                    )
            if not units:
                raise PlaybookCandidateEvidenceError(
                    f"Reviewer received {candidate_id} without evidence"
                )
            units_by_candidate[candidate_id] = units
        return units_by_candidate

    @staticmethod
    def _format_candidates(
        candidates: list[UserPlaybook],
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]],
    ) -> str:
        sections: list[str] = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{candidate_index}"
            evidence_lines = "\n".join(
                f'- [{unit.evidence_id}] [{unit.turn_ref}] "{unit.source_span}"'
                for unit in units_by_candidate[candidate_id]
            )
            sections.append(
                "\n".join(
                    (
                        f"[{candidate_id}]",
                        f"Evidence kind: {candidate.reader_angle or 'unknown'}",
                        f"Trigger: {candidate.trigger or ''}",
                        f"Rationale: {candidate.rationale or ''}",
                        f"Content: {candidate.content}",
                        "Validated evidence units:",
                        evidence_lines,
                    )
                )
            )
        return "\n\n".join(sections)

    @staticmethod
    def _format_existing(existing_playbooks: list[UserPlaybook]) -> str:
        from reflexio.server.services.playbook.components.overflow_evidence import (
            remember_historical_playbooks,
        )

        remember_historical_playbooks(existing_playbooks)
        if current_overflow_coordinator.get() is not None:
            return (
                "Historical comparison runs separately after this review. "
                "Review only this candidate's grounding in its own original evidence; "
                "do not reject it because another stored learning might cover it."
            )
        if not existing_playbooks:
            return "(none)"
        return "\n".join(
            " | ".join(
                (
                    f"[X{index}]",
                    f"Trigger: {playbook.trigger or ''}",
                    f"Rationale: {playbook.rationale or ''}",
                    f"Content: {playbook.content}",
                )
            )
            for index, playbook in enumerate(existing_playbooks, start=1)
        )

    @staticmethod
    def _validation_errors(
        output: object,
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]],
    ) -> tuple[str, ...]:
        if not isinstance(output, PlaybookCandidateReviewOutput):
            return ("review output has the wrong structured type",)
        expected_ids = set(units_by_candidate)
        seen_ids: set[str] = set()
        errors: list[str] = []
        for index, decision in enumerate(output.decisions):
            candidate_id = decision.candidate_id.strip()
            if candidate_id not in expected_ids:
                errors.append(f"decisions[{index}] has unknown candidate_id")
                continue
            if candidate_id in seen_ids:
                errors.append(f"decisions[{index}] repeats {candidate_id}")
                continue
            seen_ids.add(candidate_id)
            allowed_evidence = {
                unit.evidence_id for unit in units_by_candidate[candidate_id]
            }
            cited_evidence = set(decision.evidence_ids)
            if not cited_evidence.issubset(allowed_evidence):
                errors.append(f"decisions[{index}] introduces unknown evidence")
            if decision.decision == "accept":
                if cited_evidence != allowed_evidence:
                    errors.append(
                        f"decisions[{index}] accept must retain all evidence units"
                    )
                if decision.revision is not None:
                    errors.append(
                        f"decisions[{index}] accept must not include revision"
                    )
            elif decision.decision == "revise":
                if not cited_evidence:
                    errors.append(
                        f"decisions[{index}] revise requires retained evidence"
                    )
                if decision.revision is None:
                    errors.append(f"decisions[{index}] revise requires revision")
                elif contains_call_local_turn_ref(
                    decision.revision.content,
                    decision.revision.trigger,
                    decision.revision.rationale,
                ):
                    errors.append(
                        f"decisions[{index}] revision contains call-local turn label"
                    )
            else:
                if cited_evidence:
                    errors.append(
                        f"decisions[{index}] reject must not retain evidence units"
                    )
                if decision.revision is not None:
                    errors.append(
                        f"decisions[{index}] reject must not include revision"
                    )
        missing_ids = expected_ids - seen_ids
        if missing_ids:
            errors.append(
                "missing candidate decisions: " + ",".join(sorted(missing_ids))
            )
        return tuple(errors)

    @staticmethod
    def _evidence_first_validation_errors(
        output: object,
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]],
    ) -> tuple[str, ...]:
        if not isinstance(output, EvidenceFirstReviewOutput):
            return ("review output has the wrong evidence-first structured type",)
        decisions = PlaybookCandidateReviewOutput(decisions=output.decisions)
        errors = list(
            PlaybookCandidateReviewer._validation_errors(decisions, units_by_candidate)
        )
        supports: dict[str, CandidateSupportingEvidence] = {}
        for index, support in enumerate(output.supporting_evidence):
            candidate_id = support.candidate_id.strip()
            if candidate_id not in units_by_candidate or candidate_id in supports:
                errors.append(
                    f"supporting_evidence[{index}] has unknown or repeated candidate"
                )
                continue
            supports[candidate_id] = support
            own_units = {
                unit.evidence_id: unit for unit in units_by_candidate[candidate_id]
            }
            for excerpt in support.excerpts:
                unit = own_units.get(excerpt.evidence_id)
                if (
                    unit is None
                    or not excerpt.text.strip()
                    or excerpt.text not in unit.source_span
                ):
                    # Never include quoted customer text in errors/repair logs.
                    errors.append(
                        f"supporting_evidence[{index}] has invalid own-source excerpt"
                    )
        for index, decision in enumerate(output.decisions):
            support = supports.get(decision.candidate_id.strip())
            if decision.decision == "reject":
                if support is not None:
                    errors.append(
                        f"decisions[{index}] reject must not retain supporting excerpts"
                    )
            elif support is None:
                errors.append(
                    f"decisions[{index}] survivor requires supporting excerpts"
                )
            elif not {excerpt.evidence_id for excerpt in support.excerpts}.issubset(
                set(decision.evidence_ids)
            ):
                errors.append(
                    f"decisions[{index}] supporting excerpts must belong to final retained evidence"
                )
        return tuple(errors)

    @staticmethod
    def _apply_decisions(
        candidates: list[UserPlaybook],
        output: PlaybookCandidateReviewOutput,
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]],
    ) -> list[UserPlaybook]:
        decisions = {
            decision.candidate_id.strip(): decision for decision in output.decisions
        }
        survivors: list[UserPlaybook] = []
        coordinator = current_overflow_coordinator.get()

        def add_survivor(
            original: UserPlaybook, survivor: UserPlaybook, *, revised: bool
        ) -> None:
            if coordinator is not None:
                old_key, new_key = (
                    coordinator.playbook_key(original),
                    coordinator.playbook_key(survivor),
                )
                coordinator.candidate_models[new_key] = (
                    coordinator.review_models.get(old_key)
                    if revised
                    else coordinator.candidate_models.get(old_key)
                )
                coordinator.candidate_records[new_key] = coordinator.candidate_records[
                    old_key
                ]
                coordinator.candidate_evidence_refs[new_key] = (
                    coordinator.candidate_evidence_refs.get(old_key, ())
                )
                if old_key in coordinator.candidate_historical_matches:
                    coordinator.candidate_historical_matches[new_key] = (
                        coordinator.candidate_historical_matches[old_key]
                    )
            survivors.append(survivor)

        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{candidate_index}"
            decision = decisions[candidate_id]
            if decision.decision == "reject":
                continue
            if decision.decision == "accept":
                updates: dict[str, Any] = {
                    "notes": _append_review_note(
                        candidate.notes,
                        "accepted",
                        decision.reason_code,
                        decision.reason,
                    )
                }
                if coordinator is not None:
                    updates["source_span"] = "\n\n".join(
                        unit.source_span for unit in units_by_candidate[candidate_id]
                    )
                add_survivor(
                    candidate, candidate.model_copy(update=updates), revised=False
                )
                continue

            selected = [
                unit
                for unit in units_by_candidate[candidate_id]
                if unit.evidence_id in decision.evidence_ids
            ]
            source_ids: list[int] = []
            spans: list[str] = []
            for unit in selected:
                if unit.interaction_id not in source_ids:
                    source_ids.append(unit.interaction_id)
                if unit.source_span not in spans:
                    spans.append(unit.source_span)
            revision = decision.revision
            if revision is None:  # Guarded by structured semantic validation.
                raise ValueError(f"Reviewer omitted revision for {candidate_id}")
            add_survivor(
                candidate,
                candidate.model_copy(
                    update={
                        "content": revision.content,
                        "trigger": revision.trigger,
                        "rationale": revision.rationale,
                        "source_interaction_ids": source_ids,
                        "source_span": "\n\n".join(spans),
                        "notes": _append_review_note(
                            candidate.notes,
                            "revised",
                            decision.reason_code,
                            decision.reason,
                        ),
                    }
                ),
                revised=True,
            )
        return survivors

    def decide(
        self,
        *,
        candidates: list[UserPlaybook],
        request_interaction_data_models: list[RequestInteractionDataModel],
        existing_playbooks: list[UserPlaybook],
        agent_context: str,
        playbook_definition: str,
        tool_context: str,
    ) -> PlaybookReviewOutcome:
        """Run one fresh same-model review and return its validated decisions.

        The returned outcome carries the resolved evidence units alongside the
        decisions so ``apply_decisions`` never has to rebuild the prompt context
        (an O(window) walk) a second time.
        """
        if not candidates:
            return PlaybookReviewOutcome(
                output=PlaybookCandidateReviewOutput(decisions=[]),
                units_by_candidate={},
            )
        coordinator = current_overflow_coordinator.get()
        if coordinator is not None and len(candidates) > 1:
            decisions = []
            units: dict[str, list[CandidateEvidenceUnit]] = {}
            supporting = []
            for index, candidate in enumerate(candidates, 1):
                child = self.decide(
                    candidates=[candidate],
                    request_interaction_data_models=request_interaction_data_models,
                    existing_playbooks=existing_playbooks,
                    agent_context=agent_context,
                    playbook_definition=playbook_definition,
                    tool_context=tool_context,
                )
                target = f"C{index}"
                names = {
                    unit.evidence_id: unit.evidence_id.replace("C1-", f"{target}-", 1)
                    for unit in child.units_by_candidate["C1"]
                }
                decisions.append(
                    child.output.decisions[0].model_copy(
                        update={
                            "candidate_id": target,
                            "evidence_ids": [
                                names[ref]
                                for ref in child.output.decisions[0].evidence_ids
                            ],
                        }
                    )
                )
                units[target] = [
                    replace(unit, evidence_id=names[unit.evidence_id])
                    for unit in child.units_by_candidate["C1"]
                ]
                supporting.extend(
                    proof.model_copy(
                        update={
                            "candidate_id": target,
                            "excerpts": [
                                quote.model_copy(
                                    update={"evidence_id": names[quote.evidence_id]}
                                )
                                for quote in proof.excerpts
                            ],
                        }
                    )
                    for proof in child.supporting_evidence
                )
            return PlaybookReviewOutcome(
                PlaybookCandidateReviewOutput(decisions=decisions), units, supporting
            )
        scope_messages: list[dict[str, str]] = []
        if coordinator is not None:
            from reflexio.server.services.playbook.components.overflow_evidence import (
                EVIDENCE_SCOPE_POLICY,
            )

            scope_messages.append({"role": "system", "content": EVIDENCE_SCOPE_POLICY})
            candidate = candidates[0]
            record = coordinator.candidate_records.get(
                coordinator.playbook_key(candidate)
            )
            if record is None:
                raise PlaybookCandidateEvidenceError(
                    "Candidate lacks its verified evidence record"
                )
            requests = {
                group.request.request_id: group.request
                for group in request_interaction_data_models
            }
            sources = {
                f"T{span.source_order}": PlaybookEvidenceSource(
                    interaction_id=span.interaction_id,
                    request_id=span.request_id,
                    evidence_texts=(),
                    role=span.role,
                    request_source=requests[span.request_id].source or "",
                )
                for chunk in coordinator.require_prepared().chunks
                for span in (*chunk.neighbors, *chunk.spans)
            }
            prompt_context = PlaybookPromptContext(
                json.dumps(
                    {
                        "protected_record": coordinator.record.model_dump(mode="json"),
                        "candidate_context": record.model_dump(mode="json"),
                        "human_answers": coordinator.checkpoint.get(
                            "answer_witnesses", {}
                        ),
                        "scope": "Additional context does not alter the candidate's original source mapping.",
                        **protected_source_origins(
                            coordinator.require_prepared(), coordinator.record, record
                        ),
                    },
                    ensure_ascii=False,
                ),
                sources,
                {},
            )
            agent_context = coordinator.agent_context
        else:
            prompt_context = build_playbook_prompt_context(
                request_interaction_data_models,
                expert=False,
                # The service invokes the reviewer only for strict normal extraction,
                # whose validated evidence units are keyed by these local labels.
                label_turns=True,
            )
        units_by_candidate = self._candidate_evidence_units(candidates, prompt_context)
        version = self.request_context.prompt_manager.get_active_version(self.PROMPT_ID)
        evidence_first = version in {
            "1.14.0",
            "1.15.0",
            "1.16.0",
            "1.17.0",
            "1.18.0",
            "1.19.0",
            "1.20.0",
        }
        variables = {
            "agent_context_prompt": agent_context,
            "playbook_definition": playbook_definition,
            "tool_context": tool_context or "(none)",
            "interaction_context": prompt_context.text,
            "artifact_availability": (
                "Only visible interaction and tool-result text in the chronology "
                "is available. Unquoted artifact contents and downstream user "
                "outcomes are unavailable."
            ),
            "candidates": self._format_candidates(candidates, units_by_candidate),
            "existing_playbooks": self._format_existing(existing_playbooks),
        }
        prompt = self.request_context.prompt_manager.render_prompt(
            self.PROMPT_ID, variables
        )
        messages = [*scope_messages, {"role": "user", "content": prompt}]
        response_format: (
            type[PlaybookCandidateReviewOutput] | type[EvidenceFirstReviewOutput]
        ) = PlaybookCandidateReviewOutput
        validator = self._validation_errors
        if evidence_first:
            # Context and transcript text never enter the trusted policy message.
            payload: dict[str, Any] = dict(variables)
            payload["own_evidence"] = {
                candidate_id: [
                    {
                        "id": unit.evidence_id,
                        "role": unit.role,
                        "request_source": unit.request_source,
                        "text": unit.source_span,
                    }
                    for unit in units
                ]
                for candidate_id, units in units_by_candidate.items()
            }
            messages = [
                {"role": "system", "content": prompt},
                *scope_messages,
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ]
            response_format = EvidenceFirstReviewOutput
            validator = self._evidence_first_validation_errors
        inference_options: dict[str, Any] = {}
        if self.max_tokens is not None:
            inference_options["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            # Z.ai's reasoning_effort is absent from the installed LiteLLM
            # standard-param allowlist. extra_body preserves it on the wire.
            inference_options["extra_body"] = {
                "reasoning_effort": self.reasoning_effort,
                "thinking": {
                    "type": "disabled" if self.reasoning_effort == "none" else "enabled"
                },
            }
        if self.provider_request_guard is not None:
            inference_options["provider_request_guard"] = self.provider_request_guard
        options = dict(
            model_role=ModelRole.GENERATION,
            max_retries=0,
            response_format=response_format,
            parse_structured_output=True,
            structured_output_validator=lambda value: validator(
                value, units_by_candidate
            ),
            **inference_options,
        )
        review_key = None
        review_provenance = None
        if coordinator is not None:
            review_key = coordinator.step_key(
                "review-v1",
                {
                    "messages": messages,
                    "schema": response_format.model_json_schema(),
                    "max_tokens": self.max_tokens,
                    "reasoning_effort": self.reasoning_effort,
                },
            )
            saved = coordinator.completed_step(review_key)
            if saved is None:
                coordinator.require_remaining_work(1)
                result = self.client.generate_chat_response_with_provenance(
                    messages, **options
                )
                output, review_provenance = result.value, result.provenance
            else:
                output = response_format.model_validate(saved["output"])
                review_provenance = ModelProvenance(**saved["model_provenance"])
        else:
            output = self.client.generate_chat_response(messages, **options)
        raw_output = output
        supporting_evidence: list[CandidateSupportingEvidence] = []
        if evidence_first:
            errors = validator(output, units_by_candidate)
            if errors:
                raise ValueError("; ".join(errors))
            if not isinstance(output, EvidenceFirstReviewOutput):
                raise ValueError(
                    "Playbook reviewer returned the wrong evidence-first type"
                )
            supporting_evidence = output.supporting_evidence
            output = PlaybookCandidateReviewOutput(decisions=output.decisions)
        if not isinstance(output, PlaybookCandidateReviewOutput):
            raise ValueError("Playbook reviewer returned the wrong structured type")
        errors = self._validation_errors(output, units_by_candidate)
        if errors:
            raise ValueError("; ".join(errors))
        if (
            coordinator is not None
            and review_key is not None
            and review_provenance is not None
        ):
            if not isinstance(raw_output, BaseModel):
                raise ValueError("Reviewer lacks validated structured output")
            coordinator.complete_step(
                review_key, raw_output.model_dump(mode="json"), review_provenance
            )
            coordinator.review_models[coordinator.playbook_key(candidates[0])] = (
                review_provenance
            )
        counts = dict.fromkeys(("accept", "revise", "reject"), 0)
        reason_counts: dict[str, int] = {}
        for decision in output.decisions:
            counts[decision.decision] += 1
            reason_counts[decision.reason_code] = (
                reason_counts.get(decision.reason_code, 0) + 1
            )
        logger.info(
            "event=playbook_candidate_review_complete accepted=%d revised=%d "
            "rejected=%d reason_codes=%s",
            counts["accept"],
            counts["revise"],
            counts["reject"],
            ",".join(
                f"{reason_code}:{count}"
                for reason_code, count in sorted(reason_counts.items())
            ),
        )
        return PlaybookReviewOutcome(
            output=output,
            units_by_candidate=units_by_candidate,
            supporting_evidence=supporting_evidence,
        )

    def review(
        self,
        *,
        candidates: list[UserPlaybook],
        request_interaction_data_models: list[RequestInteractionDataModel],
        existing_playbooks: list[UserPlaybook],
        agent_context: str,
        playbook_definition: str,
        tool_context: str,
    ) -> list[UserPlaybook]:
        """Run one fresh same-model review and return accepted/revised survivors."""
        if not candidates:
            return []
        outcome = self.decide(
            candidates=candidates,
            request_interaction_data_models=request_interaction_data_models,
            existing_playbooks=existing_playbooks,
            agent_context=agent_context,
            playbook_definition=playbook_definition,
            tool_context=tool_context,
        )
        return self.apply_decisions(candidates=candidates, outcome=outcome)

    def apply_decisions(
        self,
        *,
        candidates: list[UserPlaybook],
        outcome: PlaybookReviewOutcome,
    ) -> list[UserPlaybook]:
        """Apply already-validated decisions without making another model call."""
        return self._apply_decisions(
            candidates, outcome.output, outcome.units_by_candidate
        )
