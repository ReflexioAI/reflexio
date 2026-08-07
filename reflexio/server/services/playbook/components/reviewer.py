"""Evidence-preserving second-pass review for normal user playbook candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from reflexio.models.api_schema.domain.entities import UserPlaybook
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.structured_output import normalize_provider_value
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.services.playbook.playbook_evidence import (
    contains_call_local_turn_ref,
)
from reflexio.server.services.playbook.playbook_service_utils import (
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
#: :meth:`CandidateReviewDecision.fatal_reason_codes_must_reject`.
_FATAL_REASON_CODES = frozenset({"absence_inference", "not_agent_decision"})


class CandidateReviewDecision(BaseModel):
    """One exactly-accounted-for candidate decision."""

    candidate_id: str = Field(alias="id", min_length=1)
    decision: Literal["accept", "revise", "reject"]
    reason_code: Literal[
        "grounded_useful",
        "unsupported_evidence",
        "generic",
        "speculative",
        "unsupported_causality",
        "unseen_artifact",
        "redundant",
        "late_trigger",
        "compound",
        "internal_status",
        "absence_inference",
        "not_agent_decision",
    ]
    evidence_ids: list[str] = Field(default_factory=list)
    revision: CandidateRevision | None = None
    reason: str | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def fatal_reason_codes_must_reject(self) -> CandidateReviewDecision:
        """A defect with no salvageable core cannot be narrowed into a survivor.

        The prompt instructs the reviewer to reject these outright, but a prompt
        instruction is not an invariant: a response pairing one of these codes
        with ``accept`` or ``revise`` otherwise validates cleanly, and revision
        would launder the defect into tidier prose rather than removing it. That
        pairing has been observed in practice, so enforce it structurally rather
        than by wording alone.

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


@dataclass(frozen=True)
class CandidateEvidenceUnit:
    """One validated evidence span exposed through a call-local unit id."""

    evidence_id: str
    turn_ref: str
    source_span: str
    interaction_id: int


@dataclass(frozen=True)
class PlaybookReviewOutcome:
    """A validated review result plus the evidence units it was decided against.

    Carrying the units keeps ``decide`` -> ``apply_decisions`` a single pass:
    rebuilding them means re-walking the whole interaction window.
    """

    output: PlaybookCandidateReviewOutput
    units_by_candidate: dict[str, list[CandidateEvidenceUnit]]


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
    ) -> None:
        self.request_context = request_context
        self.client = llm_client

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
    def _apply_decisions(
        candidates: list[UserPlaybook],
        output: PlaybookCandidateReviewOutput,
        units_by_candidate: dict[str, list[CandidateEvidenceUnit]],
    ) -> list[UserPlaybook]:
        decisions = {
            decision.candidate_id.strip(): decision for decision in output.decisions
        }
        survivors: list[UserPlaybook] = []
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_id = f"C{candidate_index}"
            decision = decisions[candidate_id]
            if decision.decision == "reject":
                continue
            if decision.decision == "accept":
                survivors.append(
                    candidate.model_copy(
                        update={
                            "notes": _append_review_note(
                                candidate.notes,
                                "accepted",
                                decision.reason_code,
                                decision.reason,
                            )
                        }
                    )
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
            survivors.append(
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
                )
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
        prompt_context = build_playbook_prompt_context(
            request_interaction_data_models,
            expert=False,
            # The service invokes the reviewer only for strict normal extraction,
            # whose validated evidence units are keyed by these local labels.
            label_turns=True,
        )
        units_by_candidate = self._candidate_evidence_units(candidates, prompt_context)
        prompt = self.request_context.prompt_manager.render_prompt(
            self.PROMPT_ID,
            {
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
            },
        )
        output = self.client.generate_chat_response(
            [{"role": "user", "content": prompt}],
            model_role=ModelRole.GENERATION,
            max_retries=0,
            response_format=PlaybookCandidateReviewOutput,
            parse_structured_output=True,
            structured_output_validator=lambda value: self._validation_errors(
                value, units_by_candidate
            ),
        )
        if not isinstance(output, PlaybookCandidateReviewOutput):
            raise ValueError("Playbook reviewer returned the wrong structured type")
        errors = self._validation_errors(output, units_by_candidate)
        if errors:
            raise ValueError("; ".join(errors))
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
            output=output, units_by_candidate=units_by_candidate
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
