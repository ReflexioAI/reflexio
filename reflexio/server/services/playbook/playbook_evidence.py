"""Deterministic evidence validation and output-contract selection for extraction.

These are pure functions over the extraction schemas and the call-local evidence
map — they hold no extractor state. Keeping them here (rather than as private
methods on ``PlaybookExtractor``) lets the fresh-run and resume paths select the
same contract and run the same validator without either reaching into the
other's privates.

The guards implemented here are only those provable without understanding the
domain: schema completeness, verbatim-span resolution, and provenance. Semantic
judgement (clause support, causality, atomicity, usefulness) belongs to the
extraction prompt and the second-pass reviewer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.playbook_service_utils import (
    PlaybookEvidenceSource,
    PlaybookEvidenceUnit,
    StructuredExtractedPlaybookContent,
    StructuredExtractedPlaybookList,
    StructuredPlaybookContent,
    StructuredPlaybookList,
    StructuredReferencedExtractedPlaybookContent,
    StructuredReferencedExtractedPlaybookList,
    uses_evidence_grounded_extraction,
)

logger = logging.getLogger(__name__)

_TYPOGRAPHIC_QUOTE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
    }
)


def _is_direct_user_preference_source(source: PlaybookEvidenceSource) -> bool:
    """Return whether a turn can independently establish a durable preference.

    A durable preference must come from the user's own words: an agent turn
    restating what it believes the user wants is not independent evidence.
    """
    return source.role.strip().casefold() == "user"


def contains_call_local_turn_ref(*values: str | None) -> bool:
    """Return whether persisted prose contains a call-local turn label.

    ``T#`` labels are meaningful only inside one extraction call. Evidence
    objects carry those labels long enough to resolve real provenance, but a
    persisted trigger, rationale, or instruction must remain self-contained.
    """
    prose = " ".join(value for value in values if value)
    return re.search(r"\bT\d+\b", prose, flags=re.IGNORECASE) is not None


def resolve_verbatim_source_span(
    source_span: str, evidence_texts: tuple[str, ...]
) -> str | None:
    """Resolve quote typography to one exact source substring, never fuzzily.

    Args:
        source_span (str): The excerpt the model claims it quoted.
        evidence_texts (tuple[str, ...]): Every quotable text on the cited turn.

    Returns:
        str | None: The exact substring of the source, or None when the span
        does not resolve or resolves ambiguously to more than one position.
    """
    for source_text in evidence_texts:
        if source_span in source_text:
            return source_span

    normalized_span = source_span.translate(_TYPOGRAPHIC_QUOTE_TRANSLATION)
    matches: list[str] = []
    for source_text in evidence_texts:
        normalized_source = source_text.translate(_TYPOGRAPHIC_QUOTE_TRANSLATION)
        start = normalized_source.find(normalized_span)
        while start >= 0:
            matches.append(source_text[start : start + len(source_span)])
            start = normalized_source.find(normalized_span, start + 1)
    return matches[0] if len(matches) == 1 else None


def as_playbook_content(
    entry: (
        StructuredPlaybookContent
        | StructuredReferencedExtractedPlaybookContent
        | StructuredExtractedPlaybookContent
    ),
) -> StructuredPlaybookContent:
    """Widen a strict first-pass candidate to the shared playbook-content model.

    ``StructuredReferencedExtractedPlaybookContent`` is a deliberately narrow contract
    sent to the provider; the rest of the pipeline reasons over the richer
    ``StructuredPlaybookContent``. Converting once here keeps every downstream
    helper single-typed.
    """
    if isinstance(
        entry,
        StructuredReferencedExtractedPlaybookContent
        | StructuredExtractedPlaybookContent,
    ):
        return StructuredPlaybookContent.model_validate(entry.model_dump(mode="python"))
    return entry


def _referenced_evidence_rejection_reason(
    entry: StructuredPlaybookContent,
    evidence_units: dict[str, PlaybookEvidenceUnit] | None,
) -> str | None:
    if evidence_units is None:
        return "missing_evidence_unit_map"
    seen_refs: set[str] = set()
    for raw_ref in entry.evidence_refs:
        evidence_ref = raw_ref.strip()
        if not evidence_ref:
            return "empty_evidence_ref"
        if evidence_ref in seen_refs:
            return "duplicate_evidence_ref"
        seen_refs.add(evidence_ref)
        unit = evidence_units.get(evidence_ref)
        if unit is None:
            return "unknown_evidence_ref"
        if not unit.interaction_id:
            return "unpersistable_source"
        if not unit.source_span:
            return "empty_source_span"
        if entry.evidence_kind == "preference" and unit.role.strip().casefold() != (
            "user"
        ):
            return "preference_without_direct_user_evidence"
    return None


def _span_evidence_rejection_reason(
    entry: StructuredPlaybookContent,
    evidence_sources: dict[str, PlaybookEvidenceSource],
) -> str | None:
    for evidence in entry.evidence:
        turn_ref = evidence.turn_ref.strip()
        source_span = evidence.source_span.strip()
        source = evidence_sources.get(turn_ref)
        if source is None:
            return "unknown_turn_ref"
        if not source.interaction_id:
            return "unpersistable_source"
        if not source_span:
            return "empty_source_span"
        if resolve_verbatim_source_span(source_span, source.evidence_texts) is None:
            return "non_verbatim_source_span"
        if entry.evidence_kind == "preference" and not (
            _is_direct_user_preference_source(source)
        ):
            return "preference_without_direct_user_evidence"
    return None


def candidate_rejection_reason(
    entry: (
        StructuredPlaybookContent
        | StructuredReferencedExtractedPlaybookContent
        | StructuredExtractedPlaybookContent
    ),
    evidence_sources: dict[str, PlaybookEvidenceSource],
    evidence_units: dict[str, PlaybookEvidenceUnit] | None = None,
) -> str | None:
    """Return a stable rejection code for deterministically invalid evidence.

    Semantic usefulness, clause support, causality, and atomicity cannot be
    proven with surface-word heuristics. The second-pass reviewer owns those
    judgments; this function enforces only schema and provenance invariants.
    """
    entry = as_playbook_content(entry)
    if not entry.content or not entry.content.strip():
        return "missing_content"
    if not entry.trigger or not entry.trigger.strip():
        return "missing_trigger"
    if not entry.rationale or not entry.rationale.strip():
        return "missing_rationale"
    if entry.evidence_kind is None:
        return "missing_evidence_kind"
    if not entry.evidence and not entry.evidence_refs:
        return "missing_evidence"
    if contains_call_local_turn_ref(entry.content, entry.trigger, entry.rationale):
        return "turn_reference_in_persisted_prose"
    if entry.evidence_refs:
        return _referenced_evidence_rejection_reason(entry, evidence_units)
    return _span_evidence_rejection_reason(entry, evidence_sources)


def strict_output_validation_errors(
    output: object,
    evidence_sources: dict[str, PlaybookEvidenceSource],
    evidence_units: dict[str, PlaybookEvidenceUnit] | None = None,
) -> tuple[str, ...]:
    """Reject a strict batch only when no candidate in it is usable.

    Individually ungrounded candidates are dropped downstream, so a single bad
    sibling must not cost the whole batch a repair turn (and, if repair also
    fails, the entire extraction). A batch is worth one corrective turn only
    when every candidate it returned is ungrounded. An empty list is a valid
    answer — "no lesson qualifies" — and never triggers repair.

    Args:
        output: The parsed structured response under validation.
        evidence_sources: Prompt-local turn refs mapped to persisted sources.

    Returns:
        tuple[str, ...]: One ``playbooks[i]: reason`` entry per candidate when
        the whole batch is ungrounded, otherwise an empty tuple.
    """
    if not isinstance(
        output,
        StructuredReferencedExtractedPlaybookList | StructuredExtractedPlaybookList,
    ):
        return ()
    if not output.playbooks:
        return ()

    errors = [
        f"playbooks[{index}]: {reason}"
        for index, candidate in enumerate(output.playbooks)
        if (
            reason := candidate_rejection_reason(
                candidate, evidence_sources, evidence_units
            )
        )
        is not None
    ]
    if len(errors) < len(output.playbooks):
        return ()
    logger.warning(
        "event=playbook_output_semantic_validation_failed errors=%s",
        " | ".join(errors),
    )
    return tuple(errors)


@dataclass(frozen=True)
class PlaybookExtractionContract:
    """The output schema and bounded-repair validator for one extraction call.

    Attributes:
        strict (bool): Whether the active prompt requires evidence-grounded
            candidates. Callers use this to decide whether to resolve source IDs
            from cited turns or fall back to the whole-window ID list.
        schema (type[BaseModel]): The structured-output schema to request.
        validator: Opts the call into the client's one corrective same-model
            repair turn, or None for the legacy contract.
    """

    strict: bool
    schema: type[BaseModel]
    validator: Callable[[BaseModel], Sequence[str]] | None


def build_playbook_extraction_contract(
    prompt_manager: PromptManager,
    *,
    expert: bool,
    evidence_sources: dict[str, PlaybookEvidenceSource],
    evidence_units: dict[str, PlaybookEvidenceUnit] | None = None,
    strict_override: bool | None = None,
    evidence_references_override: bool | None = None,
) -> PlaybookExtractionContract:
    """Select the extraction output contract for the active prompt version.

    Single source of truth for the fresh-run and resume paths, so a resumed run
    can never request a schema — or skip a validator — that a fresh run would
    not have used.

    Args:
        prompt_manager (PromptManager): Resolves the active prompt version.
        expert (bool): Whether the expert-content extraction path is in use.
        evidence_sources: Prompt-local turn refs mapped to persisted sources,
            bound into the validator.
        strict_override: Stored run-time contract selection for durable resume.
            ``None`` selects from the currently active prompt version.

    Returns:
        PlaybookExtractionContract: The schema/validator pair plus the strict flag.
    """
    strict = (
        uses_evidence_grounded_extraction(prompt_manager, expert=expert)
        if strict_override is None
        else strict_override
    )
    if not strict:
        return PlaybookExtractionContract(
            strict=False, schema=StructuredPlaybookList, validator=None
        )
    uses_evidence_references = (
        not expert
        if evidence_references_override is None
        else evidence_references_override
    )
    schema: type[BaseModel] = (
        StructuredReferencedExtractedPlaybookList
        if uses_evidence_references
        else StructuredExtractedPlaybookList
    )
    return PlaybookExtractionContract(
        strict=True,
        schema=schema,
        validator=lambda output: strict_output_validation_errors(
            output, evidence_sources, evidence_units
        ),
    )
