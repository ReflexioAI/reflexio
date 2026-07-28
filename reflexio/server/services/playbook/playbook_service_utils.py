from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reflexio.models.api_schema.domain.entities import Interaction, UserPlaybook
from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.structured_output import (
    StrictStructuredOutput,
    normalize_provider_keys,
    normalize_provider_value,
)
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.playbook.playbook_service_constants import (
    PlaybookServiceConstants,
)
from reflexio.server.services.service_utils import (
    MessageConstructionConfig,
    PromptConfig,
    construct_messages_from_interactions,
    extract_interactions_from_request_interaction_data_models,
    format_interactions_to_history_string,
    visible_interaction_evidence_texts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybookEvidenceSource:
    """One prompt-local turn and the persisted interaction it represents."""

    interaction_id: int
    request_id: str
    evidence_texts: tuple[str, ...]
    role: str = ""
    request_source: str = ""


@dataclass(frozen=True)
class PlaybookEvidenceUnit:
    """One prompt-visible turn addressed without model-authored quoting."""

    evidence_ref: str
    turn_ref: str
    source_span: str
    interaction_id: int
    request_id: str
    role: str = ""
    request_source: str = ""


@dataclass(frozen=True)
class PlaybookPromptContext:
    """Rendered extraction context plus its non-LLM provenance mapping."""

    text: str
    evidence_sources: dict[str, PlaybookEvidenceSource]
    evidence_units: dict[str, PlaybookEvidenceUnit]


class StructuredPlaybookEvidence(BaseModel):
    """Verbatim evidence attached to one extracted playbook candidate."""

    turn_ref: str = Field(
        min_length=1,
        description="Prompt-local turn label such as T1; never a database identifier",
    )
    source_span: str = Field(
        min_length=1,
        description="Non-empty verbatim excerpt from the referenced prompt turn",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )


# Harmless separator/short-form variants structured LLMs emit for the
# evidence-kind tag. Shared so the strict first-pass schema and the legacy
# schema can never drift into accepting different spellings.
_EVIDENCE_KIND_ALIASES = {
    "expert": "expert-gap",
    "failure": "observed-failure",
    "observed-fix": "observed-failure",
    "rejection": "rejected-approach",
    "success": "verified-success",
}


def normalize_evidence_kind_value(value: object) -> object:
    """Canonicalize an evidence-kind tag without changing its meaning.

    Args:
        value (object): Raw field value straight from the parsed LLM output.

    Returns:
        object: The canonical evidence-kind string, or the value unchanged when
        it is not a string (so Pydantic reports the real type error).
    """
    # Evidence kinds are hyphenated ("observed-failure"), so "-" is canonical here.
    return normalize_provider_value(value, _EVIDENCE_KIND_ALIASES, separator="-")


def is_evidence_validated(playbook: UserPlaybook) -> bool:
    """Return whether a row carries complete first-pass evidence provenance.

    A row is evidence-validated only when extraction resolved it to at least one
    persisted interaction, kept the verbatim span it was drawn from, and tagged
    the evidence class. Single source of truth so the review gate and the
    consolidation shortcuts can never disagree about what "grounded" means.

    Args:
        playbook (UserPlaybook): The candidate or stored row to test.

    Returns:
        bool: True when source IDs, span, and evidence-kind tag are all present.
    """
    return bool(
        playbook.source_interaction_ids
        and playbook.source_span
        and playbook.reader_angle
    )


def _normalized_playbook_key(playbook: UserPlaybook) -> tuple[str, str]:
    """Return the same-batch duplicate key for a user playbook."""
    return (
        (playbook.trigger or "").strip().casefold(),
        playbook.content.strip().casefold(),
    )


def dedupe_and_drop_empty(playbooks: list[UserPlaybook]) -> list[UserPlaybook]:
    """Drop empty user playbooks and collapse exact same-batch duplicates.

    This is intentionally deterministic and local to one extraction batch. The
    LLM-backed reviewer and consolidator handle semantic deduplication against
    sibling candidates and stored rows.
    """
    deduped: list[UserPlaybook] = []
    seen: set[tuple[str, str]] = set()
    for playbook in playbooks:
        if not playbook.content.strip():
            continue
        key = _normalized_playbook_key(playbook)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(playbook)
    return deduped


# ===============================
# Pydantic classes for playbook_extraction_main prompt output schema
# ===============================


class StructuredPlaybookContent(BaseModel):
    """
    Structured representation of a single playbook entry from LLM output.

    Field order matters for autoregressive conditioning: rationale is generated
    first, then trigger, then content is synthesized last as a summary.

    The extraction schema intentionally does not ask the model to emit a
    polarity label. The extractor writes action rules or avoidance rules, and
    the service derives the internal :class:`UserPlaybook` polarity from that
    wording before storage.
    """

    rationale: str | None = Field(
        default=None,
        description="The reasoning behind this playbook entry — generated first for autoregressive conditioning",
    )
    evidence_kind: (
        Literal[
            "correction",
            "preference",
            "observed-failure",
            "rejected-approach",
            "expert-gap",
            "verified-success",
        ]
        | None
    ) = Field(
        default=None,
        description="The observed evidence class that qualifies this candidate",
    )
    future_task_class: str | None = Field(
        default=None,
        description="The bounded class of future tasks where the lesson is applicable",
    )
    improvement_mechanism: str | None = Field(
        default=None,
        description="How the grounded action improves success or prevents the observed mistake",
    )
    trigger: str | None = Field(
        default=None,
        description="The condition or context when this rule applies, evaluable at the earliest decision point — do not require facts that only appear later in the conversation",
    )
    content: str | None = Field(
        default=None,
        description="The main actionable content of the playbook entry — what to do or what to avoid",
    )
    source_span: str | None = Field(
        default=None,
        description="Verbatim excerpt from the source that most directly supports this playbook entry",
    )
    notes: str | None = Field(
        default=None,
        description="Free-form extraction notes — confidence, caveats, or alternative readings",
    )
    reader_angle: str | None = Field(
        default=None,
        description="The extraction perspective or reader role that surfaced this entry",
    )
    evidence: list[StructuredPlaybookEvidence] = Field(
        default_factory=list,
        description="One or more prompt-local turn references with verbatim supporting spans",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="One or more prompt-local turn references",
    )
    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def normalize_evidence_kind(cls, value: object) -> object:
        """Accept harmless separator/short-form variants from structured LLMs."""
        return normalize_evidence_kind_value(value)

    @property
    def is_structured(self) -> bool:
        """Check if this playbook entry has both a trigger and content."""
        return bool(
            self.trigger
            and self.trigger.strip()
            and self.content
            and self.content.strip()
        )

    @property
    def has_content(self) -> bool:
        """Check if this output contains actual content."""
        return bool(self.content and self.content.strip())


class StructuredPlaybookList(StrictStructuredOutput):
    """
    Wrapper schema for extracting zero or more playbook entries in a single LLM call.

    The canonical shape is ``{"playbooks": [...]}``. An empty list means the model
    found no valid SOPs in the window. This wrapper exists because OpenAI structured
    output requires a JSON object at the root, so ``list[StructuredPlaybookContent]``
    cannot be used directly as ``response_format``.
    """

    playbooks: list[StructuredPlaybookContent] = Field(
        default_factory=list,
        description="Extracted playbook entries — empty list when no valid SOP was found",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )


class StructuredReferencedExtractedPlaybookContent(BaseModel):
    """Small, strict first-pass contract used only by extraction LLM calls.

    Keep the model-facing contract limited to fields that are necessary to judge
    and persist a candidate.  Additional explanatory fields on the legacy base
    model remain accepted for backwards compatibility, but are not required.
    """

    rationale: str = Field(
        default=...,
        min_length=1,
        description="Observed signal, bounded future task class, and grounded improvement mechanism",
    )
    evidence_kind: Literal[
        "correction",
        "preference",
        "observed-failure",
        "rejected-approach",
        "expert-gap",
        "verified-success",
    ] = Field(default=...)
    trigger: str = Field(
        default=...,
        min_length=1,
        description="Earliest observable future condition, never a later correction or retry request",
    )
    content: str = Field(
        default=...,
        min_length=1,
        description=(
            "One atomic grounded action. For exact duplicates, only prevent the "
            "exact resubmission; do not invent notification, replacement, reuse, "
            "or near-duplicate behavior"
        ),
    )
    evidence_refs: list[str] = Field(
        default=...,
        min_length=1,
        description="Prompt-local turn ids such as T1",
    )

    model_config = ConfigDict(
        # The provider schema still advertises additionalProperties=false, but
        # tolerate and discard legacy explanatory fields from a model that
        # remembers the former contract. Their presence must not erase an
        # otherwise complete, evidence-valid candidate.
        extra="ignore",
        json_schema_extra={"additionalProperties": False},
    )

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def normalize_evidence_kind(cls, value: object) -> object:
        """Accept the same harmless evidence-kind variants as legacy output."""
        return normalize_evidence_kind_value(value)


# Field-name variants providers emit instead of the declared candidate fields.
# Keys are in canonical ``normalize_provider_token`` form (casefolded, "_"-joined).
_EXTRACTED_CANDIDATE_ALIASES = {
    "action": "content",
    "citations": "evidence_refs",
    "evidence": "evidence_refs",
    "evidence_entries": "evidence_refs",
    "future_trigger": "trigger",
    "guidance": "content",
    "kind": "evidence_kind",
    "lesson": "content",
    "observed_signal": "rationale",
    "reason": "rationale",
    "signal_type": "evidence_kind",
    "sources": "evidence_refs",
    "type": "evidence_kind",
    "when": "trigger",
    "why": "rationale",
}


class StructuredReferencedExtractedPlaybookList(StrictStructuredOutput):
    """Minimal required-field extraction response sent to the LLM provider."""

    playbooks: list[StructuredReferencedExtractedPlaybookContent] = Field(
        default=...,
        description="Evidence-complete candidates, or an empty list when none qualify",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate_wrappers_and_keys(cls, value: object) -> object:
        """Accept complete candidates nested under harmless provider wrappers."""
        if not isinstance(value, dict) or not isinstance(value.get("playbooks"), list):
            return value

        required = {
            "rationale",
            "evidence_kind",
            "trigger",
            "content",
            "evidence_refs",
        }

        def normalize_keys(candidate: dict[object, object]) -> dict[str, object]:
            return normalize_provider_keys(candidate, _EXTRACTED_CANDIDATE_ALIASES)

        def find_complete_candidate(
            node: object, depth: int = 0
        ) -> dict[str, object] | None:
            if depth > 4:
                return None
            if isinstance(node, dict):
                candidate = normalize_keys(node)
                if required <= candidate.keys():
                    return candidate
                for nested_value in candidate.values():
                    found = find_complete_candidate(nested_value, depth + 1)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for nested_value in node:
                    found = find_complete_candidate(nested_value, depth + 1)
                    if found is not None:
                        return found
            return None

        normalized_playbooks: list[object] = []
        for raw_candidate in value["playbooks"]:
            if not isinstance(raw_candidate, dict):
                normalized_playbooks.append(raw_candidate)
                continue
            candidate = normalize_keys(raw_candidate)
            if candidate.keys() <= {"evidence_ref", "turn_ref", "source_span"}:
                logger.info(
                    "event=playbook_candidate_rejected reason=evidence_object_without_candidate"
                )
                continue
            if not required <= candidate.keys():
                candidate = find_complete_candidate(candidate) or candidate
            evidence_refs = candidate.get("evidence_refs")
            if isinstance(evidence_refs, list):
                candidate["evidence_refs"] = [
                    (
                        normalize_keys(item).get("evidence_ref")
                        if isinstance(item, dict)
                        else item
                    )
                    for item in evidence_refs
                ]
            normalized_playbooks.append(candidate)

        normalized = dict(value)
        normalized["playbooks"] = normalized_playbooks
        return normalized

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )


class StructuredExtractedPlaybookContent(BaseModel):
    """Legacy strict span-copying contract retained for expert extraction."""

    rationale: str = Field(default=..., min_length=1)
    evidence_kind: Literal[
        "correction",
        "preference",
        "observed-failure",
        "rejected-approach",
        "expert-gap",
        "verified-success",
    ] = Field(default=...)
    trigger: str = Field(default=..., min_length=1)
    content: str = Field(default=..., min_length=1)
    evidence: list[StructuredPlaybookEvidence] = Field(default=..., min_length=1)

    model_config = ConfigDict(extra="ignore")

    @field_validator("evidence_kind", mode="before")
    @classmethod
    def normalize_evidence_kind(cls, value: object) -> object:
        return normalize_evidence_kind_value(value)


class StructuredExtractedPlaybookList(StrictStructuredOutput):
    """Expert-only strict output that still cites copied source spans."""

    playbooks: list[StructuredExtractedPlaybookContent] = Field(
        default=...,
        description="Evidence-complete expert candidates, or an empty list",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate_wrappers_and_keys(cls, value: object) -> object:
        """Retain the copied-span contract's provider compatibility behavior."""
        if not isinstance(value, dict) or not isinstance(value.get("playbooks"), list):
            return value

        aliases = {
            **_EXTRACTED_CANDIDATE_ALIASES,
            "citations": "evidence",
            "evidence": "evidence",
            "evidence_entries": "evidence",
            "sources": "evidence",
        }
        required = {"rationale", "evidence_kind", "trigger", "content", "evidence"}

        def normalize_keys(candidate: dict[object, object]) -> dict[str, object]:
            return normalize_provider_keys(candidate, aliases)

        def find_complete_candidate(
            node: object, depth: int = 0
        ) -> dict[str, object] | None:
            if depth > 4:
                return None
            if isinstance(node, dict):
                candidate = normalize_keys(node)
                if required <= candidate.keys():
                    return candidate
                for nested_value in candidate.values():
                    found = find_complete_candidate(nested_value, depth + 1)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for nested_value in node:
                    found = find_complete_candidate(nested_value, depth + 1)
                    if found is not None:
                        return found
            return None

        normalized_playbooks: list[object] = []
        for raw_candidate in value["playbooks"]:
            if not isinstance(raw_candidate, dict):
                normalized_playbooks.append(raw_candidate)
                continue
            candidate = normalize_keys(raw_candidate)
            if candidate.keys() <= {"turn_ref", "source_span"}:
                logger.info(
                    "event=playbook_candidate_rejected reason=evidence_object_without_candidate"
                )
                continue
            if not required <= candidate.keys():
                candidate = find_complete_candidate(candidate) or candidate
            evidence = candidate.get("evidence")
            if isinstance(evidence, list):
                candidate["evidence"] = [
                    normalize_keys(item) if isinstance(item, dict) else item
                    for item in evidence
                ]
            normalized_playbooks.append(candidate)

        normalized = dict(value)
        normalized["playbooks"] = normalized_playbooks
        return normalized

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"additionalProperties": False},
    )


def uses_evidence_grounded_extraction(
    prompt_manager: PromptManager, *, expert: bool
) -> bool:
    """Return whether the selected prompt version requires strict evidence fields.

    Strict evidence is driven entirely by which prompt version is active, so a
    deployment can roll the contract forward or back by activating a different
    version with no code change. The normal and expert paths are versioned
    independently and cross their floors separately.

    Args:
        prompt_manager (PromptManager): Resolves the active prompt version.
        expert (bool): Whether the expert-content extraction path is in use.

    Returns:
        bool: True when the active prompt version is at or above the strict
        floor for that path.
    """
    prompt_id = (
        "playbook_extraction_context_expert"
        if expert
        else "playbook_extraction_context"
    )
    minimum_version = (3, 6, 0) if expert else (4, 6, 0)
    version = prompt_manager.get_active_version(prompt_id)
    if version is None:
        return False
    try:
        parts = [int(part) for part in version.split(".")]
    except ValueError:
        return False
    # Pad to major.minor.patch so a two-part version compares as its .0 patch
    # rather than sorting below every three-part floor ((4, 6) < (4, 6, 0)).
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3]) >= minimum_version


def _ordered_playbook_requests(
    sessions: list[RequestInteractionDataModel],
) -> list[tuple[int, list[RequestInteractionDataModel]]]:
    """Return chronologically ordered request runs with stable local session labels.

    Request rows can be persisted after the user-visible event they represent
    (for example, a quick-reply selection). Use the earliest interaction time as
    the primary chronology signal and allow requests from linked sessions to
    interleave. Consecutive requests from the same session remain grouped for a
    compact prompt; a later run of that session reuses its local session number.
    """

    def request_key(
        model: RequestInteractionDataModel,
    ) -> tuple[int, int, str]:
        interaction_times = [
            interaction.created_at for interaction in model.interactions
        ]
        visible_time = (
            min(interaction_times) if interaction_times else model.request.created_at
        )
        return visible_time, model.request.created_at, model.request.request_id

    ordered_models = sorted(sessions, key=request_key)
    session_numbers: dict[str, int] = {}
    ordered_runs: list[tuple[int, list[RequestInteractionDataModel]]] = []
    for model in ordered_models:
        session_number = session_numbers.setdefault(
            model.session_id, len(session_numbers) + 1
        )
        if ordered_runs and ordered_runs[-1][0] == session_number:
            ordered_runs[-1][1].append(model)
        else:
            ordered_runs.append((session_number, [model]))
    return ordered_runs


def build_playbook_prompt_context(
    sessions: list[RequestInteractionDataModel],
    *,
    expert: bool = False,
    label_turns: bool = False,
) -> PlaybookPromptContext:
    """Render playbook evidence and retain a call-local provenance map.

    Local ``[T#]`` labels are part of the strict evidence contract, not a
    backwards-compatible formatting detail. Legacy and expert prompts therefore
    remain unlabelled unless their selected prompt version explicitly opts into
    evidence-grounded extraction.
    """
    ordered_sessions = _ordered_playbook_requests(sessions)
    evidence_sources: dict[str, PlaybookEvidenceSource] = {}
    evidence_units: dict[str, PlaybookEvidenceUnit] = {}
    ref_by_object_id: dict[int, str] = {}
    turn_number = 0
    for _, request_models in ordered_sessions:
        for request_model in request_models:
            for interaction in sorted(
                request_model.interactions,
                key=lambda item: (item.created_at, item.interaction_id),
            ):
                turn_number += 1
                turn_ref = f"T{turn_number}"
                ref_by_object_id[id(interaction)] = turn_ref
                source = PlaybookEvidenceSource(
                    interaction_id=interaction.interaction_id,
                    request_id=interaction.request_id,
                    evidence_texts=visible_interaction_evidence_texts(interaction),
                    role=interaction.role,
                    request_source=request_model.request.source or "",
                )
                evidence_sources[turn_ref] = source
                evidence_units[turn_ref] = PlaybookEvidenceUnit(
                    evidence_ref=turn_ref,
                    turn_ref=turn_ref,
                    source_span="\n\n".join(source.evidence_texts),
                    interaction_id=interaction.interaction_id,
                    request_id=interaction.request_id,
                    role=interaction.role,
                    request_source=request_model.request.source or "",
                )

    if expert:
        flat = [
            interaction
            for _, request_models in ordered_sessions
            for request_model in request_models
            for interaction in sorted(
                request_model.interactions,
                key=lambda item: (item.created_at, item.interaction_id),
            )
        ]
        pairs: list[str] = []
        pair_number = 0
        for index, interaction in enumerate(flat):
            if not interaction.expert_content:
                continue
            pair_number += 1
            parts = [f"=== Comparison {pair_number} ==="]
            for preceding in reversed(flat[:index]):
                if preceding.role.lower() == "user":
                    prefix = (
                        f"[{ref_by_object_id[id(preceding)]}] " if label_turns else ""
                    )
                    parts.append(f"{prefix}User Question: ```{preceding.content}```")
                    break
            turn_ref = ref_by_object_id[id(interaction)]
            prefix = f"[{turn_ref}] " if label_turns else ""
            parts.append(f"{prefix}Agent Response: ```{interaction.content}```")
            parts.append(f"{prefix}Expert Response: ```{interaction.expert_content}```")
            pairs.append("\n".join(parts))
        return PlaybookPromptContext(
            text="\n\n".join(pairs),
            evidence_sources=evidence_sources,
            evidence_units=evidence_units,
        )

    rendered_sessions: list[str] = []
    request_number = 0
    for session_number, request_models in ordered_sessions:
        interactions = [
            interaction
            for request_model in request_models
            for interaction in request_model.interactions
        ]
        timestamps = [item.created_at for item in interactions if item.created_at]
        date_suffix = ""
        if timestamps:
            try:
                date_suffix = datetime.fromtimestamp(min(timestamps), tz=UTC).strftime(
                    " (date: %Y-%m-%d)"
                )
            except (OverflowError, OSError, ValueError):
                date_suffix = ""
        lines = [f"=== Session {session_number}{date_suffix} ==="]
        for request_model in request_models:
            request_number += 1
            source = request_model.request.source or "unknown"
            lines.append(f"--- [R{request_number}] Request (source: {source}) ---")
            for interaction in sorted(
                request_model.interactions,
                key=lambda item: (item.created_at, item.interaction_id),
            ):
                turn_ref = ref_by_object_id[id(interaction)]
                rendered = format_interactions_to_history_string([interaction])
                turn_prefix = f"[{turn_ref}] " if label_turns else ""
                lines.extend(f"{turn_prefix}{line}" for line in rendered.splitlines())
        rendered_sessions.append("\n".join(lines))
    return PlaybookPromptContext(
        text="\n\n".join(rendered_sessions),
        evidence_sources=evidence_sources,
        evidence_units=evidence_units,
    )


class PlaybookAggregationOutput(StrictStructuredOutput):
    """
    Output schema for the playbook_aggregation prompt.

    Contains the consolidated playbook entry or null if no new entry should be generated
    (e.g., when it duplicates existing approved playbook).
    """

    playbook: StructuredPlaybookContent | None = Field(
        default=None,
        description="The consolidated playbook entry, or null if no new entry should be generated",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_wrapper_key(cls, data: Any) -> Any:
        """Accept both 'playbook' and legacy 'feedback' as the wrapper key."""
        if isinstance(data, dict) and "feedback" in data and "playbook" not in data:
            data["playbook"] = data.pop("feedback")
        return data

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


def format_structured_fields_for_display(
    structured: StructuredPlaybookContent,
) -> str:
    """
    Format structured metadata fields for display/debug purposes.

    This is NOT for producing content values. Use ensure_playbook_content()
    when you need to obtain the freeform content string.

    Args:
        structured (StructuredPlaybookContent): The structured playbook content

    Returns:
        str: Formatted structured fields string for display
    """
    lines = []

    if structured.trigger:
        lines.append(f'Trigger: "{structured.trigger}"')

    if structured.rationale:
        lines.append(f'Rationale: "{structured.rationale}"')

    if not lines and structured.content:
        return structured.content

    return "\n".join(lines)


def ensure_playbook_content(
    playbook_content: str | None,
    structured: StructuredPlaybookContent,
) -> str:
    """
    Return playbook_content if present; legacy fallback from structured fields.

    Args:
        playbook_content (str | None): The freeform content from the LLM
        structured (StructuredPlaybookContent): Structured fields for fallback

    Returns:
        str: The freeform playbook_content, or a formatted fallback from structured fields.
    """
    if playbook_content and playbook_content.strip():
        return playbook_content
    return format_structured_fields_for_display(structured)


class PlaybookGenerationRequest(BaseModel):
    request_id: str
    agent_version: str
    user_id: str | None = None  # for per-user playbook extraction
    source: str | None = None
    rerun_start_time: int | None = None  # Unix timestamp for rerun flows
    rerun_end_time: int | None = None  # Unix timestamp for rerun flows
    auto_run: bool = (
        True  # True for regular flow (checks stride_size), False for rerun/manual
    )
    force_extraction: bool = False  # when True, bypass all extraction gates (stride_size, cheap pre-filter, LLM should_run)


class PlaybookAggregatorRequest(BaseModel):
    agent_version: str
    rerun: bool = False
    operation_key: str | None = Field(default=None, min_length=1)


def construct_playbook_extraction_messages_from_sessions(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    extraction_definition_prompt: str,
    tool_can_use: str | None = None,
    prompt_context: PlaybookPromptContext | None = None,
) -> list[dict]:
    """
    Construct LLM messages for playbook extraction from sessions.

    This function uses the shared message construction interface to build messages
    with a system prompt and a final user prompt specific to playbook extraction.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: List of request interaction groups to extract playbook entries from
        agent_context_prompt: Context about the agent for system message
        extraction_definition_prompt: Definition of what the playbook should contain
        tool_can_use: Optional formatted string of tools available to the agent

    Returns:
        list[dict]: List of messages ready for playbook extraction
    """
    # Configure system message (before interactions)
    # Stable content (instructions, examples, definitions) goes in system message for token caching
    system_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
            "tool_can_use": tool_can_use or "",
        },
    )

    # Configure final user message (after interactions)
    # Only dynamic per-call data goes in user message
    prompt_context = prompt_context or build_playbook_prompt_context(
        request_interaction_data_models,
        label_turns=uses_evidence_grounded_extraction(prompt_manager, expert=False),
    )
    user_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_PROMPT_ID,
        variables={
            "interactions": prompt_context.text,
        },
    )

    # Extract flat interactions for message construction
    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    # Use shared message construction
    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=system_config,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)


# ===============================
# Expert content utilities
# ===============================


def has_expert_content(interactions: list[Interaction]) -> bool:
    """Check if any interaction has non-empty expert_content."""
    return any(i.expert_content for i in interactions)


def format_expert_comparison_pairs(
    request_interaction_data_models: list[RequestInteractionDataModel],
) -> str:
    """
    Format interactions with expert_content as agent-vs-expert comparison blocks.

    For each agent interaction that has expert_content, includes the preceding user
    question for context, the agent's actual response, and the expert's ideal response.

    Args:
        request_interaction_data_models: Session data models containing interactions

    Returns:
        str: Formatted comparison pairs string
    """
    return build_playbook_prompt_context(
        request_interaction_data_models, expert=True, label_turns=False
    ).text


def construct_expert_playbook_extraction_messages(
    prompt_manager: PromptManager,
    request_interaction_data_models: list[RequestInteractionDataModel],
    agent_context_prompt: str,
    extraction_definition_prompt: str,
    prompt_context: PlaybookPromptContext | None = None,
) -> list[dict]:
    """
    Construct LLM messages for expert-content playbook extraction.

    Uses expert-specific prompts that compare agent responses against expert
    responses and extract playbook entries about alignment gaps.

    Args:
        prompt_manager: The prompt manager for rendering prompt templates
        request_interaction_data_models: Session data with expert_content interactions
        agent_context_prompt: Context about the agent for system message
        extraction_definition_prompt: Definition of what the playbook should contain

    Returns:
        list[dict]: List of messages ready for expert playbook extraction
    """
    system_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_CONTEXT_EXPERT_PROMPT_ID,
        variables={
            "agent_context_prompt": agent_context_prompt,
            "extraction_definition_prompt": extraction_definition_prompt,
        },
    )

    prompt_context = prompt_context or build_playbook_prompt_context(
        request_interaction_data_models,
        expert=True,
        label_turns=uses_evidence_grounded_extraction(prompt_manager, expert=True),
    )

    user_config = PromptConfig(
        prompt_id=PlaybookServiceConstants.PLAYBOOK_EXTRACTION_EXPERT_PROMPT_ID,
        variables={
            "comparison_pairs": prompt_context.text,
        },
    )

    interactions = extract_interactions_from_request_interaction_data_models(
        request_interaction_data_models
    )

    config = MessageConstructionConfig(
        prompt_manager=prompt_manager,
        system_prompt_config=system_config,
        user_prompt_config=user_config,
    )

    return construct_messages_from_interactions(interactions, config)
