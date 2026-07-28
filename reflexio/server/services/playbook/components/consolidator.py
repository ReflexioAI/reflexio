"""
Playbook consolidation service that merges duplicate user playbook entries using LLM
and hybrid search against existing entries in the database.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import (
    DeduplicationConfig,
    SearchOptions,
)
from reflexio.models.structured_output import (
    StrictStructuredOutput,
    normalize_provider_value,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMClientError,
)
from reflexio.server.services.deduplication_utils import (
    BaseDeduplicator,
    format_dedup_timestamp,
    resolve_dedup_query_embeddings,
)
from reflexio.server.services.playbook.playbook_service_utils import (
    is_evidence_validated,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    check_string_token_overlap,
)
from reflexio.server.tracing import sentry_tags

logger = logging.getLogger(__name__)

# Token-overlap thresholds for judging whether two playbook rows state the same
# lesson. Rows citing the same interaction need less lexical agreement than rows
# matched purely on wording across different extraction windows.
#
# Calibration: these are `check_string_token_overlap` ratios (shared tokens over
# the smaller token set), NOT semantic similarity. They were chosen for the
# asymmetry between the two match modes, not fitted to a labelled corpus — treat
# them as tunable policy, and re-check the consolidation regression tests in
# `tests/server/services/playbook/test_playbook_consolidator.py` after moving any
# of them. Raising a threshold trades duplicate survivors for missed merges;
# lowering it trades missed merges for wrongly collapsed distinct lessons.
#
# Shared-provenance rows already agree on *which moment* they describe, so the
# wording bar is deliberately the loosest of the set.
_SAME_EVIDENCE_CONTENT_THRESHOLD = 0.5
# Rows from different windows have only wording in common, so they must agree
# substantially before being treated as one lesson.
_CROSS_EVIDENCE_CONTENT_THRESHOLD = 0.75
# Triggers are short retrieval keys; this sits between the two content bars so a
# paraphrased trigger still anchors a match without matching every trigger.
_SAME_TRIGGER_THRESHOLD = 0.6
# Post-apply survivor collision uses the stricter default ratio, and ignores
# short rules entirely — brief rules share too much vocabulary to compare by
# ratio, so only exactly equal content counts for them. 80 chars is roughly one
# full sentence of guidance, below which the ratio is dominated by stopwords.
_SURVIVOR_CONTENT_THRESHOLD = 0.7
_SURVIVOR_MIN_CONTENT_LENGTH = 80
_REVIEWER_REVISION_NOTE_PREFIX = "First-pass reviewer: revised ("

# Decision-kind variants providers emit instead of the declared literals. Keys
# are in canonical ``normalize_provider_token`` form (casefolded, "_"-joined).
_DECISION_KIND_ALIASES = {
    "accept": "independent",
    "keep": "independent",
    "merge": "unify",
    "reject": "reject_new",
}


# ===============================
# Playbook-specific Pydantic Output Schemas for LLM
# ===============================

NewIdField = str | list[str]
ExistingIdField = int | list[int]


def _strip_prompt_brackets(value: str) -> str:
    """Strip the display brackets used in prompt labels, e.g. ``[NEW-0]``."""
    stripped = value.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped[1:-1].strip()
    return stripped


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _coerce_new_id(value: object) -> str:
    """Normalize a prompt-format NEW id to ``NEW-N``.

    LLMs sometimes echo the rendered display label (``[NEW-0]``) or wrap the
    id in a one-item list. Keep the apply path keyed on canonical ``NEW-N``.
    """
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"new_id expected one NEW id, got {value!r}")
        return _coerce_new_id(value[0])
    if not isinstance(value, str):
        raise ValueError(
            f"new_id must be 'NEW-N' label, got {type(value).__name__}: {value!r}"
        )

    stripped = _strip_prompt_brackets(value)
    for prefix in ("NEW-", "NEW_", "new-", "new_"):
        if stripped.startswith(prefix):
            suffix = stripped[len(prefix) :]
            break
    else:
        raise ValueError(f"new_id must be 'NEW-N' label, got {value!r}")

    try:
        parsed = int(suffix)
    except ValueError as exc:
        raise ValueError(f"new_id must be 'NEW-N' label, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"new_id must be >= 0, got {value!r}")
    return f"NEW-{parsed}"


def _coerce_new_ids(value: object, *, allow_many: bool) -> NewIdField:
    values = value if isinstance(value, list) else [value]
    coerced = _dedupe_preserving_order([_coerce_new_id(item) for item in values])
    if not coerced:
        raise ValueError("new_id must include at least one NEW id")
    if not allow_many and len(coerced) != 1:
        raise ValueError(f"new_id expected one NEW id, got {coerced!r}")
    return coerced if len(coerced) > 1 else coerced[0]


def _new_ids_from_field(value: NewIdField) -> list[str]:
    return list(value) if isinstance(value, list) else [value]


def _coerce_existing_position(value: object) -> int:
    """Accept either a bare int position or an ``"EXISTING-N"`` label.

    Wired to all three EXISTING-id integer fields —
    ``UnifyDecision.archive_existing_ids``,
    ``RejectNewDecision.superseded_by_existing_id`` and
    ``DifferentiateDecision.existing_id``. The consolidation prompt instructs
    the model to emit a list **position** for every one of them, and the apply
    path resolves each position-first via ``existing_by_position``
    (``f"EXISTING-{idx}"``), falling back to ``existing_by_id`` only for older
    prompt outputs. Coercing the label here keeps all three fields consistent
    so a weak model that returns ``"EXISTING-0"`` for any of them does not kill
    the whole batch.

    The consolidation prompt labels rows as ``[EXISTING-0]``, ``[EXISTING-1]``
    etc. (see ``_format_playbooks_with_prefix``) and the apply path
    reconstructs ``f"EXISTING-{position}"`` from the integer the LLM returns.
    Strong structured-output models (GPT-4o, Claude) honor the ``list[int]``
    schema and return the bare integer ``0``; weaker models (e.g. MiniMax-M3)
    ignore the int constraint and return the literal label ``"EXISTING-0"``
    instead — which then fails pydantic validation and the whole
    consolidation batch dies.

    Strip the prefix when present so the schema tolerates both shapes
    without changing the int contract downstream consumers rely on. Plain
    numeric strings (``"5"``) are also accepted for symmetry with how
    most JSON-coerced models handle ID-like values. Negative values are
    rejected — list positions are always ``>= 0``.

    Raises:
        ValueError: when ``value`` is not a non-negative int or a recognized
            position-label / numeric string.
    """
    if isinstance(value, bool):
        # ``bool`` is a subclass of ``int`` in Python; reject explicitly so a
        # stray ``True`` doesn't silently become position 1.
        raise ValueError(f"existing-position must be int, got bool: {value!r}")
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"existing-position expected one id, got {value!r}")
        return _coerce_existing_position(value[0])
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"existing-position must be >= 0, got {value!r}")
        return value
    if isinstance(value, str):
        stripped = _strip_prompt_brackets(value)
        for prefix in ("EXISTING-", "EXISTING_", "existing-", "existing_"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                break
        try:
            parsed = int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"existing-position must be int or 'EXISTING-N' label, got {value!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(f"existing-position must be >= 0, got {value!r}")
        return parsed
    raise ValueError(
        f"existing-position must be int or 'EXISTING-N' label, got {type(value).__name__}: {value!r}"
    )


def _coerce_existing_positions(value: object, *, allow_many: bool) -> ExistingIdField:
    values = value if isinstance(value, list) else [value]
    coerced = [_coerce_existing_position(item) for item in values]
    if not coerced:
        raise ValueError("existing-position must include at least one id")
    if not allow_many and len(coerced) != 1:
        raise ValueError(f"existing-position expected one id, got {coerced!r}")
    return coerced if len(coerced) > 1 else coerced[0]


def _existing_ids_from_field(value: ExistingIdField) -> list[int]:
    return list(value) if isinstance(value, list) else [value]


def _new_id_log_label(value: NewIdField) -> str:
    return ",".join(_new_ids_from_field(value)) if isinstance(value, list) else value


def _existing_id_log_label(value: ExistingIdField) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def _normalize_generation_request_id(
    generation_request_id: str | None,
    *,
    request_id: str | None = None,
) -> str:
    if generation_request_id is not None:
        if request_id is not None and request_id != generation_request_id:
            raise TypeError(
                "generation_request_id and request_id must match when both are provided"
            )
        return generation_request_id
    if request_id is not None:
        return request_id
    raise TypeError("generation_request_id is required")


class UnifyDecision(BaseModel):
    """Collapse NEW (+ 0..N EXISTING) into one row with LLM-supplied content.

    Subsumes the legacy ``duplicate`` and ``prefer_new`` kinds AND the
    ``compose`` case: the LLM picks the final ``content`` / ``trigger`` /
    ``rationale`` and lists which EXISTING ids (if any) are absorbed. An empty
    ``archive_existing_ids`` is allowed and behaves as an insert-without-archive
    distinguished from ``independent`` by the prompt's intent contract.

    A unified skill MAY hold mixed-polarity rules (do-rules and avoid-rules for
    different sub-aspects of the one task). There is no mechanical polarity
    field or apply-time polarity check: the no-self-contradiction judgment
    (do not merge rules that contradict on the same situation) is made by the
    LLM in the consolidation prompt, not by the apply path.
    """

    kind: Literal["unify"] = "unify"
    new_id: NewIdField
    archive_existing_ids: list[int] = Field(default_factory=list)
    content: str
    trigger: str
    rationale: str
    reason: str = ""

    @field_validator("new_id", mode="before")
    @classmethod
    def _coerce_new_id(cls, value: object) -> NewIdField:
        return _coerce_new_ids(value, allow_many=True)

    @field_validator("archive_existing_ids", mode="before")
    @classmethod
    def _coerce_archive_ids(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [_coerce_existing_position(item) for item in value]
        return [_coerce_existing_position(value)]

    @property
    def new_ids(self) -> list[str]:
        return _new_ids_from_field(self.new_id)

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class RejectNewDecision(BaseModel):
    """The new candidate is redundant; an existing row supersedes it (storage no-op).

    ``superseded_by_existing_id`` is a bare integer that normally refers to the
    rendered ``EXISTING-N`` list position. The apply path also accepts a DB
    ``user_playbook_id`` fallback for older prompts/tests, but list position is
    resolved first because it is the only identifier visible in the rendered
    consolidation prompt.
    """

    kind: Literal["reject_new"] = "reject_new"
    new_id: NewIdField
    superseded_by_existing_id: ExistingIdField
    reason: str = ""

    @field_validator("new_id", mode="before")
    @classmethod
    def _coerce_new_id(cls, value: object) -> NewIdField:
        return _coerce_new_ids(value, allow_many=True)

    @field_validator("superseded_by_existing_id", mode="before")
    @classmethod
    def _coerce_superseded_id(cls, value: object) -> ExistingIdField:
        return _coerce_existing_positions(value, allow_many=True)

    @property
    def new_ids(self) -> list[str]:
        return _new_ids_from_field(self.new_id)

    @property
    def superseded_by_existing_ids(self) -> list[int]:
        return _existing_ids_from_field(self.superseded_by_existing_id)

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class DifferentiateDecision(BaseModel):
    """Both rules valid in distinct contexts: refine both triggers.

    ``existing_id`` is a bare integer that normally refers to the rendered
    ``EXISTING-N`` list position. The apply path also accepts a DB
    ``user_playbook_id`` fallback for older prompts/tests, but list position is
    resolved first because it is the only identifier visible in the rendered
    consolidation prompt.
    """

    kind: Literal["differentiate"] = "differentiate"
    new_id: str
    existing_id: int
    refined_new_trigger: str
    refined_existing_trigger: str
    reason: str = ""

    @field_validator("new_id", mode="before")
    @classmethod
    def _coerce_new_id(cls, value: object) -> str:
        return _coerce_new_id(value)

    @field_validator("existing_id", mode="before")
    @classmethod
    def _coerce_existing_id(cls, value: object) -> int:
        coerced = _coerce_existing_positions(value, allow_many=False)
        if not isinstance(coerced, int):
            raise ValueError(f"existing_id expected one id, got {coerced!r}")
        return coerced

    @field_validator("refined_new_trigger", "refined_existing_trigger")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("differentiate requires non-empty refined triggers")
        return v

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class IndependentDecision(BaseModel):
    """Unrelated to any existing row: insert new as-is, no archive."""

    kind: Literal["independent"] = "independent"
    new_id: NewIdField
    reason: str = ""

    @field_validator("new_id", mode="before")
    @classmethod
    def _coerce_new_id(cls, value: object) -> NewIdField:
        return _coerce_new_ids(value, allow_many=True)

    @property
    def new_ids(self) -> list[str]:
        return _new_ids_from_field(self.new_id)

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


ConsolidationDecision = Annotated[
    UnifyDecision | RejectNewDecision | DifferentiateDecision | IndependentDecision,
    Field(discriminator="kind"),
]


# ``PlaybookConsolidationOutput`` inherits ``StrictStructuredOutput`` so the
# emitted JSON schema is folded into the provider-accepted union shape while the
# discriminator is kept for keyed validation at parse time (Sentry
# PYTHON-FASTAPI-9J). This note is a comment, NOT part of the docstring, because
# the docstring is serialized into the wire schema's ``description`` sent to the
# model — keep implementation tokens out of it.
class PlaybookConsolidationOutput(StrictStructuredOutput):
    """Output schema for playbook consolidation as a 4-kind tagged union.

    Each decision is one of ``UnifyDecision``, ``RejectNewDecision``,
    ``DifferentiateDecision``, or ``IndependentDecision``; the ``kind`` field
    selects the concrete shape.
    """

    decisions: list[ConsolidationDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _infer_missing_decision_kinds(cls, value: object) -> object:
        """Recover the decision tag from unambiguous fields or common aliases."""
        if not isinstance(value, dict) or not isinstance(value.get("decisions"), list):
            return value
        normalized = dict(value)
        decisions: list[object] = []
        for raw_decision in value["decisions"]:
            if not isinstance(raw_decision, dict) or raw_decision.get("kind"):
                decisions.append(raw_decision)
                continue
            decision = dict(raw_decision)
            aliased_kind = next(
                (
                    decision.get(key)
                    for key in ("decision", "action", "type")
                    if isinstance(decision.get(key), str)
                ),
                None,
            )
            if isinstance(aliased_kind, str):
                candidate_kind = normalize_provider_value(
                    aliased_kind, _DECISION_KIND_ALIASES
                )
                if candidate_kind in {
                    "unify",
                    "reject_new",
                    "differentiate",
                    "independent",
                }:
                    decision["kind"] = candidate_kind
            if "kind" not in decision:
                if {
                    "refined_new_trigger",
                    "refined_existing_trigger",
                } <= decision.keys():
                    decision["kind"] = "differentiate"
                elif "superseded_by_existing_id" in decision:
                    decision["kind"] = "reject_new"
                elif {"content", "trigger", "rationale"} <= decision.keys():
                    decision["kind"] = "unify"
                elif "new_id" in decision:
                    decision["kind"] = "independent"
            decisions.append(decision)
        normalized["decisions"] = decisions
        return normalized

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class PlaybookConsolidationResult(BaseModel):
    """Per-kind counters tracked over one consolidation batch.

    Bumped once per successfully-applied decision; ``failed_count`` is bumped
    when a single decision's apply path raises, allowing the rest of the batch
    to proceed unaffected.
    """

    unify_count: int = 0
    reject_new_count: int = 0
    differentiate_count: int = 0
    independent_count: int = 0
    failed_count: int = 0


_COUNTER_BY_KIND: dict[str, str] = {
    "unify": "unify_count",
    "reject_new": "reject_new_count",
    "differentiate": "differentiate_count",
    "independent": "independent_count",
}


def _decision_new_ids(decision: ConsolidationDecision) -> list[str]:
    if isinstance(decision, DifferentiateDecision):
        return [decision.new_id]
    return decision.new_ids


def validate_consolidation_output(
    new_playbooks: list[UserPlaybook],
    output: PlaybookConsolidationOutput,
) -> list[str]:
    """Return partition-contract errors for a parsed consolidation output."""
    known_new_ids = {f"NEW-{idx}" for idx in range(len(new_playbooks))}
    seen_by_id: dict[str, list[str]] = {}
    errors: list[str] = []

    for decision_index, decision in enumerate(output.decisions):
        label = f"decision[{decision_index}] {decision.kind}"
        for new_id in _decision_new_ids(decision):
            if new_id not in known_new_ids:
                errors.append(f"{label} references unknown NEW id {new_id}.")
                continue
            seen_by_id.setdefault(new_id, []).append(label)

    missing = sorted(known_new_ids - set(seen_by_id))
    if missing:
        errors.append(
            "Every NEW id must appear exactly once; missing NEW ids: "
            + ", ".join(missing)
            + "."
        )

    duplicates = {
        new_id: labels for new_id, labels in seen_by_id.items() if len(labels) > 1
    }
    for new_id, labels in sorted(duplicates.items()):
        errors.append(
            f"Every NEW id must appear exactly once; {new_id} appears in "
            + "; ".join(labels)
            + "."
        )

    return errors


class PlaybookConsolidator(BaseDeduplicator):
    """
    Consolidates new user playbook entries against each other and against existing entries
    in the database using hybrid search (vector + FTS) and LLM-based merging.
    """

    DEDUPLICATION_PROMPT_ID = "playbook_consolidation"

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        dedup_config: DeduplicationConfig | None = None,
    ):
        """
        Initialize the playbook consolidator.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client for LLM calls
            dedup_config: Optional consolidation search parameters (threshold, top_k)
        """
        super().__init__(request_context, llm_client)
        self._dedup_config = dedup_config or DeduplicationConfig()
        self.model_provenance: ModelProvenance | None = None
        self.consolidated_output_indices: set[int] = set()

    def _get_prompt_id(self) -> str:
        """Get the prompt ID for playbook consolidation."""
        return self.DEDUPLICATION_PROMPT_ID

    def _get_item_count_key(self) -> str:
        """Get the key name for item count in prompt variables."""
        return "new_playbook_count"

    def _get_items_key(self) -> str:
        """Get the key name for items in prompt variables."""
        return "new_playbooks"

    def _get_output_schema_class(self) -> type[BaseModel]:
        """Return the discriminated-union output schema for consolidation."""
        return PlaybookConsolidationOutput

    def _format_items_for_prompt(self, playbooks: list[UserPlaybook]) -> str:
        """
        Format user playbook entries list for LLM prompt with NEW-N prefix.

        Args:
            playbooks: List of user playbook entries

        Returns:
            Formatted string representation
        """
        return self._format_playbooks_with_prefix(playbooks, "NEW")

    def _format_playbooks_with_prefix(
        self,
        playbooks: list[UserPlaybook],
        prefix: str,
        evidence_groups: list[str] | None = None,
    ) -> str:
        """
        Format user playbook entries with a given prefix (NEW or EXISTING).

        Args:
            playbooks: List of user playbook entries to format
            prefix: Prefix string for indices

        Returns:
            Formatted string
        """
        if not playbooks:
            return "(None)"
        lines = []
        for idx, playbook in enumerate(playbooks):
            playbook_name = playbook.playbook_name or "unknown"
            source = playbook.source or "unknown"
            created_date = format_dedup_timestamp(playbook.created_at)
            # ``Trigger`` and ``Rationale`` are included alongside ``Content``
            # so the model can actually compare the fields it is asked to
            # refine (``differentiate``, same-trigger contradictions, trigger
            # refinements). Without ``trigger`` exposed the decisions become
            # guesswork.
            lines.append(
                f'[{prefix}-{idx}] Content: "{playbook.content}"'
                f' | Trigger: "{playbook.trigger or ""}"'
                f' | Rationale: "{playbook.rationale or ""}"'
                f' | Evidence Span: "{playbook.source_span or ""}"'
                f' | Evidence Group: "{evidence_groups[idx] if evidence_groups else "none"}"'
                f" | Name: {playbook_name}"
                f" | Source: {source} | Last Modified: {created_date}"
            )
        return "\n".join(lines)

    def retrieve_existing_playbooks(
        self,
        new_playbooks: list[UserPlaybook],
        user_id: str | None = None,
        agent_version: str | None = None,
    ) -> list[UserPlaybook]:
        """
        Retrieve existing user playbook entries from the database using hybrid search.

        For each new entry, uses its trigger field as the query with
        pre-computed embeddings for vector search.

        A failed per-query search is logged and skipped rather than raised. This
        is a retrieval-recall degradation, not a correctness failure: consolidation
        still applies its full decision contract to whatever was retrieved. Only
        decision generation and application fail the batch closed.

        Args:
            new_playbooks: List of new entries to search against
            user_id: Optional user ID to scope the search
            agent_version: Optional agent version to scope the search

        Returns:
            Deduplicated list of existing UserPlaybook objects from the database
        """
        storage = self.request_context.storage

        # Collect trigger strings for embedding
        query_texts = []
        for playbook in new_playbooks:
            trigger = playbook.trigger or playbook.content
            if trigger and trigger.strip():
                query_texts.append(trigger.strip())

        if not query_texts:
            return []

        # Embed dedup queries with the same model that indexed the store —
        # see resolve_dedup_query_embeddings for why the client's default
        # embedding model must not be used here.
        embeddings = resolve_dedup_query_embeddings(
            storage, self.client, query_texts, entity_label="Playbook"
        )

        # Search for each new entry
        seen_ids: set[int] = set()
        existing_playbooks: list[UserPlaybook] = []

        for i, query_text in enumerate(query_texts):
            try:
                search_request = SearchUserPlaybookRequest(
                    query=query_text,
                    user_id=user_id,
                    agent_version=agent_version,
                    status_filter=[None],  # Only current entries
                    threshold=self._dedup_config.search_threshold,
                    top_k=self._dedup_config.search_top_k,
                )
                search_options = SearchOptions(query_embedding=embeddings[i])
                results = storage.search_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
                    search_request, search_options
                )
                for fb in results:
                    if fb.user_playbook_id and fb.user_playbook_id not in seen_ids:
                        seen_ids.add(fb.user_playbook_id)
                        existing_playbooks.append(fb)
            except Exception:  # noqa: PERF203 — per-query isolation is the point
                logger.exception(
                    "event=playbook_consolidation_search_failed query_index=%d", i
                )

        logger.info(
            "Retrieved %d unique existing user playbook entries for deduplication "
            "(scoped to user_id=%r agent_version=%r)",
            len(existing_playbooks),
            user_id,
            agent_version,
        )
        return existing_playbooks

    def _format_new_and_existing_for_prompt(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
    ) -> tuple[str, str]:
        """
        Format new and existing entries for the deduplication prompt.

        Args:
            new_playbooks: New entries to deduplicate
            existing_playbooks: Existing entries from the database

        Returns:
            Tuple of (new_playbooks_text, existing_playbooks_text)
        """
        group_labels = self._evidence_group_labels(
            [*new_playbooks, *existing_playbooks]
        )
        new_text = self._format_playbooks_with_prefix(
            new_playbooks, "NEW", group_labels[: len(new_playbooks)]
        )
        existing_text = self._format_playbooks_with_prefix(
            existing_playbooks,
            "EXISTING",
            group_labels[len(new_playbooks) :],
        )
        return new_text, existing_text

    @staticmethod
    def _evidence_group_labels(playbooks: list[UserPlaybook]) -> list[str]:
        """Label connected source-overlap components without exposing source IDs."""
        source_sets = [set(playbook.source_interaction_ids) for playbook in playbooks]
        labels = ["none"] * len(playbooks)
        next_group = 0
        for start, source_ids in enumerate(source_sets):
            if not source_ids or labels[start] != "none":
                continue
            label = f"EVIDENCE-{next_group}"
            next_group += 1
            pending = [start]
            labels[start] = label
            while pending:
                current = pending.pop()
                for other, other_source_ids in enumerate(source_sets):
                    if labels[other] != "none" or not other_source_ids:
                        continue
                    if source_sets[current].intersection(other_source_ids):
                        labels[other] = label
                        pending.append(other)
        return labels

    def _render_consolidation_prompt(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
    ) -> str:
        """Render the consolidation prompt for the given NEW + EXISTING rows.

        Rendering is deterministic, so the repair path calls this again to
        reconstruct the exact first-turn prompt when building the follow-up
        conversation.

        Args:
            new_playbooks: New entries to deduplicate.
            existing_playbooks: Existing entries from the database.

        Returns:
            The fully rendered consolidation prompt.
        """
        new_text, existing_text = self._format_new_and_existing_for_prompt(
            new_playbooks, existing_playbooks
        )
        return self.request_context.prompt_manager.render_prompt(
            self._get_prompt_id(),
            {
                "new_playbook_count": len(new_playbooks),
                "new_playbooks": new_text,
                "existing_playbooks": existing_text,
            },
        )

    def _consolidation_decisions(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
    ) -> PlaybookConsolidationOutput:
        """Render the consolidation prompt for NEW + EXISTING playbooks and run the
        LLM decision step (prompt render + LLM call + parse only — no hybrid search,
        no apply). Returns the parsed decisions, or an empty ``PlaybookConsolidationOutput``
        if the LLM returned the wrong shape.

        EXISTING-id <-> prompt-label mapping (for downstream eval providers that
        must map a returned decision back to a source playbook):
          * Both ``new_playbooks`` and ``existing_playbooks`` are rendered by
            ``_format_playbooks_with_prefix``, which labels rows by **list
            position**, not by ``user_playbook_id``: NEW rows become
            ``[NEW-0]``, ``[NEW-1]``, ... and EXISTING rows become
            ``[EXISTING-0]``, ``[EXISTING-1]``, ... in the order passed in.
          * Consequently the integer ids returned in decisions are interpreted
            against EITHER positions OR ``user_playbook_id`` depending on the
            decision kind, in the apply path (``_build_deduplicated_results``):
              - ``UnifyDecision.archive_existing_ids`` -> **list positions**
                (resolved as ``EXISTING-{idx}``).
              - ``DifferentiateDecision.existing_id`` and
                ``RejectNewDecision.superseded_by_existing_id`` ->
                ``user_playbook_id`` (resolved against ``existing_by_id``).
              - All decisions' ``new_id`` is the ``NEW-{idx}`` position label of
                the candidate.
          * A provider that controls the inputs should therefore choose its
            ``existing_playbooks`` ordering and ``user_playbook_id`` values so it
            can map a returned ``existing_id`` (position for unify;
            ``user_playbook_id`` for differentiate/reject_new) back to its case.

        Args:
            new_playbooks: Flattened list of new (candidate) entries.
            existing_playbooks: Existing entries to consolidate against.

        Returns:
            Parsed ``PlaybookConsolidationOutput``; an empty output (no
            decisions) if the LLM returned an unexpected response shape.
        """
        prompt = self._render_consolidation_prompt(new_playbooks, existing_playbooks)

        output_schema_class = self._get_output_schema_class()

        from reflexio.server.services.service_utils import (
            log_llm_messages,
            log_model_response,
        )

        log_llm_messages(
            logger,
            "Playbook consolidation",
            [{"role": "user", "content": prompt}],
        )

        def _validate_output(output: BaseModel) -> list[str]:
            if not isinstance(output, PlaybookConsolidationOutput):
                return [f"Unexpected output type: {type(output).__name__}."]
            errors = validate_consolidation_output(new_playbooks, output)
            if not errors:
                errors = self._find_suspicious_under_consumed_new_rows(
                    new_playbooks, output
                )
            if not errors:
                try:
                    preview_rows, preview_archive_ids, _ = (
                        self._build_deduplicated_results(
                            new_playbooks=new_playbooks,
                            existing_playbooks=existing_playbooks,
                            dedup_output=output,
                            generation_request_id="validation-preview",
                            validate_only=True,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors = [f"Decisions cannot be applied: {exc}"]
                else:
                    untouched_existing = [
                        existing
                        for existing in existing_playbooks
                        if existing.user_playbook_id not in preview_archive_ids
                    ]
                    errors = self._find_overlapping_survivors(
                        preview_rows,
                        untouched_existing,
                    )
            if errors:
                return [
                    *errors,
                    "Return corrected decisions that cover every NEW id exactly once. "
                    "If a unified rule uses facts/details from multiple NEW rows, "
                    "`new_id` must list every consumed NEW id. Keep decisions that are "
                    "otherwise semantically correct. If survivors overlap, unify the "
                    "duplicate NEW ids, reject a redundant NEW row against the matching "
                    "EXISTING row, or differentiate genuinely distinct triggers.",
                ]
            return []

        try:
            completion = self.client.generate_chat_response_with_provenance(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=output_schema_class,
                structured_output_validator=_validate_output,
            )
        except (StructuredOutputRepairError, LiteLLMClientError):
            raise

        self.model_provenance = completion.provenance
        response = completion.value

        log_model_response(logger, "Consolidation response", response)

        if not isinstance(response, PlaybookConsolidationOutput):
            logger.warning(
                "Unexpected response type from consolidation LLM: %s",
                type(response),
            )
            raise TypeError(
                f"Unexpected response type from consolidation LLM: {type(response).__name__}"
            )

        return response

    def deduplicate(
        self,
        results: list[list[UserPlaybook]],
        generation_request_id: str | None = None,
        agent_version: str | None = None,
        user_id: str | None = None,
        *,
        request_id: str | None = None,
        existing_playbooks: list[UserPlaybook] | None = None,
    ) -> tuple[list[UserPlaybook], list[int], list[tuple[int, list[int]]]]:
        """
        Consolidate user playbook entries across extractors and against existing entries in DB.

        Args:
            results: List of entry lists from extractors (each extractor returns list[UserPlaybook])
            generation_request_id: Request ID for context
            agent_version: Agent version for context
            user_id: Optional user ID to scope the existing entry search
            existing_playbooks: Already-retrieved rows for these candidates. Pass
                this when an earlier stage in the same generation already ran the
                hybrid search, so consolidation does not repeat the embedding
                queries. Omit to have consolidation retrieve them itself.

        Returns:
            Tuple of ``(consolidated entries, existing ids to delete after save,
            merge_groups)``. ``merge_groups`` is a list of
            ``(survivor_index, source_existing_ids)`` where ``survivor_index``
            indexes into the returned entries list and identifies the row that
            supersedes the given existing source ids (one entry per ``unify``
            decision that archives at least one existing row). Callers persist
            the entries first (assigning survivor ids), then route each merge
            group through ``storage.merge_records`` so each source becomes a
            MERGED tombstone pointing at its survivor. The "existing ids to
            delete" set still includes ALL archived ids (merge sources +
            non-merge archives such as ``differentiate``'s split source); the
            caller subtracts the merge-covered ids to find pure-delete leftovers.
        """
        generation_request_id = _normalize_generation_request_id(
            generation_request_id, request_id=request_id
        )
        self.model_provenance = None
        self.consolidated_output_indices = set()
        if agent_version is None:
            raise TypeError("agent_version is required")

        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: skipping consolidation")
            all_playbooks: list[UserPlaybook] = []
            for result in results:
                if isinstance(result, list):
                    all_playbooks.extend(result)
            return all_playbooks, [], []

        # Flatten all new entries
        new_playbooks: list[UserPlaybook] = []
        for result in results:
            if isinstance(result, list):
                new_playbooks.extend(result)

        if not new_playbooks:
            return [], [], []

        # Retrieve existing entries via hybrid search, unless an earlier stage in
        # this generation already did (the reviewer needs the same rows).
        if existing_playbooks is None:
            existing_playbooks = self.retrieve_existing_playbooks(
                new_playbooks, user_id=user_id, agent_version=agent_version
            )

        # Evidence-validated first-pass candidates need no probabilistic
        # consolidation when there is no existing retrieval match. Exact
        # same-batch duplicates were already removed before this component; the
        # later reviewer/consolidation phase owns semantic candidate merging.
        # Keeping independent grounded siblings here prevents an unrelated
        # consolidation parse failure from erasing all first-pass lessons.
        evidence_validated = all(map(is_evidence_validated, new_playbooks))
        if evidence_validated and existing_playbooks:
            deterministic_merge = self._merge_replayed_evidence_duplicates(
                new_playbooks,
                existing_playbooks,
                generation_request_id=generation_request_id,
            )
            if deterministic_merge is not None:
                logger.info(
                    "event=playbook_consolidation_evidence_duplicate_merge "
                    "new_count=%d survivor_count=%d request_id=%s",
                    len(new_playbooks),
                    len(deterministic_merge[0]),
                    generation_request_id,
                )
                return deterministic_merge
        if not existing_playbooks and evidence_validated:
            logger.info(
                "event=playbook_consolidation_evidence_independent count=%d request_id=%s",
                len(new_playbooks),
                generation_request_id,
            )
            return (
                [
                    playbook.model_copy(update={"request_id": generation_request_id})
                    for playbook in new_playbooks
                ],
                [],
                [],
            )

        # A single validated candidate with no retrieval match has nothing to
        # consolidate. Avoid adding a probabilistic structured-output failure
        # point to the common independent path; the extractor has already
        # enforced evidence and provenance, and there is no sibling or existing
        # row that could overlap it. Still stamp the generation request exactly
        # as the independent decision apply path would.
        if len(new_playbooks) == 1 and not existing_playbooks:
            logger.info(
                "event=playbook_consolidation_single_independent request_id=%s",
                generation_request_id,
            )
            return (
                [
                    new_playbooks[0].model_copy(
                        update={"request_id": generation_request_id}
                    )
                ],
                [],
                [],
            )

        # Run the LLM decision step (prompt render + LLM call + parse only).
        try:
            dedup_output = self._consolidation_decisions(
                new_playbooks, existing_playbooks
            )
        except Exception as e:
            with sentry_tags(
                subsystem="playbook_consolidator",
                op="identify_duplicates",
                org_id=self.request_context.org_id,
                user_id=user_id,
                request_id=generation_request_id,
                agent_version=agent_version,
                error_type=type(e).__name__,
            ):
                logger.exception("Failed to identify duplicates")
            raise

        if not dedup_output.decisions:
            raise ValueError(
                "Consolidation returned no decisions for a non-empty candidate batch "
                f"(request_id={generation_request_id})"
            )

        logger.info(
            "Received %d consolidation decisions for request %s",
            len(dedup_output.decisions),
            generation_request_id,
        )
        # Build consolidated result via the discriminated-union apply path.
        consolidated = self._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            generation_request_id=generation_request_id,
            agent_version=agent_version,
        )
        new_rows, archive_ids, _ = consolidated
        untouched_existing = [
            existing
            for existing in existing_playbooks
            if existing.user_playbook_id not in archive_ids
        ]
        overlap_errors = self._find_overlapping_survivors(
            new_rows,
            untouched_existing,
        )
        if overlap_errors:
            logger.error(
                "event=playbook_consolidation_overlap_after_repair count=%d "
                "request_id=%s",
                len(overlap_errors),
                generation_request_id,
            )
            raise ValueError("; ".join(overlap_errors))
        return consolidated

    def _merge_replayed_evidence_duplicates(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
        *,
        generation_request_id: str,
    ) -> tuple[list[UserPlaybook], list[int], list[tuple[int, list[int]]]] | None:
        """Deterministically merge exact lessons repeated by overlapping windows.

        This path is intentionally narrow. A repeated NEW row must match exactly
        one persisted row using the same evidence kind (stored in ``reader_angle``)
        plus either shared validated interaction provenance and substantial
        content overlap, or high content overlap and the same future trigger. The
        latter catches the same lesson cited from adjacent replay windows. A NEW
        row with neither related provenance nor a semantic match is independently
        insertable. Ambiguous matches fall through to evidence-aware LLM
        consolidation. For matches, an explicit reviewer revision wins; otherwise
        persisted wording wins. Provenance and notes are unioned in stable order.
        """
        matches: dict[int, tuple[UserPlaybook, list[UserPlaybook]]] = {}
        # Keyed by list position, never by ``id()``: two equal candidate objects
        # would otherwise collide and silently drop one of them.
        independent_positions: set[int] = set()
        matched_existing_id_by_position: dict[int, int] = {}
        for position, candidate in enumerate(new_playbooks):
            candidate_angle = (candidate.reader_angle or "").strip().casefold()
            provenance_related = [
                existing
                for existing in existing_playbooks
                if existing.user_playbook_id
                and self._shares_evidence(candidate, existing)
            ]
            matching_existing = []
            for existing in existing_playbooks:
                if (
                    not existing.user_playbook_id
                    or candidate_angle
                    != (existing.reader_angle or "").strip().casefold()
                ):
                    continue
                # A row citing the same interaction is already known to describe
                # the same moment, so it needs less lexical agreement than one
                # matched purely on wording from a different window.
                shares_evidence = existing in provenance_related
                if self._describes_same_lesson(
                    candidate,
                    existing,
                    content_threshold=(
                        _SAME_EVIDENCE_CONTENT_THRESHOLD
                        if shares_evidence
                        else _CROSS_EVIDENCE_CONTENT_THRESHOLD
                    ),
                ):
                    matching_existing.append(existing)
            if not matching_existing and not provenance_related:
                independent_positions.add(position)
                continue
            if len(matching_existing) != 1:
                return None
            existing = matching_existing[0]
            existing_id = existing.user_playbook_id
            matched_existing_id_by_position[position] = existing_id
            current = matches.get(existing_id)
            if current is None:
                matches[existing_id] = (existing, [candidate])
            else:
                current[1].append(candidate)

        survivors: list[UserPlaybook] = []
        archive_ids: list[int] = []
        merge_groups: list[tuple[int, list[int]]] = []
        emitted_existing_ids: set[int] = set()
        for position, candidate in enumerate(new_playbooks):
            if position in independent_positions:
                survivors.append(
                    candidate.model_copy(update={"request_id": generation_request_id})
                )
                continue

            existing_id = matched_existing_id_by_position[position]
            if existing_id in emitted_existing_ids:
                continue
            emitted_existing_ids.add(existing_id)
            existing, candidates = matches[existing_id]
            members = [existing, *candidates]
            revised_candidates = [
                candidate
                for candidate in candidates
                if self._has_reviewer_revision(candidate)
            ]
            revised_shapes = {
                (candidate.content, candidate.trigger, candidate.rationale)
                for candidate in revised_candidates
            }
            if len(revised_shapes) > 1:
                # Multiple reviewer revisions disagree about the final wording;
                # defer to evidence-aware LLM consolidation instead of silently
                # selecting one by list order.
                return None
            wording_source = revised_candidates[0] if revised_candidates else existing
            survivor = wording_source.model_copy(
                update={
                    "user_playbook_id": 0,
                    "request_id": generation_request_id,
                    "created_at": int(datetime.now(UTC).timestamp()),
                    "source_interaction_ids": self._merge_source_ids(members),
                    "source_span": self._merge_source_spans(members),
                    "notes": self._merge_notes(members),
                    "reader_angle": self._shared_optional_value(
                        members, "reader_angle"
                    ),
                    "merged_into": None,
                    "superseded_by": None,
                }
            )
            survivors.append(survivor)
            archive_ids.append(existing_id)
            merge_groups.append((len(survivors) - 1, [existing_id]))

        return survivors, archive_ids, merge_groups

    @staticmethod
    def _find_suspicious_under_consumed_new_rows(
        new_playbooks: list[UserPlaybook],
        dedup_output: PlaybookConsolidationOutput,
    ) -> list[str]:
        candidates_by_id = {
            f"NEW-{idx}": playbook for idx, playbook in enumerate(new_playbooks)
        }
        # Every kind except reject_new stores a row for its consumed candidates
        # (raw for independent, refined-trigger copy for differentiate, the
        # re-synthesized decision content for a sibling unify), so any of them
        # can duplicate a unify's merged row. Only reject_new drops the row
        # outright and is never a duplicate-storage risk worth a repair call.
        consuming_kind_by_new_id = {
            new_id: decision.kind
            for decision in dedup_output.decisions
            for new_id in _decision_new_ids(decision)
        }
        source_ids_by_decision = [
            {
                source_id
                for new_id in _decision_new_ids(decision)
                if new_id in candidates_by_id
                for source_id in candidates_by_id[new_id].source_interaction_ids
            }
            for decision in dedup_output.decisions
        ]
        errors: list[str] = []

        for decision_index, decision in enumerate(dedup_output.decisions):
            if not isinstance(decision, UnifyDecision):
                continue
            consumed_ids = set(decision.new_ids)
            consumed_source_ids = source_ids_by_decision[decision_index]
            if not consumed_source_ids:
                continue

            # Candidates stored (near-)verbatim by another decision whose facts
            # this unify's merged content may have absorbed.
            for other_id, other in candidates_by_id.items():
                if other_id in consumed_ids:
                    continue
                if consuming_kind_by_new_id.get(other_id) not in (
                    "independent",
                    "differentiate",
                ):
                    continue
                if not consumed_source_ids.intersection(other.source_interaction_ids):
                    continue
                if check_string_token_overlap(decision.content, other.content):
                    errors.append(
                        f"decision[{decision_index}] unify likely absorbed facts from {other_id} but did not list it in new_id; "
                        "if the unified rule absorbed this row's facts, add its id to new_id; "
                        "if not, leave the decisions unchanged."
                    )

            # Sibling unify decisions each persist their own survivor row, so
            # two same-source unifies with heavily overlapping FINAL contents
            # are the same duplicate-storage class. Compare stored content to
            # stored content and flag each pair once (j > i).
            for sibling_index in range(decision_index + 1, len(dedup_output.decisions)):
                sibling = dedup_output.decisions[sibling_index]
                if not isinstance(sibling, UnifyDecision):
                    continue
                if not consumed_source_ids.intersection(
                    source_ids_by_decision[sibling_index]
                ):
                    continue
                if check_string_token_overlap(decision.content, sibling.content):
                    errors.append(
                        f"decision[{decision_index}] and decision[{sibling_index}] are both unify decisions over "
                        "same-source NEW rows with highly overlapping content; if they describe the same skill, "
                        "merge them into one unify decision whose new_id lists every consumed NEW id; "
                        "if they are genuinely distinct skills, leave the decisions unchanged."
                    )

        return errors

    # ===============================
    # Apply path: discriminated-union decisions -> (new rows, archive ids)
    # ===============================

    def _build_deduplicated_results(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
        dedup_output: PlaybookConsolidationOutput,
        generation_request_id: str | None = None,
        agent_version: str | None = None,  # noqa: ARG002
        *,
        request_id: str | None = None,
        validate_only: bool = False,
    ) -> tuple[list[UserPlaybook], list[int], list[tuple[int, list[int]]]]:
        """
        Build the deduplicated entry list from LLM decisions.

        Dispatches each ``ConsolidationDecision`` to its kind-specific apply
        method and accumulates resulting rows + archive ids. The batch fails
        unless every NEW playbook is referenced exactly once and every
        decision applies cleanly.

        Args:
            new_playbooks: Flattened list of new (candidate) entries.
            existing_playbooks: List of existing entries from the DB.
            dedup_output: LLM decisions output (discriminated union).
            generation_request_id: Request ID stamped onto newly-built rows.
            agent_version: Agent version (currently unused, kept for symmetry).

        Returns:
            Tuple of ``(entries ready to save, existing entry IDs to delete,
            merge_groups)``. ``merge_groups`` carries one
            ``(survivor_index, source_existing_ids)`` per ``unify`` decision
            that archives >= 1 existing row, where ``survivor_index`` indexes
            into the returned entries list (the unified survivor row) and the
            second element is the existing ids that decision supersedes. Only
            ``unify`` produces merge groups — it collapses N existing rows into
            one survivor. ``differentiate`` archives its split source but emits
            two rows (no single survivor), so its archived id appears in the
            delete set but NOT in any merge group.
        """
        generation_request_id = _normalize_generation_request_id(
            generation_request_id, request_id=request_id
        )
        partition_errors = validate_consolidation_output(new_playbooks, dedup_output)
        if partition_errors:
            raise ValueError("; ".join(partition_errors))

        candidates_by_id = {
            f"NEW-{idx}": playbook for idx, playbook in enumerate(new_playbooks)
        }
        existing_by_id = {
            playbook.user_playbook_id: playbook
            for playbook in existing_playbooks
            if playbook.user_playbook_id
        }
        existing_by_position = {
            f"EXISTING-{idx}": playbook
            for idx, playbook in enumerate(existing_playbooks)
        }

        result_counters = PlaybookConsolidationResult()
        archive_ids: list[int] = []
        seen_archive: set[int] = set()
        new_rows: list[UserPlaybook] = []
        handled_new_ids: set[str] = set()
        merge_groups: list[tuple[int, list[int]]] = []

        for decision in dedup_output.decisions:
            try:
                rows, marked_new_ids, merge_source_ids = self._apply_one(
                    decision=decision,
                    candidates_by_id=candidates_by_id,
                    existing_by_id=existing_by_id,
                    existing_by_position=existing_by_position,
                    archive_ids=archive_ids,
                    seen_archive=seen_archive,
                    generation_request_id=generation_request_id,
                    validate_only=validate_only,
                )
            except Exception as exc:  # noqa: BLE001 — add context, then fail the batch
                if validate_only:
                    raise
                result_counters.failed_count += 1
                raw_new_id = getattr(decision, "new_id", "unknown")
                new_id_str = (
                    _new_id_log_label(raw_new_id)
                    if isinstance(raw_new_id, (str, list))
                    else "unknown"
                )
                raw_existing_id = getattr(
                    decision,
                    "existing_id",
                    getattr(decision, "superseded_by_existing_id", "unknown"),
                )
                existing_id_str = (
                    _existing_id_log_label(raw_existing_id)
                    if isinstance(raw_existing_id, (int, list))
                    else "unknown"
                )
                with sentry_tags(
                    subsystem="playbook_consolidator",
                    op="apply_decision",
                    org_id=self.request_context.org_id,
                    request_id=generation_request_id,
                    kind=decision.kind,
                    new_id=new_id_str,
                    existing_id=existing_id_str,
                    error_type=type(exc).__name__,
                ):
                    logger.exception(
                        "event=consolidation_apply_failed kind=%s new_id=%s existing_id=%s",
                        decision.kind,
                        new_id_str,
                        existing_id_str,
                    )
                raise
            # Record the merge group BEFORE extending: the unified survivor is
            # the first (and only) row a ``unify`` decision emits, so its index
            # in the final list is the current length of ``new_rows``.
            if merge_source_ids:
                merge_groups.append((len(new_rows), merge_source_ids))
            if not isinstance(decision, IndependentDecision):
                self.consolidated_output_indices.update(
                    range(len(new_rows), len(new_rows) + len(rows))
                )
            new_rows.extend(rows)
            handled_new_ids.update(marked_new_ids)
            if not validate_only:
                self._bump_counter(result_counters, decision.kind)
                self._log_decision(
                    decision, candidates_by_id, existing_by_id, existing_by_position
                )

        unhandled_new_ids = set(candidates_by_id) - handled_new_ids
        if unhandled_new_ids:
            raise ValueError(
                "Consolidation apply left NEW ids unhandled: "
                + ", ".join(sorted(unhandled_new_ids))
            )

        overlap_errors = self._find_overlapping_survivors(new_rows)
        if overlap_errors:
            raise ValueError("; ".join(overlap_errors))

        if not validate_only:
            logger.info(
                "event=playbook_consolidation_done unify=%d reject_new=%d "
                "differentiate=%d independent=%d failed=%d",
                result_counters.unify_count,
                result_counters.reject_new_count,
                result_counters.differentiate_count,
                result_counters.independent_count,
                result_counters.failed_count,
            )

        return new_rows, archive_ids, merge_groups

    @staticmethod
    def _find_overlapping_survivors(
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook] | None = None,
    ) -> list[str]:
        """Reject semantic duplicates involving at least one newly produced row.

        Historical existing↔existing duplicates are outside this generation's
        write plan and must not permanently poison unrelated future batches.
        """
        errors: list[str] = []
        for left_index, left in enumerate(new_playbooks):
            for right_index in range(left_index + 1, len(new_playbooks)):
                right = new_playbooks[right_index]
                if PlaybookConsolidator._describes_same_lesson(
                    left,
                    right,
                    content_threshold=_SURVIVOR_CONTENT_THRESHOLD,
                    min_content_length=_SURVIVOR_MIN_CONTENT_LENGTH,
                ):
                    errors.append(
                        "Overlapping consolidation survivors share content and "
                        "evidence or trigger: "
                        f"result[{left_index}] and result[{right_index}]."
                    )
            for existing_index, existing in enumerate(existing_playbooks or []):
                if PlaybookConsolidator._describes_same_lesson(
                    left,
                    existing,
                    content_threshold=_SURVIVOR_CONTENT_THRESHOLD,
                    min_content_length=_SURVIVOR_MIN_CONTENT_LENGTH,
                ):
                    errors.append(
                        "Overlapping consolidation survivors share content and "
                        "evidence or trigger: "
                        f"result[{left_index}] and existing[{existing_index}]."
                    )
        return errors

    @staticmethod
    def _shares_evidence(left: UserPlaybook, right: UserPlaybook) -> bool:
        """Return whether two rows were drawn from any common interaction."""
        return bool(
            set(left.source_interaction_ids).intersection(right.source_interaction_ids)
        )

    @staticmethod
    def _describes_same_lesson(
        left: UserPlaybook,
        right: UserPlaybook,
        *,
        content_threshold: float,
        min_content_length: int = 0,
    ) -> bool:
        """Return whether two rows state the same lesson for the same situation.

        Two rows collide when their content overlaps AND they are anchored to the
        same situation — either literally the same evidence, or an equivalent
        future trigger. Content overlap alone is not enough: two genuinely
        different lessons about one task often share most of their vocabulary.

        Args:
            left (UserPlaybook): First row to compare.
            right (UserPlaybook): Second row to compare.
            content_threshold (float): Token-overlap ratio the contents must reach.
            min_content_length (int): Below this length, only exactly equal
                content counts — short rules share too much vocabulary for a
                token ratio to be meaningful.

        Returns:
            bool: True when the two rows describe the same lesson.
        """
        left_content = left.content.strip().casefold()
        right_content = right.content.strip().casefold()
        content_overlaps = left_content == right_content or (
            min(len(left_content), len(right_content)) >= min_content_length
            and check_string_token_overlap(
                left.content, right.content, threshold=content_threshold
            )
        )
        if not content_overlaps:
            return False
        same_trigger = check_string_token_overlap(
            left.trigger or "",
            right.trigger or "",
            threshold=_SAME_TRIGGER_THRESHOLD,
        )
        return PlaybookConsolidator._shares_evidence(left, right) or same_trigger

    def _apply_one(
        self,
        *,
        decision: ConsolidationDecision,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        generation_request_id: str,
        validate_only: bool = False,
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """Dispatch a single decision to its kind-specific apply method.

        Args:
            decision: The decision to apply (one of four kinds).
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate ``UserPlaybook``.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook
                (used by ``unify`` to resolve the EXISTING-M ids it archives in
                ``archive_existing_ids``).
            archive_ids: Accumulator list mutated with ids to archive/delete.
            seen_archive: Accumulator set guarding ``archive_ids`` against
                duplicate ids.
            generation_request_id: Request ID stamped onto newly-built rows.

        Returns:
            Tuple of ``(rows_to_insert, handled_new_ids, merge_source_ids)``.
            ``handled_new_ids`` is the set of ``"NEW-N"`` candidate ids consumed
            by this decision (used to suppress the safety fallback).
            ``merge_source_ids`` is non-empty ONLY for a ``unify`` decision that
            collapses >= 1 existing row into its single survivor (the first row
            in ``rows_to_insert``); for all other kinds it is ``[]`` because they
            either archive nothing or split into multiple rows with no single
            survivor.
        """
        if isinstance(decision, UnifyDecision):
            return self._apply_unify(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_position=existing_by_position,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
                generation_request_id=generation_request_id,
                validate_only=validate_only,
            )
        if isinstance(decision, RejectNewDecision):
            return self._apply_reject_new(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_id=existing_by_id,
                existing_by_position=existing_by_position,
                validate_only=validate_only,
            )
        if isinstance(decision, DifferentiateDecision):
            return self._apply_differentiate(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_id=existing_by_id,
                existing_by_position=existing_by_position,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
                generation_request_id=generation_request_id,
            )
        if isinstance(decision, IndependentDecision):
            return self._apply_independent(
                decision,
                candidates_by_id=candidates_by_id,
                generation_request_id=generation_request_id,
            )
        raise ValueError(f"unknown decision kind: {decision}")

    def _apply_unify(
        self,
        decision: UnifyDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        generation_request_id: str,
        validate_only: bool = False,
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """Collapse / compose NEW (+ 0..N EXISTING) into one row.

        Looks up each ``archive_existing_ids`` entry by position
        (``EXISTING-{idx}``) and archives it. The unified skill may carry
        mixed-polarity rules (do-rules and avoid-rules for different
        sub-aspects); there is **no** mechanical same-polarity check here. The
        no-self-contradiction judgment (do not merge rules that contradict on
        the same situation) is made by the LLM in the consolidation prompt, not
        the apply path. The new row is built by copying identity/metadata from
        the NEW candidate and overlaying ``content``, ``trigger``, and
        ``rationale`` from the decision.

        Args:
            decision: The ``UnifyDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook.
            archive_ids: Accumulator mutated with EXISTING ids to archive.
            seen_archive: Dedup set for ``archive_ids``.
            generation_request_id: Request ID stamped on the unified row.

        Returns:
            Tuple of ``([unified_row], [consumed NEW-N ids], merge_source_ids)``
            where ``merge_source_ids`` are the existing ids collapsed into the
            unified survivor (the returned row). The survivor identity is not
            known until the caller persists the row and reads its assigned id,
            so the merge is materialized by the caller, not here.

        Raises:
            KeyError: If any id in ``decision.new_ids`` does not resolve to a
                known candidate.
            ValueError: If an ``archive_existing_ids`` entry has no matching
                ``EXISTING-{idx}`` row in the position map.
        """
        new_ids = decision.new_ids
        candidates: list[UserPlaybook] = []
        for new_id in new_ids:
            candidate = candidates_by_id.get(new_id)
            if candidate is None:
                raise KeyError(f"unify references unknown NEW id: {new_id}")
            candidates.append(candidate)

        existing_members: list[UserPlaybook] = []
        for existing_position in decision.archive_existing_ids:
            existing = existing_by_position.get(f"EXISTING-{existing_position}")
            if existing is None:
                raise ValueError(
                    f"unify references unknown existing_id={existing_position}"
                )
            existing_members.append(existing)

        merge_source_ids: list[int] = []
        for existing in existing_members:
            pid = existing.user_playbook_id
            if pid and pid not in seen_archive:
                seen_archive.add(pid)
                archive_ids.append(pid)
            if pid:
                merge_source_ids.append(pid)

        budget = self._dedup_config.max_unified_content_chars
        content_len = len(decision.content)
        if content_len > budget and not validate_only:
            # Soft backstop only: the prompt instructs the model to prefer
            # `differentiate` over an over-long unify. We log a signal rather
            # than hard-fail or downgrade so we don't destabilize the 4-kind
            # apply logic; the merge still proceeds.
            logger.warning(
                "event=consolidation_over_budget new_id=%s len=%d budget=%d",
                ",".join(new_ids),
                content_len,
                budget,
            )

        primary_candidate = candidates[0]
        combined_source_ids = self._merge_source_ids([*candidates, *existing_members])
        source_members = [*candidates, *existing_members]
        unified_row = UserPlaybook(
            user_playbook_id=0,
            user_id=primary_candidate.user_id,
            agent_version=primary_candidate.agent_version,
            # Legacy storage/API field; value may be synthetic for manual/rerun flows.
            request_id=generation_request_id,
            playbook_name=primary_candidate.playbook_name,
            created_at=int(datetime.now(UTC).timestamp()),
            content=decision.content,
            trigger=decision.trigger,
            rationale=decision.rationale,
            status=primary_candidate.status,
            source=primary_candidate.source,
            source_interaction_ids=combined_source_ids,
            source_span=self._merge_source_spans(source_members),
            notes=self._merge_notes(source_members),
            reader_angle=self._shared_optional_value(source_members, "reader_angle"),
        )
        return [unified_row], new_ids, merge_source_ids

    def _apply_reject_new(
        self,
        decision: RejectNewDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        validate_only: bool = False,
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """No-op apply: the existing row(s) win and the new candidate(s) dropped.

        Resolve each superseding integer against the rendered ``EXISTING-N``
        position first, then as a DB ``user_playbook_id`` for backwards
        compatibility. If ANY referenced existing row does not resolve, the
        decision is treated as malformed: we log a warning and return
        ``([], [], [])`` so the safety fallback re-inserts every named
        candidate rather than silently dropping extracted data.

        Args:
            decision: The ``RejectNewDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook,
                used as a fallback for ``decision.superseded_by_existing_ids``.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook.

        Returns:
            Tuple of ``([], [consumed NEW-N ids], [])`` when every superseding
            id resolves, or ``([], [], [])`` when any is unknown. Never produces
            a merge group — the existing rows are kept as-is (no archive, no
            survivor).

        Raises:
            KeyError: If any id in ``decision.new_ids`` does not resolve to a
                known candidate.
        """
        new_ids = decision.new_ids
        missing_new_ids = [
            new_id for new_id in new_ids if new_id not in candidates_by_id
        ]
        if missing_new_ids:
            raise KeyError(f"reject_new references unknown NEW ids: {missing_new_ids}")

        existing_ids = decision.superseded_by_existing_ids
        existing_members = [
            self._resolve_existing_reference(
                existing_id,
                existing_by_position=existing_by_position,
                existing_by_id=existing_by_id,
            )
            for existing_id in existing_ids
        ]
        if any(existing is None for existing in existing_members):
            raise ValueError(
                "reject_new references unknown EXISTING ids: "
                + ", ".join(str(item) for item in existing_ids)
            )
        if not validate_only:
            logger.info(
                "event=consolidation_reject_new new_id=%s existing_id=%s",
                ",".join(new_ids),
                existing_ids,
            )
        return [], new_ids, []

    def _apply_differentiate(
        self,
        decision: DifferentiateDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        generation_request_id: str,
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """Archive the existing row and emit two refined rows in its place.

        Builds one ``UserPlaybook`` from the candidate's content/polarity with
        ``refined_new_trigger``, and a second from the existing row's
        content/polarity with ``refined_existing_trigger``. Polarity is
        threaded through unchanged for each side.

        Args:
            decision: The ``DifferentiateDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook.
            archive_ids: Accumulator mutated with the existing id to archive.
            seen_archive: Dedup set for ``archive_ids``.
            generation_request_id: Request ID stamped on both new rows.

        Returns:
            Tuple of ``([refined_new_row, refined_existing_row], [NEW-N id],
            [])``. ``differentiate`` is a SPLIT, not a merge: the existing row
            is archived but maps to no single survivor, so it produces NO merge
            group (its archived id is a pure-delete leftover for the caller).
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(
                f"differentiate references unknown NEW id: {decision.new_id}"
            )
        existing = self._resolve_existing_reference(
            decision.existing_id,
            existing_by_position=existing_by_position,
            existing_by_id=existing_by_id,
        )
        if existing is None:
            raise KeyError(
                f"differentiate references unknown EXISTING id: {decision.existing_id}"
            )

        existing_db_id = existing.user_playbook_id
        if existing_db_id and existing_db_id not in seen_archive:
            seen_archive.add(existing_db_id)
            archive_ids.append(existing_db_id)

        now_ts = int(datetime.now(UTC).timestamp())
        refined_candidate = candidate.model_copy(
            update={
                "user_playbook_id": 0,
                "request_id": generation_request_id,
                "trigger": decision.refined_new_trigger,
                "created_at": now_ts,
            }
        )
        refined_existing = existing.model_copy(
            update={
                "user_playbook_id": 0,
                "request_id": generation_request_id,
                "trigger": decision.refined_existing_trigger,
                "created_at": now_ts,
                "source_interaction_ids": list(existing.source_interaction_ids),
            }
        )
        return [refined_candidate, refined_existing], [decision.new_id], []

    def _apply_independent(
        self,
        decision: IndependentDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        generation_request_id: str,
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """Insert the new candidate unchanged; no archive.

        Args:
            decision: The ``IndependentDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.

        Returns:
            Tuple of ``([candidate row], [consumed NEW-N id], [])`` — no archive,
            so never a merge group.
        """
        rows: list[UserPlaybook] = []
        for new_id in decision.new_ids:
            candidate = candidates_by_id.get(new_id)
            if candidate is None:
                raise KeyError(f"independent references unknown NEW id: {new_id}")
            rows.append(
                candidate.model_copy(update={"request_id": generation_request_id})
            )
        return rows, decision.new_ids, []

    @staticmethod
    def _merge_source_ids(playbooks: list[UserPlaybook]) -> list[int]:
        """Combine ``source_interaction_ids`` across playbooks, preserving order.

        Args:
            playbooks: The playbooks whose source ids should be combined.

        Returns:
            Order-preserving deduplicated list of source interaction ids.
        """
        seen: set[int] = set()
        combined: list[int] = []
        for playbook in playbooks:
            for sid in playbook.source_interaction_ids:
                if sid not in seen:
                    seen.add(sid)
                    combined.append(sid)
        return combined

    @staticmethod
    def _merge_source_spans(playbooks: list[UserPlaybook]) -> str | None:
        """Combine non-empty evidence spans in stable order."""
        spans: list[str] = []
        for playbook in playbooks:
            # Extraction stores multiple independently validated spans separated
            # by blank lines. Split that storage representation before unioning
            # so overlapping replay windows do not preserve the same quotation
            # twice merely because each row also carried a different sibling span.
            for value in (playbook.source_span or "").split("\n\n"):
                value = value.strip()
                if value and value not in spans:
                    spans.append(value)
        return "\n\n".join(spans) or None

    @staticmethod
    def _has_reviewer_revision(playbook: UserPlaybook) -> bool:
        """Return whether metadata records a first-pass reviewer revision."""
        return _REVIEWER_REVISION_NOTE_PREFIX in (playbook.notes or "")

    @staticmethod
    def _merge_notes(playbooks: list[UserPlaybook]) -> str | None:
        """Combine note lines in stable order without erasing differing metadata."""
        lines: list[str] = []
        for playbook in playbooks:
            for line in (playbook.notes or "").splitlines():
                normalized = line.strip()
                if normalized and normalized not in lines:
                    lines.append(normalized)
        return "\n".join(lines) or None

    @staticmethod
    def _shared_optional_value(
        playbooks: list[UserPlaybook], field_name: str
    ) -> str | None:
        """Retain optional metadata only when all populated values agree."""
        values = {
            value.strip()
            for playbook in playbooks
            if isinstance((value := getattr(playbook, field_name)), str)
            and value.strip()
        }
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _resolve_existing_reference(
        raw_id: int,
        *,
        existing_by_position: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
    ) -> UserPlaybook | None:
        """Resolve an LLM existing-row integer.

        The rendered prompt labels rows as ``EXISTING-N`` and asks the model to
        emit bare integers, so position is the primary interpretation. DB id is
        retained as a compatibility fallback for older prompt outputs.
        """
        if 0 <= raw_id < len(existing_by_position):
            existing = existing_by_position.get(f"EXISTING-{raw_id}")
            if existing is not None:
                return existing
        return existing_by_id.get(raw_id)

    @staticmethod
    def _bump_counter(result: PlaybookConsolidationResult, kind: str) -> None:
        """Increment the per-kind counter on ``result`` for a successful apply.

        Args:
            result: The result counters object to mutate.
            kind: One of ``unify``, ``reject_new``, ``differentiate``, or
                ``independent``.
        """
        field = _COUNTER_BY_KIND[kind]
        setattr(result, field, getattr(result, field) + 1)

    @staticmethod
    def _log_decision(
        decision: ConsolidationDecision,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
    ) -> None:
        """Emit a structured per-decision log line for probe ingest.

        Emits ``playbook_consolidation.decision`` with the 4-kind name,
        new/existing ids, and trigger_match. Polarity is intentionally NOT
        derived or logged: under Option B a skill may hold mixed-polarity
        rules, so a single whole-content polarity label is no longer
        meaningful. The no-self-contradiction judgment lives in the LLM.

        Args:
            decision: The applied consolidation decision.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
        """
        kind = decision.kind
        raw_new_id = getattr(decision, "new_id", "")
        new_ids = (
            _new_ids_from_field(raw_new_id)
            if isinstance(raw_new_id, (str, list))
            else []
        )
        new_id_label = ",".join(new_ids)
        new_pb = candidates_by_id.get(new_ids[0]) if new_ids else None

        # UnifyDecision archives by position (EXISTING-{idx}) rather than a
        # single existing_id; log a synthetic "multi" so the probe parser sees
        # one line per decision regardless of arity.
        if isinstance(decision, UnifyDecision):
            existing_id_label: str = (
                "multi" if decision.archive_existing_ids else "none"
            )
            logger.info(
                "playbook_consolidation.decision kind=%s new_id=%s existing_id=%s "
                "trigger_match=%s",
                kind,
                new_id_label,
                existing_id_label,
                "unknown",
            )
            return

        existing_ids: list[int] = []
        if isinstance(decision, RejectNewDecision):
            existing_ids = decision.superseded_by_existing_ids
        elif isinstance(decision, DifferentiateDecision):
            existing_ids = [decision.existing_id]

        existing_pb = (
            PlaybookConsolidator._resolve_existing_reference(
                existing_ids[0],
                existing_by_position=existing_by_position,
                existing_by_id=existing_by_id,
            )
            if existing_ids
            else None
        )
        trigger_match = (
            new_pb is not None
            and existing_pb is not None
            and new_pb.trigger == existing_pb.trigger
        )
        existing_id_label = (
            ",".join(str(existing_id) for existing_id in existing_ids)
            if existing_ids
            else "none"
        )
        logger.info(
            "playbook_consolidation.decision kind=%s new_id=%s existing_id=%s "
            "trigger_match=%s",
            kind,
            new_id_label,
            existing_id_label,
            str(trigger_match).lower(),
        )
