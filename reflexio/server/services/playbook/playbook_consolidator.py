"""
Playbook consolidation service that merges duplicate user playbook entries using LLM
and hybrid search against existing entries in the database.
"""

import logging
import os
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import (
    EMBEDDING_DIMENSIONS,
    DeduplicationConfig,
    SearchOptions,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.llm.llm_utils import ProviderSafeUnionMixin
from reflexio.server.services.deduplication_utils import (
    BaseDeduplicator,
    format_dedup_timestamp,
)
from reflexio.server.tracing import sentry_tags

logger = logging.getLogger(__name__)


# ===============================
# Playbook-specific Pydantic Output Schemas for LLM
# ===============================


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
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"existing-position must be >= 0, got {value!r}")
        return value
    if isinstance(value, str):
        stripped = value.strip()
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
    new_id: str
    archive_existing_ids: list[int] = Field(default_factory=list)
    content: str
    trigger: str
    rationale: str
    reason: str = ""

    @field_validator("archive_existing_ids", mode="before")
    @classmethod
    def _coerce_archive_ids(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, list):
            return [_coerce_existing_position(item) for item in value]
        return value

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
    new_id: str
    superseded_by_existing_id: int
    reason: str = ""

    @field_validator("superseded_by_existing_id", mode="before")
    @classmethod
    def _coerce_superseded_id(cls, value: object) -> int:
        return _coerce_existing_position(value)

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

    @field_validator("existing_id", mode="before")
    @classmethod
    def _coerce_existing_id(cls, value: object) -> int:
        return _coerce_existing_position(value)

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
    new_id: str
    reason: str = ""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


ConsolidationDecision = Annotated[
    UnifyDecision | RejectNewDecision | DifferentiateDecision | IndependentDecision,
    Field(discriminator="kind"),
]


# ``PlaybookConsolidationOutput`` mixes in ``ProviderSafeUnionMixin`` so the
# emitted JSON schema is folded into the provider-accepted union shape while the
# discriminator is kept for keyed validation at parse time (Sentry
# PYTHON-FASTAPI-9J). This note is a comment, NOT part of the docstring, because
# the docstring is serialized into the wire schema's ``description`` sent to the
# model — keep implementation tokens out of it.
class PlaybookConsolidationOutput(ProviderSafeUnionMixin, BaseModel):
    """Output schema for playbook consolidation as a 4-kind tagged union.

    Each decision is one of ``UnifyDecision``, ``RejectNewDecision``,
    ``DifferentiateDecision``, or ``IndependentDecision``; the ``kind`` field
    selects the concrete shape.
    """

    decisions: list[ConsolidationDecision] = Field(default_factory=list)

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
        self, playbooks: list[UserPlaybook], prefix: str
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
                f" | Name: {playbook_name}"
                f" | Source: {source} | Last Modified: {created_date}"
            )
        return "\n".join(lines)

    def _retrieve_existing_playbooks(
        self,
        new_playbooks: list[UserPlaybook],
        user_id: str | None = None,
        agent_version: str | None = None,
    ) -> list[UserPlaybook]:
        """
        Retrieve existing user playbook entries from the database using hybrid search.

        For each new entry, uses its trigger field as the query with
        pre-computed embeddings for vector search.

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

        # Batch-generate embeddings
        try:
            embeddings = self.client.get_embeddings(
                query_texts, dimensions=EMBEDDING_DIMENSIONS
            )
        except Exception as e:
            logger.warning("Failed to generate embeddings for dedup search: %s", e)
            # Fall back to text-only search
            embeddings = [None] * len(query_texts)

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
            except Exception as e:  # noqa: PERF203
                logger.warning(
                    "Failed to search existing entries for query %d: %s", i, e
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
        new_text = self._format_playbooks_with_prefix(new_playbooks, "NEW")
        existing_text = self._format_playbooks_with_prefix(
            existing_playbooks, "EXISTING"
        )
        return new_text, existing_text

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
        # Format for prompt
        new_text, existing_text = self._format_new_and_existing_for_prompt(
            new_playbooks, existing_playbooks
        )

        # Build and call LLM
        prompt = self.request_context.prompt_manager.render_prompt(
            self._get_prompt_id(),
            {
                "new_playbook_count": len(new_playbooks),
                "new_playbooks": new_text,
                "existing_playbooks": existing_text,
            },
        )

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

        response = self.client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format=output_schema_class,
        )

        log_model_response(logger, "Consolidation response", response)

        if not isinstance(response, PlaybookConsolidationOutput):
            logger.warning(
                "Unexpected response type from consolidation LLM: %s",
                type(response),
            )
            return PlaybookConsolidationOutput()

        return response

    def deduplicate(
        self,
        results: list[list[UserPlaybook]],
        request_id: str,
        agent_version: str,
        user_id: str | None = None,
    ) -> tuple[list[UserPlaybook], list[int], list[tuple[int, list[int]]]]:
        """
        Consolidate user playbook entries across extractors and against existing entries in DB.

        Args:
            results: List of entry lists from extractors (each extractor returns list[UserPlaybook])
            request_id: Request ID for context
            agent_version: Agent version for context
            user_id: Optional user ID to scope the existing entry search

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

        # Retrieve existing entries via hybrid search
        existing_playbooks = self._retrieve_existing_playbooks(
            new_playbooks, user_id=user_id, agent_version=agent_version
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
                request_id=request_id,
                agent_version=agent_version,
                error_type=type(e).__name__,
            ):
                logger.exception("Failed to identify duplicates")
            return new_playbooks, [], []

        if not dedup_output.decisions:
            logger.info(
                "No consolidation decisions returned for request %s", request_id
            )
            return new_playbooks, [], []

        logger.info(
            "Received %d consolidation decisions for request %s",
            len(dedup_output.decisions),
            request_id,
        )

        # Build consolidated result via the discriminated-union apply path
        return self._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id=request_id,
            agent_version=agent_version,
        )

    # ===============================
    # Apply path: discriminated-union decisions -> (new rows, archive ids)
    # ===============================

    def _build_deduplicated_results(
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
        dedup_output: PlaybookConsolidationOutput,
        request_id: str,
        agent_version: str,  # noqa: ARG002
    ) -> tuple[list[UserPlaybook], list[int], list[tuple[int, list[int]]]]:
        """
        Build the deduplicated entry list from LLM decisions.

        Dispatches each ``ConsolidationDecision`` to its kind-specific apply
        method, accumulates resulting rows + archive ids, and adds any NEW
        playbooks the LLM didn't reference as a safety fallback so a
        misbehaving LLM cannot silently drop extracted playbooks.

        Args:
            new_playbooks: Flattened list of new (candidate) entries.
            existing_playbooks: List of existing entries from the DB.
            dedup_output: LLM decisions output (discriminated union).
            request_id: Request ID stamped onto newly-built rows.
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
                    request_id=request_id,
                )
            except Exception as exc:  # noqa: BLE001 — per-decision isolation
                result_counters.failed_count += 1
                new_id_str = getattr(decision, "new_id", "unknown")
                existing_id_str = getattr(decision, "existing_id", "unknown")
                with sentry_tags(
                    subsystem="playbook_consolidator",
                    op="apply_decision",
                    org_id=self.request_context.org_id,
                    request_id=request_id,
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
                continue
            # Record the merge group BEFORE extending: the unified survivor is
            # the first (and only) row a ``unify`` decision emits, so its index
            # in the final list is the current length of ``new_rows``.
            if merge_source_ids:
                merge_groups.append((len(new_rows), merge_source_ids))
            new_rows.extend(rows)
            handled_new_ids.update(marked_new_ids)
            self._bump_counter(result_counters, decision.kind)
            self._log_decision(
                decision, candidates_by_id, existing_by_id, existing_by_position
            )

        # Safety fallback: add any NEW entries the LLM did not reference, so a
        # misbehaving model cannot silently drop extracted playbooks.
        for new_id, candidate in candidates_by_id.items():
            if new_id not in handled_new_ids:
                logger.warning(
                    "event=consolidation_unhandled_new id=%s — adding as-is",
                    new_id,
                )
                new_rows.append(candidate)

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

    def _apply_one(
        self,
        *,
        decision: ConsolidationDecision,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        request_id: str,
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
            request_id: Request ID stamped onto newly-built rows.

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
                request_id=request_id,
            )
        if isinstance(decision, RejectNewDecision):
            return self._apply_reject_new(
                decision,
                existing_by_id=existing_by_id,
                existing_by_position=existing_by_position,
            )
        if isinstance(decision, DifferentiateDecision):
            return self._apply_differentiate(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_id=existing_by_id,
                existing_by_position=existing_by_position,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
                request_id=request_id,
            )
        if isinstance(decision, IndependentDecision):
            return self._apply_independent(decision, candidates_by_id=candidates_by_id)
        raise ValueError(f"unknown decision kind: {decision}")

    def _apply_unify(
        self,
        decision: UnifyDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        request_id: str,
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
            request_id: Request ID stamped on the unified row.

        Returns:
            Tuple of ``([unified_row], [consumed NEW-N ids], merge_source_ids)``
            where ``merge_source_ids`` are the existing ids collapsed into the
            unified survivor (the returned row). The survivor identity is not
            known until the caller persists the row and reads its assigned id,
            so the merge is materialized by the caller, not here.

        Raises:
            KeyError: If ``decision.new_id`` does not resolve to a known
                candidate.
            ValueError: If an ``archive_existing_ids`` entry has no matching
                ``EXISTING-{idx}`` row in the position map.
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(f"unify references unknown NEW id: {decision.new_id}")

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
        if content_len > budget:
            # Soft backstop only: the prompt instructs the model to prefer
            # `differentiate` over an over-long unify. We log a signal rather
            # than hard-fail or downgrade so we don't destabilize the 4-kind
            # apply logic; the merge still proceeds.
            logger.warning(
                "event=consolidation_over_budget new_id=%s len=%d budget=%d",
                decision.new_id,
                content_len,
                budget,
            )

        combined_source_ids = self._merge_source_ids([candidate, *existing_members])
        unified_row = UserPlaybook(
            user_playbook_id=0,
            user_id=candidate.user_id,
            agent_version=candidate.agent_version,
            request_id=request_id,
            playbook_name=candidate.playbook_name,
            created_at=int(datetime.now(UTC).timestamp()),
            content=decision.content,
            trigger=decision.trigger,
            rationale=decision.rationale,
            status=candidate.status,
            source=candidate.source,
            source_interaction_ids=combined_source_ids,
        )
        return [unified_row], [decision.new_id], merge_source_ids

    def _apply_reject_new(
        self,
        decision: RejectNewDecision,
        *,
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """No-op apply: the existing row wins and the new candidate is dropped.

        Resolve the integer against the rendered ``EXISTING-N`` position first,
        then as a DB ``user_playbook_id`` for backwards compatibility. If it
        does not resolve to a known existing row, the decision is treated as
        malformed: we log a warning and return ``([], [])`` so the safety
        fallback re-inserts the candidate rather than silently dropping
        extracted data.

        Args:
            decision: The ``RejectNewDecision`` to apply.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook,
                used as a fallback for ``decision.superseded_by_existing_id``.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook.

        Returns:
            Tuple of ``([], [consumed NEW-N id], [])`` when the existing id
            resolves, or ``([], [], [])`` when the existing id is unknown.
            Never produces a merge group — the existing row is kept as-is (no
            archive, no survivor).
        """
        existing = self._resolve_existing_reference(
            decision.superseded_by_existing_id,
            existing_by_position=existing_by_position,
            existing_by_id=existing_by_id,
        )
        if existing is None:
            logger.warning(
                "event=consolidation_reject_new_invalid new_id=%s existing_id=%d",
                decision.new_id,
                decision.superseded_by_existing_id,
            )
            return [], [], []
        logger.info(
            "event=consolidation_reject_new new_id=%s existing_id=%d",
            decision.new_id,
            decision.superseded_by_existing_id,
        )
        return [], [decision.new_id], []

    def _apply_differentiate(
        self,
        decision: DifferentiateDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        request_id: str,
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
            request_id: Request ID stamped on both new rows.

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
                "request_id": request_id,
                "trigger": decision.refined_new_trigger,
                "created_at": now_ts,
            }
        )
        refined_existing = existing.model_copy(
            update={
                "user_playbook_id": 0,
                "request_id": request_id,
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
    ) -> tuple[list[UserPlaybook], list[str], list[int]]:
        """Insert the new candidate unchanged; no archive.

        Args:
            decision: The ``IndependentDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.

        Returns:
            Tuple of ``([candidate row], [consumed NEW-N id], [])`` — no archive,
            so never a merge group.
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(f"independent references unknown NEW id: {decision.new_id}")
        return [candidate], [decision.new_id], []

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
        new_id: str = getattr(decision, "new_id", "")
        new_pb = candidates_by_id.get(new_id)

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
                new_id,
                existing_id_label,
                "unknown",
            )
            return

        # RejectNewDecision exposes ``superseded_by_existing_id``; the other
        # two surviving kinds expose ``existing_id`` directly.
        existing_id_raw: int = getattr(
            decision,
            "existing_id",
            getattr(decision, "superseded_by_existing_id", 0),
        )
        existing_pb = PlaybookConsolidator._resolve_existing_reference(
            existing_id_raw,
            existing_by_position=existing_by_position,
            existing_by_id=existing_by_id,
        )
        trigger_match = (
            new_pb is not None
            and existing_pb is not None
            and new_pb.trigger == existing_pb.trigger
        )
        logger.info(
            "playbook_consolidation.decision kind=%s new_id=%s existing_id=%s "
            "trigger_match=%s",
            kind,
            new_id,
            existing_id_raw,
            str(trigger_match).lower(),
        )
