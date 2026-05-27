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
from reflexio.server.services.deduplication_utils import (
    BaseDeduplicator,
    format_dedup_timestamp,
    parse_item_id,
)

logger = logging.getLogger(__name__)


# ===============================
# Playbook-specific Pydantic Output Schemas for LLM
# ===============================


class DuplicateDecision(BaseModel):
    """Multiple rows collapse into one merged row (same intent)."""

    kind: Literal["duplicate"] = "duplicate"
    item_ids: list[str] = Field(
        description="Mix of 'NEW-N' / 'EXISTING-M' ids in the duplicate group"
    )
    merged_content: str
    merged_trigger: str
    merged_rationale: str
    merged_polarity: Literal["positive", "negative"]
    reason: str = ""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class PreferNewDecision(BaseModel):
    """The new candidate wins: archive existing, insert new as-is."""

    kind: Literal["prefer_new"] = "prefer_new"
    new_id: str
    existing_id: int
    reason: str = ""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class PreferExistingDecision(BaseModel):
    """The existing row wins: drop the new candidate (storage no-op)."""

    kind: Literal["prefer_existing"] = "prefer_existing"
    new_id: str
    existing_id: int
    reason: str = ""

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class DifferentiateDecision(BaseModel):
    """Both rules valid in distinct contexts: refine both triggers."""

    kind: Literal["differentiate"] = "differentiate"
    new_id: str
    existing_id: int
    refined_new_trigger: str
    refined_existing_trigger: str
    reason: str = ""

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
    DuplicateDecision
    | PreferNewDecision
    | PreferExistingDecision
    | DifferentiateDecision
    | IndependentDecision,
    Field(discriminator="kind"),
]


class PlaybookConsolidationOutput(BaseModel):
    """Output schema for playbook consolidation as a 5-kind discriminated union.

    Each decision is one of ``DuplicateDecision``, ``PreferNewDecision``,
    ``PreferExistingDecision``, ``DifferentiateDecision``, or
    ``IndependentDecision``; the ``kind`` literal selects the concrete shape.
    """

    decisions: list[ConsolidationDecision] = Field(default_factory=list)

    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})


class PlaybookConsolidationResult(BaseModel):
    """Per-kind counters tracked over one consolidation batch.

    Bumped once per successfully-applied decision; ``failed_count`` is bumped
    when a single decision's apply path raises, allowing the rest of the batch
    to proceed unaffected.
    """

    duplicates_count: int = 0
    prefer_new_count: int = 0
    prefer_existing_count: int = 0
    differentiates_count: int = 0
    independents_count: int = 0
    failed_count: int = 0


_COUNTER_BY_KIND: dict[str, str] = {
    "duplicate": "duplicates_count",
    "prefer_new": "prefer_new_count",
    "prefer_existing": "prefer_existing_count",
    "differentiate": "differentiates_count",
    "independent": "independents_count",
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
            lines.append(
                f'[{prefix}-{idx}] Content: "{playbook.content}" | Name: {playbook_name} | Source: {source} | Last Modified: {created_date}'
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

    def deduplicate(
        self,
        results: list[list[UserPlaybook]],
        request_id: str,
        agent_version: str,
        user_id: str | None = None,
    ) -> tuple[list[UserPlaybook], list[int]]:
        """
        Consolidate user playbook entries across extractors and against existing entries in DB.

        Args:
            results: List of entry lists from extractors (each extractor returns list[UserPlaybook])
            request_id: Request ID for context
            agent_version: Agent version for context
            user_id: Optional user ID to scope the existing entry search

        Returns:
            Tuple of (consolidated entries, list of existing entry IDs to delete after save)
        """
        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: skipping consolidation")
            all_playbooks: list[UserPlaybook] = []
            for result in results:
                if isinstance(result, list):
                    all_playbooks.extend(result)
            return all_playbooks, []

        # Flatten all new entries
        new_playbooks: list[UserPlaybook] = []
        for result in results:
            if isinstance(result, list):
                new_playbooks.extend(result)

        if not new_playbooks:
            return [], []

        # Retrieve existing entries via hybrid search
        existing_playbooks = self._retrieve_existing_playbooks(
            new_playbooks, user_id=user_id, agent_version=agent_version
        )

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
                "existing_playbook_count": len(existing_playbooks),
                "existing_playbooks": existing_text,
            },
        )

        output_schema_class = self._get_output_schema_class()

        try:
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
                return new_playbooks, []

            dedup_output = response
        except Exception as e:
            logger.error("Failed to identify duplicates: %s", str(e))
            return new_playbooks, []

        if not dedup_output.decisions:
            logger.info(
                "No consolidation decisions returned for request %s", request_id
            )
            return new_playbooks, []

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
    ) -> tuple[list[UserPlaybook], list[int]]:
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
            Tuple of (entries ready to save, existing entry IDs to delete).
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

        for decision in dedup_output.decisions:
            try:
                rows, marked_new_ids = self._apply_one(
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
                logger.warning(
                    "event=consolidation_apply_failed kind=%s error_type=%s error=%s",
                    decision.kind,
                    type(exc).__name__,
                    exc,
                )
                continue
            new_rows.extend(rows)
            handled_new_ids.update(marked_new_ids)
            self._bump_counter(result_counters, decision.kind)

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
            "event=playbook_consolidation_done duplicates=%d prefer_new=%d "
            "prefer_existing=%d differentiates=%d independents=%d failed=%d",
            result_counters.duplicates_count,
            result_counters.prefer_new_count,
            result_counters.prefer_existing_count,
            result_counters.differentiates_count,
            result_counters.independents_count,
            result_counters.failed_count,
        )

        return new_rows, archive_ids

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
    ) -> tuple[list[UserPlaybook], list[str]]:
        """Dispatch a single decision to its kind-specific apply method.

        Args:
            decision: The decision to apply (one of five kinds).
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate ``UserPlaybook``.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook
                (used by ``duplicate`` to resolve EXISTING-M ids in ``item_ids``).
            archive_ids: Accumulator list mutated with ids to archive/delete.
            seen_archive: Accumulator set guarding ``archive_ids`` against
                duplicate ids.
            request_id: Request ID stamped onto newly-built rows.

        Returns:
            Tuple of ``(rows_to_insert, handled_new_ids)`` where the second
            element is the set of ``"NEW-N"`` candidate ids consumed by this
            decision (used to suppress the safety fallback).
        """
        if isinstance(decision, DuplicateDecision):
            return self._apply_duplicate(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_position=existing_by_position,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
                request_id=request_id,
            )
        if isinstance(decision, PreferNewDecision):
            return self._apply_prefer_new(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_id=existing_by_id,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
            )
        if isinstance(decision, PreferExistingDecision):
            return self._apply_prefer_existing(decision)
        if isinstance(decision, DifferentiateDecision):
            return self._apply_differentiate(
                decision,
                candidates_by_id=candidates_by_id,
                existing_by_id=existing_by_id,
                archive_ids=archive_ids,
                seen_archive=seen_archive,
                request_id=request_id,
            )
        if isinstance(decision, IndependentDecision):
            return self._apply_independent(decision, candidates_by_id=candidates_by_id)
        raise ValueError(f"unknown decision kind: {decision}")

    def _apply_duplicate(
        self,
        decision: DuplicateDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_position: dict[str, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        request_id: str,
    ) -> tuple[list[UserPlaybook], list[str]]:
        """Collapse multiple rows into one merged row.

        Archives every ``EXISTING-M`` member's id and emits one new
        ``UserPlaybook`` built from a template (first ``NEW-N`` member, or the
        first ``EXISTING-M`` if no NEW members) with the LLM-supplied merged
        fields. ``polarity`` is taken from ``decision.merged_polarity``.

        Args:
            decision: The ``DuplicateDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_position: Mapping ``"EXISTING-M"`` -> existing playbook.
            archive_ids: Accumulator mutated with EXISTING ids to archive.
            seen_archive: Dedup set for ``archive_ids``.
            request_id: Request ID stamped on the merged row.

        Returns:
            Tuple of ([merged_row], [consumed NEW-N ids]).
        """
        new_members: list[UserPlaybook] = []
        existing_members: list[UserPlaybook] = []
        handled_new_ids: list[str] = []

        for item_id in decision.item_ids:
            parsed = parse_item_id(item_id)
            if parsed is None:
                continue
            prefix, _idx = parsed
            if prefix == "NEW" and item_id in candidates_by_id:
                new_members.append(candidates_by_id[item_id])
                handled_new_ids.append(item_id)
            elif prefix == "EXISTING" and item_id in existing_by_position:
                existing_members.append(existing_by_position[item_id])

        template = (
            new_members[0]
            if new_members
            else (existing_members[0] if existing_members else None)
        )
        if template is None:
            logger.warning(
                "event=consolidation_duplicate_no_template item_ids=%s",
                decision.item_ids,
            )
            return [], []

        for existing in existing_members:
            pid = existing.user_playbook_id
            if pid and pid not in seen_archive:
                seen_archive.add(pid)
                archive_ids.append(pid)

        combined_source_ids = self._merge_source_ids(new_members + existing_members)
        merged_row = UserPlaybook(
            user_playbook_id=0,
            user_id=template.user_id,
            agent_version=template.agent_version,
            request_id=request_id,
            playbook_name=template.playbook_name,
            created_at=int(datetime.now(UTC).timestamp()),
            content=decision.merged_content,
            trigger=decision.merged_trigger,
            rationale=decision.merged_rationale,
            blocking_issue=template.blocking_issue,
            polarity=decision.merged_polarity,
            status=template.status,
            source=template.source,
            source_interaction_ids=combined_source_ids,
        )
        return [merged_row], handled_new_ids

    def _apply_prefer_new(
        self,
        decision: PreferNewDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
    ) -> tuple[list[UserPlaybook], list[str]]:
        """Archive the existing row and insert the new candidate unchanged.

        Args:
            decision: The ``PreferNewDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
            archive_ids: Accumulator mutated with the existing id to archive.
            seen_archive: Dedup set for ``archive_ids``.

        Returns:
            Tuple of ([candidate row], [consumed NEW-N id]).
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(f"prefer_new references unknown NEW id: {decision.new_id}")
        if (
            decision.existing_id in existing_by_id
            and decision.existing_id not in seen_archive
        ):
            seen_archive.add(decision.existing_id)
            archive_ids.append(decision.existing_id)
        return [candidate], [decision.new_id]

    def _apply_prefer_existing(
        self,
        decision: PreferExistingDecision,
    ) -> tuple[list[UserPlaybook], list[str]]:
        """No-op apply: the existing row wins and the new candidate is dropped.

        Args:
            decision: The ``PreferExistingDecision`` to apply.

        Returns:
            Tuple of ([], [consumed NEW-N id]) — the new id is marked handled
            so the safety fallback does not re-insert the dropped candidate.
        """
        logger.info(
            "event=consolidation_prefer_existing new_id=%s existing_id=%d",
            decision.new_id,
            decision.existing_id,
        )
        return [], [decision.new_id]

    def _apply_differentiate(
        self,
        decision: DifferentiateDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
        existing_by_id: dict[int, UserPlaybook],
        archive_ids: list[int],
        seen_archive: set[int],
        request_id: str,
    ) -> tuple[list[UserPlaybook], list[str]]:
        """Archive the existing row and emit two refined rows in its place.

        Builds one ``UserPlaybook`` from the candidate's content/polarity with
        ``refined_new_trigger``, and a second from the existing row's
        content/polarity with ``refined_existing_trigger``. Polarity is
        threaded through unchanged for each side.

        Args:
            decision: The ``DifferentiateDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.
            existing_by_id: Mapping ``user_playbook_id`` -> existing playbook.
            archive_ids: Accumulator mutated with the existing id to archive.
            seen_archive: Dedup set for ``archive_ids``.
            request_id: Request ID stamped on both new rows.

        Returns:
            Tuple of ([refined_new_row, refined_existing_row], [NEW-N id]).
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(
                f"differentiate references unknown NEW id: {decision.new_id}"
            )
        existing = existing_by_id.get(decision.existing_id)
        if existing is None:
            raise KeyError(
                f"differentiate references unknown EXISTING id: {decision.existing_id}"
            )

        if decision.existing_id not in seen_archive:
            seen_archive.add(decision.existing_id)
            archive_ids.append(decision.existing_id)

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
        return [refined_candidate, refined_existing], [decision.new_id]

    def _apply_independent(
        self,
        decision: IndependentDecision,
        *,
        candidates_by_id: dict[str, UserPlaybook],
    ) -> tuple[list[UserPlaybook], list[str]]:
        """Insert the new candidate unchanged; no archive.

        Args:
            decision: The ``IndependentDecision`` to apply.
            candidates_by_id: Mapping ``"NEW-N"`` -> candidate playbook.

        Returns:
            Tuple of ([candidate row], [consumed NEW-N id]).
        """
        candidate = candidates_by_id.get(decision.new_id)
        if candidate is None:
            raise KeyError(f"independent references unknown NEW id: {decision.new_id}")
        return [candidate], [decision.new_id]

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
    def _bump_counter(result: PlaybookConsolidationResult, kind: str) -> None:
        """Increment the per-kind counter on ``result`` for a successful apply.

        Args:
            result: The result counters object to mutate.
            kind: One of ``duplicate``, ``prefer_new``, ``prefer_existing``,
                ``differentiate``, or ``independent``.
        """
        field = _COUNTER_BY_KIND[kind]
        setattr(result, field, getattr(result, field) + 1)
