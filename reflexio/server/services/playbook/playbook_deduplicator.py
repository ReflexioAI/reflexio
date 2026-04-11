"""
Playbook deduplication service that merges duplicate user playbook entries using LLM
and hybrid search against existing entries in the database.
"""

import logging
import os
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import StructuredData, UserPlaybook
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
from reflexio.server.services.playbook.playbook_service_utils import (
    StructuredPlaybookContent,
    ensure_playbook_content,
)

logger = logging.getLogger(__name__)


# ===============================
# Playbook-specific Pydantic Output Schemas for LLM
# ===============================


class PlaybookDeduplicationDuplicateGroup(BaseModel):
    """A group of duplicate playbook entries to merge, with old entries to delete."""

    item_ids: list[str] = Field(
        description="IDs of items in this group matching prompt format (e.g., 'NEW-0', 'EXISTING-1')"
    )
    merged_content: StructuredPlaybookContent = Field(
        description="Consolidated playbook entry in structured format (trigger, instruction, pitfall, blocking_issue)"
    )
    reasoning: str = Field(description="Brief explanation of the merge decision")

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class PlaybookDeduplicationOutput(BaseModel):
    """Output schema for playbook deduplication with NEW vs EXISTING merge support."""

    duplicate_groups: list[PlaybookDeduplicationDuplicateGroup] = Field(
        default=[], description="Groups of duplicate playbook entries to merge"
    )
    unique_ids: list[str] = Field(
        default=[], description="IDs of unique NEW entries (e.g., 'NEW-2')"
    )

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={"additionalProperties": False},
    )


class PlaybookDeduplicator(BaseDeduplicator):
    """
    Deduplicates new user playbook entries against each other and against existing entries
    in the database using hybrid search (vector + FTS) and LLM-based merging.
    """

    DEDUPLICATION_PROMPT_ID = "playbook_deduplication"

    def __init__(
        self,
        request_context: RequestContext,
        llm_client: LiteLLMClient,
        dedup_config: DeduplicationConfig | None = None,
    ):
        """
        Initialize the playbook deduplicator.

        Args:
            request_context: Request context with storage and prompt manager
            llm_client: Unified LLM client for LLM calls
            dedup_config: Optional deduplication search parameters (threshold, top_k)
        """
        super().__init__(request_context, llm_client)
        self._dedup_config = dedup_config or DeduplicationConfig()

    def _get_prompt_id(self) -> str:
        """Get the prompt ID for playbook deduplication."""
        return self.DEDUPLICATION_PROMPT_ID

    def _get_item_count_key(self) -> str:
        """Get the key name for item count in prompt variables."""
        return "new_playbook_count"

    def _get_items_key(self) -> str:
        """Get the key name for items in prompt variables."""
        return "new_playbooks"

    def _get_output_schema_class(self) -> type[BaseModel]:
        """Return PlaybookDeduplicationOutput for new/existing merge."""
        return PlaybookDeduplicationOutput

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

        For each new entry, uses its structured_data.trigger as the query with
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
            trigger = playbook.structured_data.trigger or playbook.content
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
            "Retrieved %d unique existing user playbook entries for deduplication",
            len(existing_playbooks),
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
        Deduplicate user playbook entries across extractors and against existing entries in DB.

        Args:
            results: List of entry lists from extractors (each extractor returns list[UserPlaybook])
            request_id: Request ID for context
            agent_version: Agent version for context
            user_id: Optional user ID to scope the existing entry search

        Returns:
            Tuple of (deduplicated entries, list of existing entry IDs to delete after save)
        """
        # Check if mock mode is enabled
        if os.getenv("MOCK_LLM_RESPONSE", "").lower() == "true":
            logger.info("Mock mode: skipping deduplication")
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
                "Playbook deduplication",
                [{"role": "user", "content": prompt}],
            )

            response = self.client.generate_chat_response(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=output_schema_class,
            )

            log_model_response(logger, "Deduplication response", response)

            if not isinstance(response, PlaybookDeduplicationOutput):
                logger.warning(
                    "Unexpected response type from deduplication LLM: %s",
                    type(response),
                )
                return new_playbooks, []

            dedup_output = response
        except Exception as e:
            logger.error("Failed to identify duplicates: %s", str(e))
            return new_playbooks, []

        if not dedup_output.duplicate_groups:
            logger.info(
                "No duplicate playbook entries found for request %s", request_id
            )
            return new_playbooks, []

        logger.info(
            "Found %d duplicate playbook groups for request %s",
            len(dedup_output.duplicate_groups),
            request_id,
        )

        # Build deduplicated result
        return self._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id=request_id,
            agent_version=agent_version,
        )

    def _build_deduplicated_results(  # noqa: C901
        self,
        new_playbooks: list[UserPlaybook],
        existing_playbooks: list[UserPlaybook],
        dedup_output: PlaybookDeduplicationOutput,
        request_id: str,
        agent_version: str,  # noqa: ARG002
    ) -> tuple[list[UserPlaybook], list[int]]:
        """
        Build the deduplicated entry list from LLM output.

        Handles merged groups (creating new entries from merged content)
        and unique entries. Returns IDs of existing entries to delete
        so the caller can delete them after save succeeds.

        Args:
            new_playbooks: Flattened list of new entries
            existing_playbooks: List of existing entries from DB
            dedup_output: LLM deduplication output
            request_id: Request ID
            agent_version: Agent version

        Returns:
            Tuple of (entries ready to save, existing entry IDs to delete)
        """
        handled_new_indices: set[int] = set()
        result_playbooks: list[UserPlaybook] = []
        existing_ids_to_delete: list[int] = []
        seen_delete_ids: set[int] = set()

        now_ts = int(datetime.now(UTC).timestamp())

        # Process duplicate groups
        for group in dedup_output.duplicate_groups:
            group_new_indices: list[int] = []
            group_existing_indices: list[int] = []

            for item_id in group.item_ids:
                parsed = parse_item_id(item_id)
                if parsed is None:
                    continue
                prefix, idx = parsed
                if prefix == "NEW":
                    group_new_indices.append(idx)
                    handled_new_indices.add(idx)
                elif prefix == "EXISTING":
                    group_existing_indices.append(idx)

            # Collect existing entry IDs to delete (deduplicated)
            for eidx in group_existing_indices:
                if 0 <= eidx < len(existing_playbooks):
                    fb_id = existing_playbooks[eidx].user_playbook_id
                    if fb_id and fb_id not in seen_delete_ids:
                        seen_delete_ids.add(fb_id)
                        existing_ids_to_delete.append(fb_id)

            # Get template from first NEW entry in group (for metadata)
            template_playbook: UserPlaybook | None = None
            if group_new_indices:
                first_new_idx = group_new_indices[0]
                if 0 <= first_new_idx < len(new_playbooks):
                    template_playbook = new_playbooks[first_new_idx]

            if template_playbook is None:
                # Fallback: use first existing entry as template
                if group_existing_indices:
                    for eidx in group_existing_indices:
                        if 0 <= eidx < len(existing_playbooks):
                            template_playbook = existing_playbooks[eidx]
                            break
                if template_playbook is None:
                    logger.warning("Could not find template entry for group, skipping")
                    continue

            # Combine source_interaction_ids from all NEW entries in group
            combined_source_ids: list[int] = []
            seen_ids: set[int] = set()
            for idx in group_new_indices:
                if 0 <= idx < len(new_playbooks):
                    for sid in new_playbooks[idx].source_interaction_ids:
                        if sid not in seen_ids:
                            combined_source_ids.append(sid)
                            seen_ids.add(sid)

            # Also include source_interaction_ids from existing entries being merged
            for eidx in group_existing_indices:
                if 0 <= eidx < len(existing_playbooks):
                    for sid in existing_playbooks[eidx].source_interaction_ids:
                        if sid not in seen_ids:
                            combined_source_ids.append(sid)
                            seen_ids.add(sid)

            # Format content from merged structured content
            merged_content = group.merged_content
            playbook_content = ensure_playbook_content(
                merged_content.content, merged_content
            )
            logger.info(
                "Deduplicated playbook content (freeform): %.200s",
                playbook_content,
            )

            embedding_text = merged_content.trigger or merged_content.content or ""

            merged_playbook = UserPlaybook(
                user_playbook_id=0,  # Will be assigned by storage
                user_id=template_playbook.user_id,
                agent_version=template_playbook.agent_version,
                request_id=request_id,
                playbook_name=template_playbook.playbook_name,
                created_at=now_ts,
                content=playbook_content,
                structured_data=StructuredData(
                    rationale=merged_content.rationale,
                    trigger=merged_content.trigger,
                    instruction=merged_content.instruction,
                    pitfall=merged_content.pitfall,
                    blocking_issue=merged_content.blocking_issue,
                    embedding_text=embedding_text,
                ),
                status=template_playbook.status,
                source=template_playbook.source,
                source_interaction_ids=combined_source_ids,
            )
            result_playbooks.append(merged_playbook)

        # Add unique NEW entries
        for uid in dedup_output.unique_ids:
            parsed = parse_item_id(uid)
            if parsed is None:
                continue
            prefix, idx = parsed
            if (
                prefix == "NEW"
                and idx not in handled_new_indices
                and 0 <= idx < len(new_playbooks)
            ):
                result_playbooks.append(new_playbooks[idx])
                handled_new_indices.add(idx)

        # Safety fallback: add any NEW entries not mentioned by LLM
        for idx, playbook in enumerate(new_playbooks):
            if idx not in handled_new_indices:
                logger.warning(
                    "New entry at index %d was not handled by LLM, adding as-is",
                    idx,
                )
                result_playbooks.append(playbook)

        return result_playbooks, existing_ids_to_delete
