"""Candidate retrieval and audited playbook consolidation."""

import logging

from reflexio.models.api_schema.retriever_schema import SearchUserPlaybookRequest
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.models.config_schema import DeduplicationConfig, SearchOptions
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_bool
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.deduplication_utils import (
    BaseDeduplicator,
    resolve_dedup_query_embeddings,
)
from reflexio.server.services.playbook.source_quoted_overflow import (
    is_source_quoted_playbook,
)
from reflexio.server.services.storage.error import ReadinessUnavailableError

logger = logging.getLogger(__name__)


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


class PlaybookConsolidator(BaseDeduplicator):
    """Share indexed candidate retrieval and one audited consolidation method."""

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
        self.lineage_sources_by_output_index: dict[int, list[str]] = {}

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
        query_playbooks = []
        for playbook in new_playbooks:
            trigger = playbook.trigger or playbook.content
            if trigger and trigger.strip():
                query_texts.append(trigger.strip())
                query_playbooks.append(playbook)

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
                search_options = SearchOptions(
                    query_embedding=embeddings[i], exclude_source_quoted=True
                )
                results = storage.search_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
                    search_request, search_options
                )
                results = [row for row in results if not is_source_quoted_playbook(row)]
                self._remember_retrieval_matches(query_playbooks[i], results)
                for fb in results:
                    if fb.user_playbook_id and fb.user_playbook_id not in seen_ids:
                        seen_ids.add(fb.user_playbook_id)
                        existing_playbooks.append(fb)
            except ReadinessUnavailableError as exc:  # noqa: PERF203
                self._raise_overflow_retrieval_error(exc)
                # Embedding not ready yet (a new task's warm-up after a deploy):
                # expected and self-clearing, so not an ERROR. Recall degrades
                # exactly as for any other failed search below.
                logger.warning(
                    "event=playbook_consolidation_search_not_ready query_index=%d "
                    "error=%s",
                    i,
                    exc,
                )
            except Exception as exc:  # noqa: PERF203 — per-query isolation is the point
                self._raise_overflow_retrieval_error(exc)
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

    @staticmethod
    def _raise_overflow_retrieval_error(error: Exception) -> None:
        from reflexio.server.llm._litellm_types import ProviderRequestGuardError
        from reflexio.server.services.extraction.overflow_coordinator import (
            current_overflow_coordinator,
        )

        if current_overflow_coordinator.get() is not None:
            raise ProviderRequestGuardError(
                "Historical retrieval is undetermined"
            ) from error

    @staticmethod
    def _remember_retrieval_matches(
        candidate: UserPlaybook, results: list[UserPlaybook]
    ) -> None:
        from reflexio.server.services.extraction.overflow_coordinator import (
            current_overflow_coordinator,
        )

        coordinator = current_overflow_coordinator.get()
        if coordinator is not None:
            coordinator.candidate_historical_matches[
                coordinator.playbook_key(candidate)
            ] = tuple(row.user_playbook_id for row in results if row.user_playbook_id)

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
        """Draft and independently audit one Create/Replace/Remove write plan.

        Returns rows, superseded IDs, and lineage merge groups for the existing
        transactional publisher. Computation performs no storage writes. A
        refusal or exhausted allowance never falls back to unaudited decisions.
        """
        generation_request_id = _normalize_generation_request_id(
            generation_request_id, request_id=request_id
        )
        self.model_provenance = None
        self.consolidated_output_indices = set()
        self.lineage_sources_by_output_index = {}
        if agent_version is None:
            raise TypeError("agent_version is required")

        # Check if mock mode is enabled
        if env_bool("MOCK_LLM_RESPONSE", default=False):
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

        if not new_playbooks and not existing_playbooks:
            return [], [], []

        # Retrieve existing entries via hybrid search, unless an earlier stage in
        # this generation already did (the reviewer needs the same rows).
        if existing_playbooks is None:
            existing_playbooks = self.retrieve_existing_playbooks(
                new_playbooks, user_id=user_id, agent_version=agent_version
            )
        # Earlier review stages can supply their own retrieved rows. Preserve
        # copied instructions through every automatic consolidation strategy.
        existing_playbooks = [
            row for row in existing_playbooks if not is_source_quoted_playbook(row)
        ]

        from reflexio.server.services.playbook.components.overflow_evidence import (
            remember_historical_playbooks,
        )

        remember_historical_playbooks(existing_playbooks)
        from reflexio.server.services.playbook.components.reconciliation import (
            reconcile_playbooks,
        )

        return reconcile_playbooks(
            self,
            new_playbooks,
            existing_playbooks,
            user_id=user_id,
            agent_version=agent_version,
            generation_request_id=generation_request_id,
        )

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
