"""Independent query reformulation module for pre-retrieval optimization.

Reformulates search queries into clean, normalized natural language AND
extracts the query's temporal signals (relative time window,
recency_dominant, wants_current) in the same structured LLM call — so
retrieval can be time-sensitive with no additional pre-search latency.
No FTS expansion -- that is handled by document-side expansion at storage time.

Provides two interfaces:
  - rewrite(): pure query transformation + temporal signals (no search)
  - search(): rewrite + execute search via callable + merge results
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Generic, TypeVar

from pydantic import BaseModel

from reflexio.models.api_schema.retriever_schema import (
    ConversationTurn,
    ReformulationResult,
)
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.service_utils import log_model_response

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ReformulationSearchResult(BaseModel, Generic[T]):  # noqa: UP046
    """Output of reformulation + search.

    Args:
        standalone_query (str): The reformulated query that was used for search.
        items (list[T]): Search results from the search function.
    """

    standalone_query: str
    items: list[T] = []


class QueryReformulator:
    """Independent, reusable query reformulation module.

    Does ONE thing: reformulate queries into clean, normalized natural language.
    Accepts any search function as a callable, making it usable across all search paths.
    """

    MAX_CONVERSATION_TURNS = 10
    MAX_CONVERSATION_CHARS = 4000
    MAX_REFORMULATION_LENGTH = 512
    LLM_TIMEOUT = 5
    LLM_MAX_RETRIES = 1
    _UNSAFE_PHRASES = (
        "here is",
        "output:",
        "json {",
        "```json",
        "i cannot",
        "i can't",
    )

    def __init__(
        self,
        llm_client: LiteLLMClient,
        prompt_manager: PromptManager,
        model_name: str | None = None,
    ):
        """Initialize the QueryReformulator.

        Args:
            llm_client (LiteLLMClient): Shared LLM client instance
            prompt_manager (PromptManager): Prompt manager for rendering prompts
            model_name (str, optional): Model name override for reformulation.
                When set, passed as model= kwarg to generate_response().
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.model_name = model_name

    def rewrite(
        self,
        query: str,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> ReformulationResult:
        """Reformulate a query and extract its temporal signals.

        One structured LLM call resolves conversation context (pronouns,
        ellipsis, implicit references), expands abbreviations, fixes grammar,
        normalizes terminology — and reads the question's time semantics
        (relative window, recency_dominant, wants_current).

        Falls back to the original query with no temporal signals on any
        LLM/parse failure.

        Args:
            query (str): The original user search query
            conversation_history (list, optional): Prior conversation turns for
                context-aware reformulation.

        Returns:
            ReformulationResult: Reformulated standalone query + temporal
                signals.
        """
        try:
            return self._reformulate(query, conversation_history)
        except Exception as e:
            logger.warning("Query reformulation failed, using original: %s", e)
            return ReformulationResult(standalone_query=query)

    def search(
        self,
        query: str,
        search_fn: Callable[[str], list[T]],
        conversation_history: list[ConversationTurn] | None = None,
        dedup_key: Callable[[T], str] | None = None,
    ) -> ReformulationSearchResult[T]:
        """Reformulate query, execute search, return results.

        Calls search_fn with the reformulated query. The search function receives
        a clean natural language query and returns results of any type.

        Args:
            query (str): The original user search query
            search_fn (Callable[[str], list[T]]): Search function that takes a
                query string and returns a list of results. The module calls this
                with the reformulated query.
            conversation_history (list, optional): Prior conversation turns
            dedup_key (Callable[[T], str], optional): Function to extract a unique
                key from each result for deduplication. If None, no dedup.

        Returns:
            ReformulationSearchResult[T]: Reformulated query + search results
        """
        result = self.rewrite(query, conversation_history)
        try:
            items = search_fn(result.standalone_query)
        except Exception:
            logger.warning(
                "Search function failed for reformulated query", exc_info=True
            )
            items = []
        if dedup_key:
            items = self._deduplicate(items, dedup_key)
        return ReformulationSearchResult(
            standalone_query=result.standalone_query,
            items=items,
        )

    def _reformulate(
        self,
        query: str,
        conversation_history: list[ConversationTurn] | None = None,
    ) -> ReformulationResult:
        """One structured LLM call: rewritten query + temporal signals.

        Args:
            query (str): The original search query
            conversation_history (list, optional): Prior conversation turns

        Returns:
            ReformulationResult: Validated result. A suspect rewritten-query
                field degrades the whole result to the original query with no
                signals (a model that mangles the rewrite is not trusted for
                the signals either).

        Raises:
            Exception: If the LLM call fails
        """
        conversation_context = self._format_conversation_context(conversation_history)
        conversation_context_block = (
            f"\nConversation context: {conversation_context}\n"
            if conversation_context
            else ""
        )
        prompt = self.prompt_manager.render_prompt(
            "query_reformulation",
            {
                "query": query,
                "conversation_context_block": conversation_context_block,
                "today_iso": datetime.now(UTC).date().isoformat(),
            },
        )
        logger.debug("Query reformulation prompt: %s", prompt)
        model_kwargs = {}
        if self.model_name:
            model_kwargs["model"] = self.model_name
        result = self.llm_client.generate_chat_response(
            messages=[{"role": "user", "content": prompt}],
            response_format=ReformulationResult,
            timeout=self.LLM_TIMEOUT,
            max_retries=self.LLM_MAX_RETRIES,
            **model_kwargs,
        )
        log_model_response(logger, "Query reformulation model response", result)

        if not isinstance(result, ReformulationResult):
            logger.warning("LLM returned non-structured reformulation: %r", result)
            return ReformulationResult(standalone_query=query)

        extracted = self._extract_reformulated_query(result.standalone_query)
        if not extracted:
            logger.warning(
                "LLM returned invalid reformulation: %s", result.standalone_query
            )
            return ReformulationResult(standalone_query=query)
        return result.model_copy(update={"standalone_query": extracted})

    @classmethod
    def _extract_reformulated_query(cls, output: str) -> str | None:
        """Extract the reformulated query from raw LLM output.

        Simpler than the old query rewriter -- expects clean natural language,
        not FTS syntax. Just strips whitespace, prefixes, and validates.

        Args:
            output (str): Raw LLM response text

        Returns:
            Optional[str]: Extracted query, or None if invalid
        """
        if not output or not output.strip():
            return None

        candidate = output.strip()

        # If multi-line, take first non-empty line
        if "\n" in candidate:
            lines = [line.strip() for line in candidate.splitlines() if line.strip()]
            if not lines:
                return None
            candidate = lines[0]

        candidate = candidate.strip("`\"' ")
        candidate = " ".join(candidate.split())

        if not candidate:
            return None
        if len(candidate) > cls.MAX_REFORMULATION_LENGTH:
            return None
        if not any(char.isalnum() for char in candidate):
            return None

        lower = candidate.lower()
        if any(phrase in lower for phrase in cls._UNSAFE_PHRASES):
            return None

        if any(ch in candidate for ch in ("{", "}", "[", "]")):
            return None

        return candidate

    @staticmethod
    def _format_conversation_context(
        conversation_history: list[ConversationTurn] | None = None,
    ) -> str:
        """Format conversation history into a string for the prompt.

        Args:
            conversation_history (list, optional): List of conversation turns
                with role and content attributes.

        Returns:
            str: Formatted conversation context, or empty string when empty/None
        """
        if not conversation_history:
            return ""

        truncated = conversation_history[-QueryReformulator.MAX_CONVERSATION_TURNS :]

        lines = []
        total_chars = 0
        for turn in truncated:
            if isinstance(turn, dict):
                role = turn.get("role", "unknown")
                content = turn.get("content", "")
            else:
                role = turn.role
                content = turn.content
            line = f"[{role}]: {content}"
            if total_chars + len(line) > QueryReformulator.MAX_CONVERSATION_CHARS:
                break
            lines.append(line)
            total_chars += len(line)
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(
        items: list[T],
        dedup_key: Callable[[T], str],
    ) -> list[T]:
        """Remove duplicates preserving order of first occurrence.

        Args:
            items (list[T]): Items to deduplicate
            dedup_key (Callable[[T], str]): Function to extract unique key

        Returns:
            list[T]: Deduplicated items
        """
        seen: set[str] = set()
        result: list[T] = []
        for item in items:
            key = dedup_key(item)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
