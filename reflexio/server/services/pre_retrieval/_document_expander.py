"""Independent document expansion module for storage-time enrichment.

Expands document content with synonyms and related terms to improve FTS recall.
Called during the storage pipeline, before document indexing.

The expanded terms are stored in a structured plain text format:
  "backup, sync, replica; deploy, release, ship"
Each semicolon-separated group contains a key term followed by its synonyms.
"""

import json
import logging
import re

from pydantic import BaseModel

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.service_utils import log_model_response

logger = logging.getLogger(__name__)


class ExpansionResult(BaseModel):
    """Output of document expansion.

    Args:
        expansions (dict[str, list[str]]): Mapping of key terms to synonym lists.
        expanded_text (str): Punctuated plain text of all expansions.
    """

    expansions: dict[str, list[str]] = {}
    expanded_text: str = ""


class DocumentExpander:
    """Independent, reusable document expansion module.

    Expands document content with synonyms and related terms via LLM
    for improved full-text search recall at storage time.
    """

    MAX_CONTENT_LENGTH = 4000
    LLM_TIMEOUT = 25
    LLM_MAX_RETRIES = 1

    def __init__(
        self,
        llm_client: LiteLLMClient,
        prompt_manager: PromptManager,
        model_name: str | None = None,
    ):
        """Initialize the DocumentExpander.

        Args:
            llm_client (LiteLLMClient): Shared LLM client instance
            prompt_manager (PromptManager): Prompt manager for rendering prompts
            model_name (str, optional): Model name override for expansion.
                When set, passed as model= kwarg to generate_response().
        """
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.model_name = model_name

    def expand(self, content: str) -> ExpansionResult:
        """Expand document content with synonyms and related terms.

        Uses LLM to extract key concepts and generate synonyms.
        Returns structured result with both the expansion mapping
        and a punctuated plain text representation.

        Args:
            content (str): Document text to expand

        Returns:
            ExpansionResult: Expansion mapping and formatted text
        """
        if len(content) > self.MAX_CONTENT_LENGTH:
            content = content[: self.MAX_CONTENT_LENGTH]
        try:
            expansions = self._extract_expansions(content)
            expanded_text = self._format_expanded_terms(expansions)
            return ExpansionResult(
                expansions=expansions,
                expanded_text=expanded_text,
            )
        except Exception as e:
            logger.warning("Document expansion failed, returning empty: %s", e)
            return ExpansionResult()

    def _extract_expansions(self, content: str) -> dict[str, list[str]]:
        """Use LLM to extract key terms and their synonyms from document content.

        Args:
            content (str): Document text

        Returns:
            dict[str, list[str]]: Mapping of key terms to synonym lists

        Raises:
            Exception: If LLM call or parsing fails
        """
        prompt = self.prompt_manager.render_prompt(
            "document_expansion",
            {"content": content},
        )
        logger.debug("Document expansion prompt: %s", prompt)
        model_kwargs = {}
        if self.model_name:
            model_kwargs["model"] = self.model_name
        result = self.llm_client.generate_response(
            prompt,
            timeout=self.LLM_TIMEOUT,
            max_retries=self.LLM_MAX_RETRIES,
            **model_kwargs,
        )
        log_model_response(logger, "Document expansion model response", result)

        if isinstance(result, str):
            return self._parse_expansion_json(result)
        return {}

    @staticmethod
    def _parse_expansion_json(output: str) -> dict[str, list[str]]:
        """Parse LLM output as JSON expansion mapping.

        Handles code-fenced JSON and bare JSON. Validates structure.

        Args:
            output (str): Raw LLM output

        Returns:
            dict[str, list[str]]: Parsed expansion mapping
        """
        text = output.strip()

        # Handle code-fenced JSON
        if "```" in text:
            match = re.search(r"```(?:\w+)?\s*([\s\S]*?)```", text)
            if match:
                text = match.group(1).strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                expansions: dict[str, list[str]] = {}
                for key, value in parsed.items():
                    if isinstance(key, str) and isinstance(value, list):
                        synonyms = [
                            s for s in value if isinstance(s, str) and s.strip()
                        ]
                        if synonyms:
                            expansions[key.strip()] = synonyms
                return expansions
        except json.JSONDecodeError:
            logger.warning("Failed to parse expansion JSON: %s", text[:200])

        return {}

    @staticmethod
    def _format_expanded_terms(expansions: dict[str, list[str]]) -> str:
        """Format expansions as punctuated plain text.

        Output format: "term, syn1, syn2; term2, syn3, syn4"
        - Semicolons separate synonym groups
        - Commas separate terms within a group
        - First term in each group is the original key term

        Args:
            expansions (dict[str, list[str]]): Term-to-synonyms mapping

        Returns:
            str: Formatted expansion text
        """
        groups = []
        for term, synonyms in expansions.items():
            parts = [term, *synonyms]
            groups.append(", ".join(parts))
        return "; ".join(groups)
