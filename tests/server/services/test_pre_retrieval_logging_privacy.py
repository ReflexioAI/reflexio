"""Privacy regressions for pre-retrieval model-response logging."""

from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.retriever_schema import ReformulationResult
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.pre_retrieval._document_expander import DocumentExpander
from reflexio.server.services.pre_retrieval._query_reformulator import QueryReformulator


def _dependencies() -> tuple[MagicMock, MagicMock]:
    llm_client = MagicMock(spec=LiteLLMClient)
    prompt_manager = MagicMock(spec=PromptManager)
    prompt_manager.render_prompt.return_value = "test prompt"
    return llm_client, prompt_manager


def test_document_expansion_does_not_log_model_response_content() -> None:
    """Document expansion output must not be mirrored into llm_io.log."""
    llm_client, prompt_manager = _dependencies()
    llm_client.generate_response.return_value = (
        '{"customer-secret": ["sensitive synonym"]}'
    )
    expander = DocumentExpander(llm_client, prompt_manager)

    with patch(
        "reflexio.server.services.pre_retrieval._document_expander.logger.log"
    ) as log:
        result = expander.expand("some content")

    assert result.expansions == {"customer-secret": ["sensitive synonym"]}
    log.assert_not_called()


def test_non_structured_reformulation_is_not_logged() -> None:
    """Unvalidated model output must not be mirrored into llm_io.log."""
    llm_client, prompt_manager = _dependencies()
    llm_client.generate_chat_response.return_value = "customer-secret query text"
    reformulator = QueryReformulator(llm_client, prompt_manager)

    with patch(
        "reflexio.server.services.pre_retrieval._query_reformulator.log_model_response"
    ) as log_model_response:
        result = reformulator.rewrite("original query")

    assert result == ReformulationResult(standalone_query="original query")
    log_model_response.assert_not_called()
