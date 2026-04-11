"""LangChain retriever that fetches relevant Reflexio documents."""

from __future__ import annotations

import logging
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install reflexio-client[langchain]"
    ) from e

logger = logging.getLogger(__name__)


class ReflexioRetriever(BaseRetriever):  # noqa: ARG002
    """LangChain retriever that searches Reflexio for relevant playbooks and profiles.

    Returns results as LangChain Document objects, enabling standard RAG patterns
    like create_retrieval_chain.

    Args:
        client: Reflexio client instance
        agent_version (str): Filter results by agent version
        user_id (str): Filter profile results by user ID
        search_type (str): Entity types to search — "unified" (all), "feedbacks", "raw_feedbacks", or "profiles"
        top_k (int): Maximum results per entity type

    Example:
        >>> from reflexio import ReflexioClient
        >>> from reflexio.integrations.langchain import ReflexioRetriever
        >>>
        >>> client = ReflexioClient(api_key="...", url_endpoint="http://localhost:8081/")
        >>> retriever = ReflexioRetriever(client=client, agent_version="v1")
        >>> docs = retriever.invoke("How should I handle password reset requests?")
    """

    client: Any  # ReflexioClient — typed as Any for Pydantic compatibility
    agent_version: str = ""
    user_id: str = ""
    search_type: str = "unified"
    top_k: int = 5

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> list[Document]:
        """Search Reflexio and return results as LangChain Documents.

        Args:
            query (str): Search query

        Returns:
            list[Document]: Matching documents with metadata indicating type
        """
        try:
            resp = self.client.search(
                query=query,
                agent_version=self.agent_version or None,
                user_id=self.user_id or None,
                top_k=self.top_k,
            )
        except Exception:
            logger.debug("Failed to search Reflexio", exc_info=True)
            return []

        docs: list[Document] = []

        if self.search_type in ("unified", "feedbacks") and resp.agent_playbooks:
            docs.extend(
                Document(
                    page_content=fb.content,
                    metadata={
                        "type": "agent_playbook",
                        "instruction": fb.structured_data.instruction or "",
                        "pitfall": fb.structured_data.pitfall or "",
                        "trigger": fb.structured_data.trigger or "",
                        "agent_version": fb.agent_version,
                    },
                )
                for fb in resp.agent_playbooks
                if fb.content
            )

        if self.search_type in ("unified", "raw_feedbacks") and resp.user_playbooks:
            docs.extend(
                Document(
                    page_content=rf.content,
                    metadata={
                        "type": "user_playbook",
                        "instruction": rf.structured_data.instruction or "",
                        "pitfall": rf.structured_data.pitfall or "",
                        "trigger": rf.structured_data.trigger or "",
                        "agent_version": rf.agent_version,
                    },
                )
                for rf in resp.user_playbooks
                if rf.content
            )

        if self.search_type in ("unified", "profiles") and resp.profiles:
            docs.extend(
                Document(
                    page_content=profile.content,
                    metadata={
                        "type": "profile",
                        "user_id": profile.user_id,
                    },
                )
                for profile in resp.profiles
                if profile.content
            )

        return docs
