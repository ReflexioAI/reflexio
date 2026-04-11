"""LangChain tool for searching Reflexio mid-conversation."""

from __future__ import annotations

import logging
from typing import Any

try:
    from langchain_core.tools import BaseTool
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install reflexio-client[langchain]"
    ) from e

from reflexio.integrations.langchain.prompt import get_reflexio_context

logger = logging.getLogger(__name__)


class ReflexioSearchTool(BaseTool):  # noqa: ARG002
    """LangChain tool that searches Reflexio for playbooks and user profiles.

    Allows agents to query Reflexio mid-conversation for relevant guidance
    based on past experience.

    Args:
        client: Reflexio client instance
        agent_version (str): Filter results by agent version
        user_id (str): Filter profile results by user ID

    Example:
        >>> from reflexio import ReflexioClient
        >>> from reflexio.integrations.langchain import ReflexioSearchTool
        >>>
        >>> client = ReflexioClient(api_key="...", url_endpoint="http://localhost:8081/")
        >>> tool = ReflexioSearchTool(client=client, agent_version="v1")
        >>> tools = [tool, ...other_tools]
        >>> agent = create_react_agent(llm, tools, prompt)
    """

    name: str = "reflexio_search"
    description: str = (
        "Search for behavioral playbooks and user profiles "
        "relevant to the current task. Use this to look up guidance on how to "
        "handle specific types of requests based on past experience and playbooks."
    )
    client: Any  # ReflexioClient — typed as Any for Pydantic compatibility
    agent_version: str = ""
    user_id: str = ""

    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str, **kwargs: Any) -> str:
        """Search Reflexio for relevant context.

        Args:
            query (str): Search query describing what guidance is needed

        Returns:
            str: Formatted search results with playbooks and profiles
        """
        result = get_reflexio_context(
            self.client,
            query,
            agent_version=self.agent_version,
            user_id=self.user_id,
        )
        return result or "No relevant playbooks or profiles found."
