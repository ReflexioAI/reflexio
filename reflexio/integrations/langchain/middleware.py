"""ChatModel wrapper that automatically enriches every LLM call with Reflexio context.

Wraps any LangChain ChatModel so that each invocation searches Reflexio for
relevant playbooks and user profiles, then injects them as a system
message before the user's query reaches the underlying model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from typing import Any

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage,
        BaseMessage,
        HumanMessage,
        SystemMessage,
    )
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install reflexio-client[langchain]"
    ) from e

from reflexio.integrations.langchain.prompt import get_reflexio_context

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_HEADER = (
    "The following context from Reflexio contains relevant "
    "behavioral guidelines and user profile information. "
    "Use this context to inform your response."
)


class ReflexioChatModel(BaseChatModel):
    """ChatModel wrapper that automatically searches Reflexio before every LLM call.

    Intercepts each call to the underlying model, extracts the user's latest
    message as a search query, fetches relevant context from Reflexio, and
    injects it as a SystemMessage before delegating to the wrapped model.

    Works as a drop-in replacement anywhere a ChatModel is accepted — LCEL
    chains, AgentExecutor, LangGraph, ``create_react_agent``, etc.

    Args:
        llm (BaseChatModel): The chat model to wrap
        client (ReflexioClient): Reflexio client instance
        agent_version (str): Filter results by agent version
        user_id (str): Filter profile results by user ID
        top_k (int): Maximum results per entity type from Reflexio search
        context_header (str): Header text prepended to the injected context message

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from reflexio import ReflexioClient
        >>> from reflexio.integrations.langchain import ReflexioChatModel
        >>>
        >>> llm = ChatOpenAI(model="gpt-4o")
        >>> client = ReflexioClient(url_endpoint="http://localhost:8081/")
        >>> reflexio_llm = ReflexioChatModel(
        ...     llm=llm,
        ...     client=client,
        ...     agent_version="v1",
        ...     user_id="user_alice",
        ... )
        >>> # Use as a drop-in replacement
        >>> response = reflexio_llm.invoke("How should I handle this request?")
    """

    llm: Any  # BaseChatModel — typed as Any for Pydantic compatibility
    client: Any  # ReflexioClient — typed as Any for Pydantic compatibility
    agent_version: str = ""
    user_id: str = ""
    top_k: int = 5
    context_header: str = _DEFAULT_CONTEXT_HEADER

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return f"reflexio-{self.llm._llm_type}"

    @staticmethod
    def _extract_query(messages: list[BaseMessage]) -> str:
        """Extract the latest user message content as a search query.

        Args:
            messages (list[BaseMessage]): The message list to search

        Returns:
            str: Content of the last HumanMessage, or empty string if none found
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                return msg.content
        return ""

    def _inject_context(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Search Reflexio and inject results as a SystemMessage before the last user message.

        Args:
            messages (list[BaseMessage]): Original message list

        Returns:
            list[BaseMessage]: Message list with Reflexio context injected, or
                the original list if no relevant context was found
        """
        if not (query := self._extract_query(messages)):
            return messages

        context = get_reflexio_context(
            self.client,
            query,
            agent_version=self.agent_version,
            user_id=self.user_id,
            top_k=self.top_k,
        )
        if not context:
            return messages

        context_msg = SystemMessage(content=f"{self.context_header}\n\n{context}")
        enriched = list(messages)

        # Insert before the last HumanMessage
        for i in range(len(enriched) - 1, -1, -1):
            if isinstance(enriched[i], HumanMessage):
                enriched.insert(i, context_msg)
                return enriched

        # Fallback: prepend if no HumanMessage found
        enriched.insert(0, context_msg)
        return enriched

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        enriched = self._inject_context(messages)
        if isinstance(self.llm, BaseChatModel):
            return self.llm._generate(enriched, stop, run_manager, **kwargs)
        # RunnableBinding (e.g. after bind_tools) — use invoke to preserve bound kwargs
        result = self.llm.invoke(enriched, stop=stop, **kwargs)
        if isinstance(result, ChatResult):
            return result
        return ChatResult(generations=[ChatGeneration(message=result)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        enriched = self._inject_context(messages)
        if isinstance(self.llm, BaseChatModel):
            return await self.llm._agenerate(enriched, stop, run_manager, **kwargs)
        # RunnableBinding (e.g. after bind_tools) — use ainvoke to preserve bound kwargs
        result = await self.llm.ainvoke(enriched, stop=stop, **kwargs)
        if isinstance(result, ChatResult):
            return result
        return ChatResult(generations=[ChatGeneration(message=result)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        enriched = self._inject_context(messages)
        if isinstance(self.llm, BaseChatModel):
            yield from self.llm._stream(enriched, stop, run_manager, **kwargs)
            return
        # RunnableBinding (e.g. after bind_tools) — use stream to preserve bound kwargs
        for chunk in self.llm.stream(enriched, stop=stop, **kwargs):
            if isinstance(chunk, ChatGenerationChunk):
                yield chunk
            else:
                yield ChatGenerationChunk(message=chunk)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Bind tools to the underlying model, preserving Reflexio context injection.

        Args:
            tools: Sequence of tools to bind to the model.
            tool_choice (str | None): The tool to use.

        Returns:
            Runnable: A new ReflexioChatModel wrapping the tool-bound inner model.
        """
        bound = self.llm.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        return self.model_copy(update={"llm": bound})

    def with_structured_output(
        self,
        schema: dict[str, Any] | type,
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable:
        """Return a runnable that produces structured output, preserving Reflexio context injection.

        Composes this model (which handles context injection) with the underlying
        model's structured output parser, rather than wrapping the structured
        runnable back into ReflexioChatModel. This avoids issues where the
        structured output runnable returns parsed objects instead of AIMessage.

        Args:
            schema: The output schema.
            include_raw (bool): Whether to include raw model output alongside parsed.

        Returns:
            Runnable: A chain of this ReflexioChatModel piped into the structured output parser.
        """
        # Get the structured output runnable from the underlying model.
        # This is typically model | parser, so we extract just the parser portion
        # by building it from the original (unwrapped) llm.
        if isinstance(self.llm, BaseChatModel):
            structured = self.llm.with_structured_output(
                schema, include_raw=include_raw, **kwargs
            )
            # structured is typically llm | parser — we want self | parser,
            # so rebuild with self as the base model
            if hasattr(structured, "first") and hasattr(structured, "last"):
                # RunnableSequence: replace the model step with self
                return self | structured.last
            # Fallback: just pipe self into the structured runnable's logic
            return structured
        # Non-BaseChatModel (e.g. RunnableBinding from bind_tools)
        return self.llm.with_structured_output(
            schema, include_raw=include_raw, **kwargs
        )
