"""LangChain callback handler that captures conversations and publishes them to Reflexio."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import ChatGeneration, LLMResult
except ImportError as e:
    raise ImportError(
        "LangChain integration requires langchain-core. "
        "Install with: pip install reflexio-client[langchain]"
    ) from e

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.messages import BaseMessage

    from reflexio.client.client import ReflexioClient

from reflexio.models.api_schema.service_schemas import InteractionData, ToolUsed

logger = logging.getLogger(__name__)

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
}


class _RunBuffer:
    """Accumulates interactions and tool calls for a single chain run."""

    __slots__ = (
        "interactions",
        "pending_tools",
        "root_run_id",
        "seen_user_messages",
    )

    def __init__(self, root_run_id: UUID) -> None:
        self.root_run_id = root_run_id
        self.interactions: list[InteractionData] = []
        self.pending_tools: dict[UUID, dict[str, Any]] = {}
        self.seen_user_messages: set[tuple[str, str, int]] = set()


class ReflexioCallbackHandler(BaseCallbackHandler):  # noqa: ARG002
    """LangChain callback handler that captures conversations and publishes them to Reflexio.

    Accumulates messages during a chain/agent run and publishes the full conversation
    to Reflexio when the outermost chain completes (fire-and-forget).

    Args:
        client (ReflexioClient): Reflexio client instance
        user_id (str): User identifier for the interaction
        agent_version (str): Agent version string for tracking
        session_id (str | None): Optional session ID for grouping requests
        source (str): Source identifier for the interaction

    Example:
        >>> from reflexio import ReflexioClient
        >>> from reflexio.integrations.langchain import ReflexioCallbackHandler
        >>>
        >>> client = ReflexioClient(api_key="...", url_endpoint="http://localhost:8081/")
        >>> handler = ReflexioCallbackHandler(client, user_id="user_123", agent_version="v1")
        >>> chain.invoke({"input": "..."}, config={"callbacks": [handler]})
    """

    def __init__(
        self,
        client: ReflexioClient,
        user_id: str,
        agent_version: str = "",
        session_id: str | None = None,
        source: str = "",
    ) -> None:
        super().__init__()
        self.client = client
        self.user_id = user_id
        self.agent_version = agent_version
        self.session_id = session_id
        self.source = source
        self._buffers: dict[UUID, _RunBuffer] = {}
        self._run_roots: dict[UUID, UUID] = {}

    @staticmethod
    def _message_content(message: Any) -> str:
        """Normalize LangChain message content into a string."""
        return (
            message.content
            if isinstance(message.content, str)
            else str(message.content)
        )

    def _merge_roots(self, from_root: UUID, to_root: UUID) -> None:
        """Merge a provisional root buffer into the resolved root buffer."""
        if from_root == to_root:
            return

        source = self._buffers.pop(from_root, None)
        target = self._buffers.get(to_root)
        if target is None:
            if source is not None:
                source.root_run_id = to_root
                self._buffers[to_root] = source
        elif source is not None:
            target.interactions.extend(source.interactions)
            target.pending_tools.update(source.pending_tools)
            target.seen_user_messages.update(source.seen_user_messages)

        for tracked_run_id, tracked_root in list(self._run_roots.items()):
            if tracked_root == from_root:
                self._run_roots[tracked_run_id] = to_root

    def _get_or_create_buffer(
        self, run_id: UUID, parent_run_id: UUID | None
    ) -> _RunBuffer:
        """Get the buffer for the root run, creating one if this is the root."""
        root_run_id = self._run_roots.get(run_id)
        parent_root_run_id = (
            self._run_roots.get(parent_run_id) if parent_run_id is not None else None
        )

        if (
            root_run_id is not None
            and parent_root_run_id is not None
            and root_run_id != parent_root_run_id
        ):
            self._merge_roots(root_run_id, parent_root_run_id)
            root_run_id = parent_root_run_id

        if root_run_id is None:
            if parent_root_run_id is not None:
                root_run_id = parent_root_run_id
            elif parent_run_id is not None:
                root_run_id = parent_run_id
            else:
                root_run_id = run_id

        self._run_roots[run_id] = root_run_id
        if parent_run_id is not None:
            self._run_roots.setdefault(parent_run_id, root_run_id)

        buf = self._buffers.get(root_run_id)
        if buf is None:
            buf = _RunBuffer(root_run_id=root_run_id)
            self._buffers[root_run_id] = buf
        return buf

    def _pop_buffer(self, root_run_id: UUID) -> _RunBuffer | None:
        """Remove a completed root run buffer and all of its tracked child runs."""
        for tracked_run_id, tracked_root in list(self._run_roots.items()):
            if tracked_root == root_run_id:
                self._run_roots.pop(tracked_run_id, None)
        return self._buffers.pop(root_run_id, None)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture new user messages sent to the chat model."""
        buf = self._get_or_create_buffer(run_id, parent_run_id)
        prompt_occurrences: dict[tuple[str, str], int] = {}
        for message_list in messages:
            for msg in message_list:
                if (role := _ROLE_MAP.get(msg.type)) is None:
                    continue
                if role != "user":
                    continue
                content = self._message_content(msg)
                if not content:
                    continue

                occurrence_key = (role, content)
                prompt_occurrences[occurrence_key] = (
                    prompt_occurrences.get(occurrence_key, 0) + 1
                )
                message_key = (*occurrence_key, prompt_occurrences[occurrence_key])
                if message_key in buf.seen_user_messages:
                    continue

                buf.seen_user_messages.add(message_key)
                buf.interactions.append(
                    InteractionData(
                        role=role,
                        content=content,
                        created_at=int(time.time()),
                    )
                )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Capture LLM response."""
        buf = self._get_or_create_buffer(run_id, parent_run_id)
        for generation_list in response.generations:
            for generation in generation_list:
                content = generation.text
                if (
                    not content
                    and isinstance(generation, ChatGeneration)
                    and generation.message
                ):
                    content = self._message_content(generation.message)
                if content:
                    buf.interactions.append(
                        InteractionData(
                            role="assistant",
                            content=content,
                            created_at=int(time.time()),
                        )
                    )

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool invocation start."""
        buf = self._get_or_create_buffer(run_id, parent_run_id)
        tool_name = serialized.get("name", kwargs.get("name", "unknown_tool"))
        buf.pending_tools[run_id] = {
            "tool_name": tool_name,
            "tool_data": input_str,
        }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool invocation end and attach to an assistant interaction."""
        buf = self._get_or_create_buffer(run_id, parent_run_id)
        if (tool_info := buf.pending_tools.pop(run_id, None)) is None:
            return
        tool_used = ToolUsed(
            tool_name=tool_info["tool_name"],
            tool_data={"input": tool_info["tool_data"], "output": str(output)},
        )
        # Attach to the most recent assistant interaction, or create one
        for interaction in reversed(buf.interactions):
            if interaction.role == "assistant":
                interaction.tools_used.append(tool_used)
                return
        buf.interactions.append(
            InteractionData(
                role="assistant",
                content="",
                created_at=int(time.time()),
                tools_used=[tool_used],
            )
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Publish accumulated interactions when the root chain completes."""
        if parent_run_id is not None:
            return

        buf = self._pop_buffer(self._run_roots.get(run_id, run_id))
        if buf is None:
            return

        if not buf.interactions:
            logger.debug("No interactions captured, skipping publish")
            return

        try:
            self.client.publish_interaction(
                user_id=self.user_id,
                interactions=list(buf.interactions),  # type: ignore[arg-type]
                source=self.source,
                agent_version=self.agent_version,
                session_id=self.session_id,
                wait_for_response=False,
            )
        except Exception:
            logger.exception("Failed to publish interactions to Reflexio")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Clean up buffer on chain error. Still publish what was captured."""
        self.on_chain_end({}, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any
    ) -> None:
        """No-op for non-chat LLM calls."""

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """No-op."""

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        """No-op."""
