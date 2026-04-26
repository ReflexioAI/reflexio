"""Plan types for the agentic-v2 search pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from reflexio.server.llm.tools import ToolLoopTrace


class SearchResult(BaseModel):
    """Outcome of one SearchAgent run.

    Args:
        answer (str | None): The LLM-synthesised answer from finish(answer); None
            when the agent ran in search-only mode (``enable_agent_answer=False``)
            and deliberately did not synthesize a free-text answer.
        outcome (str): How the loop terminated.
        budget_exceeded (bool): True when outcome == "max_steps".
        trace (ToolLoopTrace): Full tool-loop trace — ids harvested by callers for entity fetch.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    answer: str | None
    outcome: Literal["finish_tool", "max_steps", "error"]
    budget_exceeded: bool
    trace: ToolLoopTrace
