"""Intent-specialist search agents that surface profile / playbook candidates.

Each agent drives a tool-calling loop for one retrieval intent ("direct",
"context", "temporal" for both profiles and playbooks). The LLM issues
``search_profiles`` / ``search_playbooks`` calls, may ``reformulate`` the
query, and ends the turn by calling ``submit_candidates`` with the chosen
IDs. Submissions are collected into the agent's ``SearchCtx`` and returned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.retriever_schema import (
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
)
from reflexio.server.llm.model_defaults import ModelRole
from reflexio.server.llm.tools import Tool, ToolRegistry, run_tool_loop

if TYPE_CHECKING:
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.prompt.prompt_manager import PromptManager
    from reflexio.server.services.storage.storage_base import BaseStorage


ProfileIntent = Literal["direct", "context", "temporal"]
PlaybookIntent = Literal["direct", "context", "temporal"]


# ---------------- tool argument schemas ---------------- #


class SearchProfilesArgs(BaseModel):
    """Search the profile store for candidates matching the query.

    Args:
        query (str): Text query to run against the profile store.
        top_k (int): Maximum number of candidates to return.
        respect_ttl (bool): When True, exclude archived / expired items.
    """

    query: str
    top_k: int = 10
    respect_ttl: bool = True


class SearchPlaybooksArgs(BaseModel):
    """Search the playbook store for candidates matching the query.

    Args:
        query (str): Text query to run against the playbook store.
        top_k (int): Maximum number of candidates to return.
        respect_ttl (bool): When True, exclude archived / expired items.
    """

    query: str
    top_k: int = 10
    respect_ttl: bool = True


class ReformulateArgs(BaseModel):
    """Replace the live query with a reformulated version.

    Args:
        new_query (str): Updated query to use on the next search call.
    """

    new_query: str


class SubmitCandidatesArgs(BaseModel):
    """Submit the final candidate IDs and a one-sentence justification.

    Args:
        ids (list[str]): IDs of the selected candidates.
        why (str): One-sentence justification for the selection.
    """

    ids: list[str]
    why: str


# ---------------- ctx + handlers ---------------- #


@dataclass
class SearchCtx:
    """Mutable accumulator passed to tool handlers during one search agent run.

    Attributes:
        query (str): Current live query (reformulations mutate this).
        req (object): Caller-supplied request object; ``user_id`` attribute is read.
        storage (BaseStorage): Storage backend used by search tool handlers.
        lane (Literal["profile", "playbook"]): Lane this ctx serves.
        hits (list): Raw hits returned by tool calls, in call order.
        ids (list[str]): IDs submitted via ``submit_candidates``.
        why (str): Justification submitted via ``submit_candidates``.
        finished (bool): True once ``submit_candidates`` has been called.
    """

    query: str
    req: object
    storage: Any
    lane: Literal["profile", "playbook"]
    hits: list = field(default_factory=list)
    ids: list[str] = field(default_factory=list)
    why: str = ""
    finished: bool = False


def _status_filter_for_ttl(respect_ttl: bool) -> list[Status | None] | None:
    """Translate the agent-facing ``respect_ttl`` flag into a storage filter.

    ``respect_ttl=True`` returns ``[None]`` — only CURRENT items. ``False``
    returns ``None`` — no status filter, so archived / superseded items are
    included (used by the TEMPORAL agents).
    """
    return [None] if respect_ttl else None


def _search_profiles(args: BaseModel, ctx: SearchCtx) -> dict:
    """Tool handler: search the profile store and extend ``ctx.hits``."""
    a = cast(SearchProfilesArgs, args)
    user_id = getattr(ctx.req, "user_id", None)
    if not user_id:
        return {"hit_count": 0, "ids": []}
    request = SearchUserProfileRequest(user_id=user_id, query=a.query, top_k=a.top_k)
    results = ctx.storage.search_user_profile(
        request, status_filter=_status_filter_for_ttl(a.respect_ttl)
    )
    ctx.hits.extend(results)
    return {
        "hit_count": len(results),
        "ids": [getattr(r, "profile_id", None) for r in results],
    }


def _search_playbooks(args: BaseModel, ctx: SearchCtx) -> dict:
    """Tool handler: search the playbook store and extend ``ctx.hits``."""
    a = cast(SearchPlaybooksArgs, args)
    user_id = getattr(ctx.req, "user_id", None)
    if not user_id:
        return {"hit_count": 0, "ids": []}
    request = SearchUserPlaybookRequest(
        query=a.query,
        user_id=user_id,
        top_k=a.top_k,
        status_filter=_status_filter_for_ttl(a.respect_ttl),
    )
    results = ctx.storage.search_user_playbooks(request)
    ctx.hits.extend(results)
    return {
        "hit_count": len(results),
        "ids": [getattr(r, "user_playbook_id", None) for r in results],
    }


def _reformulate(args: BaseModel, ctx: SearchCtx) -> dict:
    """Tool handler: replace ``ctx.query`` with the reformulated text."""
    a = cast(ReformulateArgs, args)
    ctx.query = a.new_query
    return {"query_updated": True}


def _submit(args: BaseModel, ctx: SearchCtx) -> dict:
    """Tool handler: record the final candidate selection and terminate."""
    a = cast(SubmitCandidatesArgs, args)
    ctx.ids = list(a.ids)
    ctx.why = a.why
    ctx.finished = True
    return {"submitted": True}


PROFILE_SEARCH_TOOLS = ToolRegistry(
    [
        Tool(
            name="search_profiles",
            args_model=SearchProfilesArgs,
            handler=_search_profiles,
        ),
        Tool(name="reformulate", args_model=ReformulateArgs, handler=_reformulate),
        Tool(
            name="submit_candidates", args_model=SubmitCandidatesArgs, handler=_submit
        ),
    ]
)

PLAYBOOK_SEARCH_TOOLS = ToolRegistry(
    [
        Tool(
            name="search_playbooks",
            args_model=SearchPlaybooksArgs,
            handler=_search_playbooks,
        ),
        Tool(name="reformulate", args_model=ReformulateArgs, handler=_reformulate),
        Tool(
            name="submit_candidates", args_model=SubmitCandidatesArgs, handler=_submit
        ),
    ]
)


# ---------------- agents ---------------- #


class ProfileSearchAgent:
    """Intent-specialist agent that picks profile candidates for a query.

    Args:
        intent (ProfileIntent): Which intent prompt to render ("direct",
            "context", "temporal").
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        storage (BaseStorage): Storage backend used by tool handlers.
        max_steps (int): Cap on tool-calling turns for one agent run.
    """

    def __init__(
        self,
        intent: ProfileIntent,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        storage: BaseStorage,
        max_steps: int = 6,
    ) -> None:
        self.intent = intent
        self.client = client
        self.prompt_manager = prompt_manager
        self.storage = storage
        self.max_steps = max_steps

    def run(self, *, query: str, req: object) -> SearchCtx:
        """Run the tool loop for one profile-search intent and return its ctx.

        Args:
            query (str): User-supplied query to rendered into the prompt.
            req (object): Request-like object; ``user_id`` attribute is read.

        Returns:
            SearchCtx: Ctx with ``ids``, ``why``, and raw ``hits`` populated.
        """
        ctx = SearchCtx(query=query, req=req, storage=self.storage, lane="profile")
        prompt = self.prompt_manager.render_prompt(
            f"profile_search_{self.intent}",
            variables={"query": query},
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PROFILE_SEARCH_TOOLS,
            model_role=ModelRole.ANGLE_READER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="submit_candidates",
        )
        return ctx


class PlaybookSearchAgent:
    """Intent-specialist agent that picks playbook candidates for a query.

    Args:
        intent (PlaybookIntent): Which intent prompt to render ("direct",
            "context", "temporal").
        client (LiteLLMClient): LLM client driving the tool loop.
        prompt_manager (PromptManager): Prompt store for the rendered system prompt.
        storage (BaseStorage): Storage backend used by tool handlers.
        max_steps (int): Cap on tool-calling turns for one agent run.
    """

    def __init__(
        self,
        intent: PlaybookIntent,
        *,
        client: LiteLLMClient,
        prompt_manager: PromptManager,
        storage: BaseStorage,
        max_steps: int = 6,
    ) -> None:
        self.intent = intent
        self.client = client
        self.prompt_manager = prompt_manager
        self.storage = storage
        self.max_steps = max_steps

    def run(self, *, query: str, req: object) -> SearchCtx:
        """Run the tool loop for one playbook-search intent and return its ctx.

        Args:
            query (str): User-supplied query to rendered into the prompt.
            req (object): Request-like object; ``user_id`` attribute is read.

        Returns:
            SearchCtx: Ctx with ``ids``, ``why``, and raw ``hits`` populated.
        """
        ctx = SearchCtx(query=query, req=req, storage=self.storage, lane="playbook")
        prompt = self.prompt_manager.render_prompt(
            f"playbook_search_{self.intent}",
            variables={"query": query},
        )
        run_tool_loop(
            client=self.client,
            messages=[{"role": "user", "content": prompt}],
            registry=PLAYBOOK_SEARCH_TOOLS,
            model_role=ModelRole.ANGLE_READER,
            max_steps=self.max_steps,
            ctx=ctx,
            finish_tool_name="submit_candidates",
        )
        return ctx
