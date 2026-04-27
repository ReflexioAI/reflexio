"""Atomic tool handlers for the agentic-v2 extraction + search pipelines.

Each handler:
  - Receives args (Pydantic model validated by ToolRegistry)
  - Receives (storage, ctx)
  - Calls an existing BaseStorage method
  - Returns a dict projection suitable for the LLM

Read handlers populate ctx.known_ids (for invariant B) and ctx.search_count
(for invariant A). Mutating handlers (Task 5) append PlanOps to ctx.plan
without hitting storage; commit_plan applies them via apply_plan_op after
invariants pass.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.entities import (
    Status,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive
from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    SearchMode,
    SearchUserPlaybookRequest,
    SearchUserProfileRequest,
)
from reflexio.models.config_schema import SearchOptions
from reflexio.server.services.extraction.plan import (
    CreateUserPlaybookOp,
    CreateUserProfileOp,
    DeleteUserPlaybookOp,
    DeleteUserProfileOp,
    ExtractionCtx,
    PlaybookStrength,
    ProfileTTL,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    calculate_expiration_timestamp,
)

TOP_K_CAP = 25


# ====================================================================
# Arg schemas (what the LLM emits)
# ====================================================================


class SearchUserProfilesArgs(BaseModel):
    """Semantic/keyword search the current user's profiles."""

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10


class GetUserProfileArgs(BaseModel):
    """Retrieve a single UserProfile by id."""

    id: Annotated[str, Field(min_length=1)]


class SearchUserPlaybooksArgs(BaseModel):
    """Search the current user's playbooks."""

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"


class GetUserPlaybookArgs(BaseModel):
    """Retrieve a single UserPlaybook by id."""

    id: Annotated[str, Field(min_length=1)]


class SearchAgentPlaybooksArgs(BaseModel):
    """Search agent-version-scoped playbooks (read-only; search pipeline only)."""

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"


class GetAgentPlaybookArgs(BaseModel):
    """Retrieve a single AgentPlaybook by id."""

    id: Annotated[str, Field(min_length=1)]


class GetSessionExcerptArgs(BaseModel):
    """Retrieve a verbatim excerpt from a session by matching a span."""

    session_id: Annotated[str, Field(min_length=1)]
    span: Annotated[str, Field(min_length=1)]


class RerankUserProfilesArgs(BaseModel):
    """Rerank a list of profile ids by query relevance using a cross-encoder.

    Use after `search_user_profiles` when the initial results are noisy and
    you need to surface the most semantically relevant ones to the question.
    """

    query: Annotated[str, Field(min_length=1)]
    profile_ids: list[str]
    top_k: int = 10


class StorageStatsArgs(BaseModel):
    """Get a quick count of how many profiles/playbooks the user has and the date range.

    Useful for sizing search top_k appropriately before retrieval.
    """


# Mutating arg models (handlers in Task 5)
class CreateUserProfileArgs(BaseModel):
    """Propose creating a new UserProfile record."""

    content: Annotated[str, Field(min_length=1)]
    ttl: ProfileTTL
    source_span: Annotated[str, Field(min_length=1)]


class DeleteUserProfileArgs(BaseModel):
    """Propose deleting an existing UserProfile by id."""

    id: Annotated[str, Field(min_length=1)]


class CreateUserPlaybookArgs(BaseModel):
    """Propose creating a new UserPlaybook record."""

    trigger: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]
    rationale: str = ""
    strength: PlaybookStrength = "soft"
    source_span: Annotated[str, Field(min_length=1)]


class DeleteUserPlaybookArgs(BaseModel):
    """Propose deleting an existing UserPlaybook by id."""

    id: Annotated[str, Field(min_length=1)]


class FinishArgs(BaseModel):
    """Terminate the loop."""


class SearchFinishArgs(BaseModel):
    """Terminate the search loop, optionally with a final answer.

    ``answer`` is opt-in: when the host runs the agent in search-only mode
    (``enable_agent_answer=False``) the agent is instructed to call ``finish()``
    without an answer; the host synthesizes the final response itself from the
    entities the agent harvested.
    """

    answer: str | None = None


# ====================================================================
# Helpers
# ====================================================================


def _cap_top_k(k: int) -> int:
    return min(max(1, k), TOP_K_CAP)


def _maybe_embed_query(storage: Any, query: str) -> list[float] | None:
    """Compute a query embedding via the storage backend's embedder.

    Returns ``None`` on any failure (backend doesn't expose ``_get_embedding``,
    embedding provider unavailable, or embed call raises). Without an embedding,
    storage downgrades HYBRID/VECTOR search to FTS-only — the classic search
    path (``unified_search_service.py:151-158``) uses the same helper pattern.

    Args:
        storage (Any): BaseStorage instance.
        query (str): The search query to embed.

    Returns:
        list[float] | None: The embedding vector, or ``None`` when unavailable.
    """
    embed_fn = getattr(storage, "_get_embedding", None)
    if embed_fn is None:
        return None
    try:
        return embed_fn(query)
    except Exception:  # noqa: BLE001 — embedder failures must not break search
        return None


def _status_from_str(s: str) -> Status | None:
    return {"current": None, "pending": Status.PENDING, "archived": Status.ARCHIVED}[s]


def _project_profile_for_llm(p: Any) -> dict[str, Any]:
    return {
        "id": getattr(p, "profile_id", "") or "",
        "content": p.content,
        "ttl": p.profile_time_to_live,
        "last_modified": p.last_modified_timestamp,
        "source_span": getattr(p, "source_span", None),
    }


def _project_user_playbook_for_llm(pb: Any) -> dict[str, Any]:
    return {
        "id": str(pb.user_playbook_id),
        "trigger": pb.trigger,
        "content": pb.content,
        "rationale": pb.rationale,
        "last_modified": getattr(pb, "created_at", 0),
    }


def _project_agent_playbook_for_llm(pb: Any) -> dict[str, Any]:
    return {
        "id": str(pb.agent_playbook_id),
        "trigger": pb.trigger,
        "content": pb.content,
        "rationale": pb.rationale,
        "playbook_status": getattr(pb, "playbook_status", None),
        "last_modified": getattr(pb, "created_at", 0),
    }


# ====================================================================
# Read handlers
# ====================================================================


def _handle_search_user_profiles(
    args: SearchUserProfilesArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Search the current user's profiles and bump search_count.

    Args:
        args (SearchUserProfilesArgs): Query and top_k.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count incremented in place.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing profile projections.
    """
    request = SearchUserProfileRequest(
        query=args.query,
        user_id=ctx.user_id,
        top_k=_cap_top_k(args.top_k),
    )
    hits = storage.search_user_profile(
        request,
        query_embedding=_maybe_embed_query(storage, args.query),
    )
    ctx.search_count += 1
    for h in hits:
        pid = getattr(h, "profile_id", "") or ""
        if pid:
            ctx.known_ids.add(pid)
    return {"hits": [_project_profile_for_llm(h) for h in hits]}


def _handle_get_user_profile(
    args: GetUserProfileArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Retrieve a single UserProfile by id without bumping search_count.

    Args:
        args (GetUserProfileArgs): Profile id to look up.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; known_ids updated on hit.

    Returns:
        dict[str, Any]: ``{"profile": {...}}`` on hit, ``{"error": "not found"}`` on miss.
    """
    all_profiles = storage.get_user_profile(ctx.user_id)
    for p in all_profiles:
        if (getattr(p, "profile_id", "") or "") == args.id:
            ctx.known_ids.add(args.id)
            return {"profile": _project_profile_for_llm(p)}
    return {"error": "not found"}


def _handle_search_user_playbooks(
    args: SearchUserPlaybooksArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Search the current user's playbooks and bump search_count.

    Args:
        args (SearchUserPlaybooksArgs): Query, top_k, and status filter.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count and known_ids updated.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing playbook projections.
    """
    request = SearchUserPlaybookRequest(
        query=args.query,
        user_id=ctx.user_id,
        agent_version=ctx.agent_version,
        top_k=_cap_top_k(args.top_k),
        status_filter=[_status_from_str(args.status)],
        search_mode=SearchMode.HYBRID,
        threshold=0.4,
    )
    if ctx.extractor_name:
        request.playbook_name = ctx.extractor_name
    hits = storage.search_user_playbooks(
        request,
        options=SearchOptions(query_embedding=_maybe_embed_query(storage, args.query)),
    )
    ctx.search_count += 1
    for h in hits:
        ctx.known_ids.add(str(h.user_playbook_id))
    return {"hits": [_project_user_playbook_for_llm(h) for h in hits]}


def _handle_get_user_playbook(
    args: GetUserPlaybookArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Retrieve a single UserPlaybook by id without bumping search_count.

    Args:
        args (GetUserPlaybookArgs): Playbook id to look up.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; known_ids updated on hit.

    Returns:
        dict[str, Any]: ``{"playbook": {...}}`` on hit, ``{"error": "not found"}`` on miss.
    """
    candidates = storage.get_user_playbooks(
        user_id=ctx.user_id, agent_version=ctx.agent_version
    )
    for pb in candidates:
        if str(pb.user_playbook_id) == args.id:
            ctx.known_ids.add(args.id)
            return {"playbook": _project_user_playbook_for_llm(pb)}
    return {"error": "not found"}


def _handle_search_agent_playbooks(
    args: SearchAgentPlaybooksArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Search agent-version-scoped playbooks and bump search_count.

    Args:
        args (SearchAgentPlaybooksArgs): Query, top_k, and status filter.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count and known_ids updated.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing agent playbook projections.
    """
    request = SearchAgentPlaybookRequest(
        query=args.query,
        agent_version=ctx.agent_version,
        top_k=_cap_top_k(args.top_k),
        status_filter=[_status_from_str(args.status)],
        search_mode=SearchMode.HYBRID,
        threshold=0.4,
    )
    if ctx.extractor_name:
        request.playbook_name = ctx.extractor_name
    hits = storage.search_agent_playbooks(
        request,
        options=SearchOptions(query_embedding=_maybe_embed_query(storage, args.query)),
    )
    ctx.search_count += 1
    for h in hits:
        ctx.known_ids.add(str(h.agent_playbook_id))
    return {"hits": [_project_agent_playbook_for_llm(h) for h in hits]}


def _handle_get_agent_playbook(
    args: GetAgentPlaybookArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Retrieve a single AgentPlaybook by id without bumping search_count.

    Args:
        args (GetAgentPlaybookArgs): Agent playbook id to look up.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; known_ids updated on hit.

    Returns:
        dict[str, Any]: ``{"playbook": {...}}`` on hit, ``{"error": "not found"}`` on miss.
    """
    candidates = storage.get_agent_playbooks(agent_version=ctx.agent_version)
    for pb in candidates:
        if str(pb.agent_playbook_id) == args.id:
            ctx.known_ids.add(args.id)
            return {"playbook": _project_agent_playbook_for_llm(pb)}
    return {"error": "not found"}


def _handle_get_session_excerpt(
    args: GetSessionExcerptArgs,
    storage: Any,
    ctx: ExtractionCtx,  # noqa: ARG001
) -> dict[str, Any]:
    """Return the closest verbatim match of ``span`` inside ``session_id``.

    Args:
        args (GetSessionExcerptArgs): Session id and span string to match.
        storage (Any): BaseStorage instance; must have ``get_interactions_by_session``.
        ctx (ExtractionCtx): Per-run state (unused for reads, present for consistency).

    Returns:
        dict[str, Any]: ``{"excerpt": str}`` on hit, ``{"error": str}`` on miss or
            when the storage backend doesn't support this method.
    """
    try:
        interactions = storage.get_interactions_by_session(args.session_id)
    except AttributeError:
        return {"error": "get_session_excerpt requires get_interactions_by_session"}
    matches = [
        i.content for i in interactions if args.span.strip() in (i.content or "")
    ]
    if not matches:
        return {"error": "span not found"}
    return {"excerpt": matches[0]}


def _handle_rerank_user_profiles(
    args: RerankUserProfilesArgs, storage: Any, ctx: ExtractionCtx
) -> dict[str, Any]:
    """Rerank known profile ids with a local cross-encoder.

    Fetches the candidate profiles (scoped to ``ctx.user_id``), scores
    ``(query, content)`` pairs, and returns the top_k by descending score.
    Bumps ``search_count`` so reranking still counts against the search
    budget enforced by invariant A.

    Args:
        args (RerankUserProfilesArgs): Query, candidate ids, and top_k.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; ``search_count`` and
            ``known_ids`` updated in place.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing profile
            projections sorted by descending relevance.
    """
    if not args.profile_ids:
        ctx.search_count += 1
        return {"hits": []}
    all_profiles = storage.get_user_profile(ctx.user_id)
    wanted = set(args.profile_ids)
    candidates = [
        p for p in all_profiles if (getattr(p, "profile_id", "") or "") in wanted
    ]
    ctx.search_count += 1
    if not candidates:
        return {"hits": []}
    # Lazy import — keeps unit-test collection fast and avoids loading
    # torch when no rerank tool call is made in a given run.
    from reflexio.server.llm.rerank import score_pairs

    scores = score_pairs(args.query, [p.content for p in candidates])
    ranked = sorted(
        zip(candidates, scores, strict=True),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top = [profile for profile, _score in ranked[: _cap_top_k(args.top_k)]]
    for h in top:
        pid = getattr(h, "profile_id", "") or ""
        if pid:
            ctx.known_ids.add(pid)
    return {"hits": [_project_profile_for_llm(h) for h in top]}


def _handle_storage_stats(
    args: StorageStatsArgs,  # noqa: ARG001
    storage: Any,
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Return profile/playbook counts and modified-time range for ``ctx.user_id``.

    Does not bump ``search_count`` — this is metadata, not retrieval.

    Args:
        args (StorageStatsArgs): No fields (sentinel call).
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; only ``user_id`` is read.

    Returns:
        dict[str, Any]: Counts and ISO 8601 timestamps. Timestamps are
            ``None`` when the user has no profiles.
    """
    profiles = storage.get_user_profile(ctx.user_id)
    if profiles:
        timestamps = [p.last_modified_timestamp for p in profiles]
        oldest_ts = datetime.fromtimestamp(min(timestamps), tz=UTC).isoformat()
        newest_ts = datetime.fromtimestamp(max(timestamps), tz=UTC).isoformat()
    else:
        oldest_ts = None
        newest_ts = None
    playbook_count = storage.count_user_playbooks(user_id=ctx.user_id)
    return {
        "profile_count": len(profiles),
        "playbook_count": playbook_count,
        "oldest_profile_modified": oldest_ts,
        "newest_profile_modified": newest_ts,
    }


def _next_tentative_id(ctx: ExtractionCtx, kind: str) -> str:
    """Generate a deterministic tentative-id scoped to this run.

    Format: ``tentative::<kind>::<plan_length>`` — unique within the run,
    recognizable in logs.

    Args:
        ctx (ExtractionCtx): Per-run state; plan length used as counter.
        kind (str): Entity type label, e.g. ``"profile"`` or ``"playbook"``.

    Returns:
        str: Tentative id string unique within this run.
    """
    return f"tentative::{kind}::{len(ctx.plan)}"


def new_profile_id() -> str:
    """Generate a short (12-char hex) profile id.

    Format chosen for LLM tool-call reliability: full ``str(uuid.uuid4())``
    is 36 characters of hex+dashes, error-prone for smaller LLMs to copy
    verbatim from a search result back into a delete/update tool arg.
    Twelve hex chars is short enough for high-fidelity copy and long enough
    that birthday-paradox collision probability is vanishingly small at any
    realistic per-user scale (16^12 ≈ 2.8e14 unique values; PRIMARY KEY
    constraint catches the rare collision).

    Profile ids are LLM-facing because the agent receives them in
    ``search_user_profiles`` results and must echo them back when calling
    ``delete_user_profile`` / ``update_user_profile``. Playbook ids are
    INTEGER autoincrements and don't have this problem.

    Returns:
        str: 12 lowercase hex characters, e.g. ``"b8a3f74e2c91"``.
    """
    return uuid.uuid4().hex[:12]


# ====================================================================
# Mutating handlers — append to ctx.plan, no storage writes
# ====================================================================


def _handle_create_user_profile(
    args: CreateUserProfileArgs,
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Propose creating a new UserProfile; appends CreateUserProfileOp to ctx.plan.

    No storage write occurs here — apply_plan_op commits ops after invariants pass.

    Args:
        args (CreateUserProfileArgs): Validated args from the LLM tool call.
        storage (Any): BaseStorage instance (unused; present for handler signature consistency).
        ctx (ExtractionCtx): Per-run state; plan and known_ids are mutated.

    Returns:
        dict[str, Any]: ``{"op_idx": int, "tentative_id": str}`` for LLM feedback.
    """
    tid = _next_tentative_id(ctx, "profile")
    op = CreateUserProfileOp(
        content=args.content, ttl=args.ttl, source_span=args.source_span
    )
    ctx.plan.append(op)
    ctx.known_ids.add(tid)
    return {"op_idx": len(ctx.plan) - 1, "tentative_id": tid}


def _handle_delete_user_profile(
    args: DeleteUserProfileArgs,
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Propose deleting an existing UserProfile; appends DeleteUserProfileOp to ctx.plan.

    No storage write occurs here.

    Args:
        args (DeleteUserProfileArgs): Validated args from the LLM tool call.
        storage (Any): BaseStorage instance (unused).
        ctx (ExtractionCtx): Per-run state; plan is mutated.

    Returns:
        dict[str, Any]: ``{"op_idx": int}`` for LLM feedback.
    """
    op = DeleteUserProfileOp(id=args.id)
    ctx.plan.append(op)
    return {"op_idx": len(ctx.plan) - 1}


def _handle_create_user_playbook(
    args: CreateUserPlaybookArgs,
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Propose creating a new UserPlaybook; appends CreateUserPlaybookOp to ctx.plan.

    No storage write occurs here.

    Args:
        args (CreateUserPlaybookArgs): Validated args from the LLM tool call.
        storage (Any): BaseStorage instance (unused).
        ctx (ExtractionCtx): Per-run state; plan and known_ids are mutated.

    Returns:
        dict[str, Any]: ``{"op_idx": int, "tentative_id": str}`` for LLM feedback.
    """
    tid = _next_tentative_id(ctx, "playbook")
    op = CreateUserPlaybookOp(
        trigger=args.trigger,
        content=args.content,
        rationale=args.rationale,
        strength=args.strength,
        source_span=args.source_span,
    )
    ctx.plan.append(op)
    ctx.known_ids.add(tid)
    return {"op_idx": len(ctx.plan) - 1, "tentative_id": tid}


def _handle_delete_user_playbook(
    args: DeleteUserPlaybookArgs,
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Propose deleting an existing UserPlaybook; appends DeleteUserPlaybookOp to ctx.plan.

    No storage write occurs here.

    Args:
        args (DeleteUserPlaybookArgs): Validated args from the LLM tool call.
        storage (Any): BaseStorage instance (unused).
        ctx (ExtractionCtx): Per-run state; plan is mutated.

    Returns:
        dict[str, Any]: ``{"op_idx": int}`` for LLM feedback.
    """
    op = DeleteUserPlaybookOp(id=args.id)
    ctx.plan.append(op)
    return {"op_idx": len(ctx.plan) - 1}


def _handle_finish(
    args: FinishArgs,  # noqa: ARG001
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Terminate the agent loop.

    Args:
        args (FinishArgs): No fields (sentinel call).
        storage (Any): BaseStorage instance (unused).
        ctx (ExtractionCtx): Per-run state; ``finished`` is set to True.

    Returns:
        dict[str, Any]: ``{"finished": True}``.
    """
    ctx.finished = True
    return {"finished": True}


def _handle_search_finish(
    args: SearchFinishArgs,
    storage: Any,  # noqa: ARG001
    ctx: ExtractionCtx,
) -> dict[str, Any]:
    """Terminate the search loop and stash the optional answer on ctx.

    Args:
        args (SearchFinishArgs): Contains the optional final answer string. When
            None (search-only mode) only the termination signal is emitted.
        storage (Any): BaseStorage instance (unused).
        ctx (ExtractionCtx): Per-run state; ``finished`` set True and
            ``search_answer`` populated for retrieval by SearchAgent.

    Returns:
        dict[str, Any]: ``{"finished": True, "answer": str | None}``.
    """
    ctx.finished = True
    ctx.search_answer = args.answer
    return {"finished": True, "answer": args.answer}


# ====================================================================
# Commit-stage: apply a PlanOp to storage
# ====================================================================


def apply_plan_op(op: Any, storage: Any, ctx: ExtractionCtx) -> None:
    """Deterministically apply one PlanOp to storage. Called by commit_plan.

    Args:
        op (Any): A PlanOp variant (CreateUserProfileOp, DeleteUserProfileOp,
            CreateUserPlaybookOp, DeleteUserPlaybookOp).
        storage (Any): BaseStorage handle.
        ctx (ExtractionCtx): Per-run state providing user_id, agent_version,
            extractor_name.

    Raises:
        TypeError: If ``op`` is not a recognised PlanOp type.
    """
    if isinstance(op, CreateUserProfileOp):
        now_ts = int(datetime.now(UTC).timestamp())
        ttl = ProfileTimeToLive(op.ttl)
        storage.add_user_profile(
            ctx.user_id,
            [
                UserProfile(
                    user_id=ctx.user_id,
                    profile_id=new_profile_id(),
                    content=op.content,
                    profile_time_to_live=ttl,
                    last_modified_timestamp=now_ts,
                    expiration_timestamp=calculate_expiration_timestamp(now_ts, ttl),
                    source=f"agentic_v2/{ctx.extractor_name or 'default'}",
                    source_span=op.source_span,
                    generated_from_request_id=ctx.request_id,
                )
            ],
        )
    elif isinstance(op, DeleteUserProfileOp):
        storage.delete_profiles_by_ids([op.id])
    elif isinstance(op, CreateUserPlaybookOp):
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    user_playbook_id=0,  # storage assigns
                    user_id=ctx.user_id,
                    agent_version=ctx.agent_version,
                    request_id=ctx.request_id,
                    playbook_name=ctx.extractor_name or "default",
                    content=op.content,
                    trigger=op.trigger,
                    rationale=op.rationale,
                    source_span=op.source_span,
                )
            ]
        )
    elif isinstance(op, DeleteUserPlaybookOp):
        try:
            playbook_id = int(op.id)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"DeleteUserPlaybookOp.id must be a numeric string, got {op.id!r}"
            ) from e
        storage.delete_user_playbooks_by_ids([playbook_id])
    else:
        raise TypeError(f"Unknown PlanOp: {type(op).__name__}")


# ====================================================================
# Bundle adapter + Tool registries
# ====================================================================

from collections.abc import Callable  # noqa: E402

from reflexio.server.llm.tools import Tool, ToolRegistry  # noqa: E402


def _bundle_handler(
    inner: Callable[[Any, Any, Any], dict[str, Any]],
) -> Callable[[Any, Any], dict[str, Any]]:
    """Adapt a (args, storage, ctx)-style handler to (args, bundle) for run_tool_loop.

    ExtractionAgent and SearchAgent build a HandlerBundle with .storage and
    .ctx attributes; this adapter unpacks them so the registry accepts our
    3-arg handlers.

    Args:
        inner (Callable[[Any, Any, Any], dict[str, Any]]): A handler callable
            with signature ``(args, storage, ctx) -> dict``.

    Returns:
        Callable[[Any, Any], dict[str, Any]]: A 2-arg callable
            ``(args, bundle) -> dict`` compatible with ``Tool.handler``.
    """

    def wrapped(args: Any, bundle: Any) -> dict[str, Any]:
        return inner(args, bundle.storage, bundle.ctx)

    return wrapped


_READ_TOOLS = [
    Tool(
        name="search_user_profiles",
        args_model=SearchUserProfilesArgs,
        handler=_bundle_handler(_handle_search_user_profiles),
    ),
    Tool(
        name="get_user_profile",
        args_model=GetUserProfileArgs,
        handler=_bundle_handler(_handle_get_user_profile),
    ),
    Tool(
        name="search_user_playbooks",
        args_model=SearchUserPlaybooksArgs,
        handler=_bundle_handler(_handle_search_user_playbooks),
    ),
    Tool(
        name="get_user_playbook",
        args_model=GetUserPlaybookArgs,
        handler=_bundle_handler(_handle_get_user_playbook),
    ),
    Tool(
        name="search_agent_playbooks",
        args_model=SearchAgentPlaybooksArgs,
        handler=_bundle_handler(_handle_search_agent_playbooks),
    ),
    Tool(
        name="get_agent_playbook",
        args_model=GetAgentPlaybookArgs,
        handler=_bundle_handler(_handle_get_agent_playbook),
    ),
    Tool(
        name="get_session_excerpt",
        args_model=GetSessionExcerptArgs,
        handler=_bundle_handler(_handle_get_session_excerpt),
    ),
]

_FINISH_TOOL = Tool(
    name="finish",
    args_model=FinishArgs,
    handler=_bundle_handler(_handle_finish),
)

PROFILE_EXTRACTION_TOOLS = ToolRegistry(
    [
        *_READ_TOOLS,
        Tool(
            name="create_user_profile",
            args_model=CreateUserProfileArgs,
            handler=_bundle_handler(_handle_create_user_profile),
        ),
        Tool(
            name="delete_user_profile",
            args_model=DeleteUserProfileArgs,
            handler=_bundle_handler(_handle_delete_user_profile),
        ),
        _FINISH_TOOL,
    ]
)

PLAYBOOK_EXTRACTION_TOOLS = ToolRegistry(
    [
        *_READ_TOOLS,
        Tool(
            name="create_user_playbook",
            args_model=CreateUserPlaybookArgs,
            handler=_bundle_handler(_handle_create_user_playbook),
        ),
        Tool(
            name="delete_user_playbook",
            args_model=DeleteUserPlaybookArgs,
            handler=_bundle_handler(_handle_delete_user_playbook),
        ),
        _FINISH_TOOL,
    ]
)

# Backward-compat alias: exposes all four create/delete tools.
# New production code should use PROFILE_EXTRACTION_TOOLS or
# PLAYBOOK_EXTRACTION_TOOLS to restrict the LLM to the correct entity kind.
EXTRACTION_TOOLS = ToolRegistry(
    [
        *_READ_TOOLS,
        Tool(
            name="create_user_profile",
            args_model=CreateUserProfileArgs,
            handler=_bundle_handler(_handle_create_user_profile),
        ),
        Tool(
            name="delete_user_profile",
            args_model=DeleteUserProfileArgs,
            handler=_bundle_handler(_handle_delete_user_profile),
        ),
        Tool(
            name="create_user_playbook",
            args_model=CreateUserPlaybookArgs,
            handler=_bundle_handler(_handle_create_user_playbook),
        ),
        Tool(
            name="delete_user_playbook",
            args_model=DeleteUserPlaybookArgs,
            handler=_bundle_handler(_handle_delete_user_playbook),
        ),
        _FINISH_TOOL,
    ]
)


SEARCH_TOOLS = ToolRegistry(
    [
        Tool(
            name="search_user_profiles",
            args_model=SearchUserProfilesArgs,
            handler=_bundle_handler(_handle_search_user_profiles),
        ),
        Tool(
            name="get_user_profile",
            args_model=GetUserProfileArgs,
            handler=_bundle_handler(_handle_get_user_profile),
        ),
        Tool(
            name="rerank_user_profiles",
            args_model=RerankUserProfilesArgs,
            handler=_bundle_handler(_handle_rerank_user_profiles),
        ),
        Tool(
            name="storage_stats",
            args_model=StorageStatsArgs,
            handler=_bundle_handler(_handle_storage_stats),
        ),
        Tool(
            name="search_user_playbooks",
            args_model=SearchUserPlaybooksArgs,
            handler=_bundle_handler(_handle_search_user_playbooks),
        ),
        Tool(
            name="get_user_playbook",
            args_model=GetUserPlaybookArgs,
            handler=_bundle_handler(_handle_get_user_playbook),
        ),
        Tool(
            name="search_agent_playbooks",
            args_model=SearchAgentPlaybooksArgs,
            handler=_bundle_handler(_handle_search_agent_playbooks),
        ),
        Tool(
            name="get_agent_playbook",
            args_model=GetAgentPlaybookArgs,
            handler=_bundle_handler(_handle_get_agent_playbook),
        ),
        Tool(
            name="get_session_excerpt",
            args_model=GetSessionExcerptArgs,
            handler=_bundle_handler(_handle_get_session_excerpt),
        ),
        Tool(
            name="finish",
            args_model=SearchFinishArgs,
            handler=_bundle_handler(_handle_search_finish),
        ),
    ]
)
