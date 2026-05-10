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

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from reflexio.models.api_schema.domain.entities import (
    PlaybookStatus,
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
    """Semantic/keyword search the current user's profiles, with optional
    cross-encoder rerank.

    Default behaviour (``rerank=False``): hybrid retrieval (BM25 + vector
    via RRF) only, return top ``top_k``. Cheap.

    Set ``rerank=True`` to add a cross-encoder reranking pass on top of the
    hybrid retrieval. The server over-fetches a wider candidate pool, scores
    each (query, content) pair with the cross-encoder, and returns the top
    ``top_k`` by descending rerank score. More accurate ranking when the
    bi-encoder near-duplicate confuses the LLM, but adds ~5-15 ms latency
    and the encoder may not weight all facets the same as the LLM would.

    Optional ``refine_with``: when set, used as the rerank query INSTEAD of
    ``query``. Only takes effect when ``rerank=True`` or ``llm_rerank=True``.
    Lets you broadly fetch ("bike maintenance") then narrow on a specific
    facet ("dollar amounts spent") without round-tripping candidate ids
    through the agent.

    Set ``llm_rerank=True`` to use an LLM relevance-judge instead of the
    cross-encoder. The LLM brings world knowledge that lexical/semantic
    cross-encoders lack — bridges brand→category gaps (e.g. "Thrive Market"
    = grocery service, "Italian designer boots" = luxury footwear) that
    sink the relevant profile under literal-keyword matches. Costs ~1 s
    of latency and one LLM call per search; use it on the FIRST pass when
    the query is brand/category/synonym-prone, NOT as a refinement step.
    Mutually exclusive with ``rerank``: when both are True, ``llm_rerank``
    wins.

    Each hit includes a ``session_id`` field — pass that to
    ``read_session_text`` if you need the verbatim source turns behind a
    profile.
    """

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    rerank: bool = False
    llm_rerank: bool = False
    refine_with: str | None = None


class GetUserProfileArgs(BaseModel):
    """Retrieve a single UserProfile by id."""

    id: Annotated[str, Field(min_length=1)]


class SearchUserPlaybooksArgs(BaseModel):
    """Search the current user's playbooks. See SearchUserProfilesArgs for
    the rerank semantics — same toggles, same handler shape.
    """

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"
    rerank: bool = False
    llm_rerank: bool = False
    refine_with: str | None = None


class GetUserPlaybookArgs(BaseModel):
    """Retrieve a single UserPlaybook by id."""

    id: Annotated[str, Field(min_length=1)]


class SearchAgentPlaybooksArgs(BaseModel):
    """Search agent-version-scoped playbooks (read-only; search pipeline only).
    See SearchUserProfilesArgs for the rerank semantics.
    """

    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"
    rerank: bool = False
    llm_rerank: bool = False
    refine_with: str | None = None


class GetAgentPlaybookArgs(BaseModel):
    """Retrieve a single AgentPlaybook by id."""

    id: Annotated[str, Field(min_length=1)]


class ReadSessionTextArgs(BaseModel):
    """Retrieve and compress the verbatim turns of one or more sessions.

    Fetches raw interactions for each ``session_id`` and runs an in-tool
    compression pass against ``query`` so the returned text is a denoised
    excerpt focused on the question — not the full transcript. Sessions are
    concatenated with ``=== session <id> ===`` headers. Falls back to
    role-prefixed raw turns (truncated at ``max_chars_per_session``) when
    compression is unavailable or fails.

    Use after ``search_user_profiles`` when the profile content compresses
    away detail you need (multi-action structure, exact dates buried in
    narrative, prior assistant statements). Pass 1–4 session_ids in a single
    call when the question requires data from multiple sessions (e.g.
    ordering 4 events, summing per-event amounts across 3 transactions).

    The ``query`` field is what compression scores against — typically the
    user's question or a focused subquery. Pass an empty string to skip
    compression and receive raw role-prefixed turns instead.

    The ``session_ids`` come from profile hits' ``session_id`` field (the
    same value the search response exposes). ``max_chars_per_session``
    bounds the raw-fallback output; compressed output is bounded by the
    compressor's own discretion (typically much smaller).
    """

    session_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    query: str = ""
    max_chars_per_session: int = 16000


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
    source_span: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Verbatim excerpt from the source conversation that most "
                "directly supports this profile item. Quote the original turn "
                "verbatim — do NOT paraphrase, summarise, or copy the value of "
                "the `content` field. Include enough surrounding words for the "
                "quote to stand on its own (one sentence is usually enough); "
                "preserve any temporal qualifiers, names, numbers, or exact "
                "phrases from the original."
            ),
        ),
    ]


class DeleteUserProfileArgs(BaseModel):
    """Propose deleting an existing UserProfile by id."""

    id: Annotated[str, Field(min_length=1)]


class CreateUserPlaybookArgs(BaseModel):
    """Propose creating a new UserPlaybook record."""

    trigger: Annotated[str, Field(min_length=1)]
    content: Annotated[str, Field(min_length=1)]
    rationale: str = ""
    strength: PlaybookStrength = "soft"
    source_span: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Verbatim excerpt from the source conversation that most "
                "directly supports this playbook entry. Quote the original "
                "turn verbatim — do NOT paraphrase, summarise, or copy the "
                "value of the `content` field. Include enough surrounding "
                "words for the quote to stand on its own (one sentence is "
                "usually enough); preserve any temporal qualifiers, names, "
                "numbers, or exact phrases from the original."
            ),
        ),
    ]


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


def _agent_playbook_statuses_from_ctx(ctx: ExtractionCtx) -> list[PlaybookStatus]:
    values = ctx.agent_playbook_status_filter or ["approved", "pending"]
    out: list[PlaybookStatus] = []
    for value in values:
        try:
            status = PlaybookStatus(value)
        except ValueError:
            continue
        if status not in out:
            out.append(status)
    return out


RERANK_POOL_SIZE = 30
"""How many candidates we hand the reranker when ``rerank=True`` or
``llm_rerank=True``.

If ``final_k`` exceeds this constant (e.g. agent asks for top_k=50 with
rerank=True), we still fetch ``final_k`` so we have a full set to return.
"""


def _fetch_k_for_rerank(final_k: int, rerank: bool, llm_rerank: bool = False) -> int:
    """Pick the initial-fetch size given whether any rerank stage is enabled.

    Without rerank: trust the hybrid retrieval order, fetch exactly ``final_k``.
    With rerank (cross-encoder OR LLM): fetch ``RERANK_POOL_SIZE`` candidates
    so the reranker has headroom to reorder. If ``final_k`` already exceeds
    the pool size, use ``final_k`` (agent asked for more than we'd otherwise
    fetch).

    Args:
        final_k (int): Number of hits the agent asked for.
        rerank (bool): Whether the cross-encoder rerank stage is enabled.
        llm_rerank (bool): Whether the LLM rerank stage is enabled.

    Returns:
        int: The initial-fetch size for hybrid retrieval.
    """
    if not (rerank or llm_rerank):
        return final_k
    return max(final_k, RERANK_POOL_SIZE)


def _maybe_rerank_hits(
    hits: list[Any],
    rerank: bool,
    rerank_query: str,
    final_k: int,
    llm_rerank: bool = False,
    llm_client: Any | None = None,
    prompt_manager: Any | None = None,
) -> list[Any]:
    """Apply rerank if any rerank flag is set and we have headroom.

    Dispatch order: ``llm_rerank`` (if True and infra available) takes
    precedence over ``rerank`` (cross-encoder). Both fall back to hybrid
    order on any failure. ``llm_rerank`` further falls back to the
    cross-encoder if the LLM call fails AND ``rerank`` was also requested.

    Args:
        hits: Candidates from the initial hybrid retrieval. Each must expose
            a ``content`` attribute (profile, playbook, etc).
        rerank: Cross-encoder toggle from the agent's tool args.
        rerank_query: The string to score candidates against. The caller
            decides whether this is the original query or a refinement
            (e.g. ``args.refine_with or args.query``).
        final_k: Number of hits to return after reranking.
        llm_rerank: LLM-rerank toggle. When True (and ``llm_client`` and
            ``prompt_manager`` are wired in), the LLM judges relevance.
            Use this for synonym/brand/category-knowledge gaps where the
            cross-encoder misses semantic equivalents.
        llm_client: A ``LiteLLMClient`` for the LLM rerank call. Required
            when ``llm_rerank=True``.
        prompt_manager: A ``PromptManager`` for rendering the rerank prompt.
            Required when ``llm_rerank=True``.

    Returns:
        The re-ordered hits, capped at ``final_k``. If neither flag is set
        or ``len(hits) <= final_k``, returns ``hits[:final_k]`` unchanged.
    """
    if not (rerank or llm_rerank) or len(hits) <= final_k:
        return hits[:final_k]

    if llm_rerank:
        ordered = _try_llm_rerank(hits, rerank_query, final_k, llm_client, prompt_manager)
        if ordered is not None:
            return ordered
        # LLM rerank failed — fall through to cross-encoder ONLY if also requested.
        if not rerank:
            return hits[:final_k]

    return _try_cross_encoder_rerank(hits, rerank_query, final_k)


def _try_llm_rerank(
    hits: list[Any],
    rerank_query: str,
    final_k: int,
    llm_client: Any | None,
    prompt_manager: Any | None,
) -> list[Any] | None:
    """Run the LLM rerank stage; return ``None`` on any failure.

    Args:
        hits: Candidate hits to rerank.
        rerank_query: Query to score against.
        final_k: How many to return.
        llm_client: ``LiteLLMClient`` instance, or ``None``.
        prompt_manager: ``PromptManager`` instance, or ``None``.

    Returns:
        Re-ordered top-``final_k`` on success; ``None`` on failure so the
        caller can chain to the cross-encoder fallback.
    """
    try:
        from reflexio.server.llm.rerank import score_pairs_llm
    except ImportError:
        return None
    scores = score_pairs_llm(
        rerank_query, [h.content for h in hits], llm_client, prompt_manager
    )
    if scores is None:
        return None
    ranked = sorted(
        zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True
    )
    return [h for h, _ in ranked[:final_k]]


def _try_cross_encoder_rerank(
    hits: list[Any], rerank_query: str, final_k: int
) -> list[Any]:
    """Run the cross-encoder rerank stage; fall back to hybrid order on failure.

    Args:
        hits: Candidate hits to rerank.
        rerank_query: Query to score against.
        final_k: How many to return.

    Returns:
        Re-ordered top-``final_k`` on success; ``hits[:final_k]`` on failure.
    """
    try:
        from reflexio.server.llm.rerank import score_pairs

        scores = score_pairs(rerank_query, [h.content for h in hits])
        ranked = sorted(
            zip(hits, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [h for h, _ in ranked[:final_k]]
    except Exception:  # noqa: BLE001 — fall back to hybrid order on failure
        return hits[:final_k]


def _project_profile_for_llm(p: Any) -> dict[str, Any]:
    # ``session_id`` is the agent-facing name for the storage-internal
    # ``generated_from_request_id``. Exposing it lets the agent chain
    # ``search_user_profiles`` → ``read_session_text(session_ids=[...])``
    # without needing a separate lookup. The internal field name stays as
    # ``generated_from_request_id`` everywhere else (storage, schemas);
    # the rename is purely a cognitive affordance for the LLM.
    return {
        "id": getattr(p, "profile_id", "") or "",
        "content": p.content,
        "session_id": getattr(p, "generated_from_request_id", "") or "",
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
    args: SearchUserProfilesArgs,
    storage: Any,
    ctx: ExtractionCtx,
    llm_client: Any | None = None,
    prompt_manager: Any | None = None,
) -> dict[str, Any]:
    """Search the current user's profiles and bump search_count.

    Two-stage retrieval: hybrid (BM25 + vector via RRF) over-fetches a wider
    candidate pool, then an optional rerank scores ``(query, content)`` pairs
    and returns the top ``args.top_k``. ``args.rerank`` selects the local
    cross-encoder; ``args.llm_rerank`` selects an LLM relevance-judge that
    has world knowledge for brand→category gaps the cross-encoder can't
    bridge. ``llm_rerank`` wins when both are set.

    The over-fetch + rerank pattern fixes the class of failures where the
    bi-encoder ranks the right profile at #2-#15 by cosine but the top-1
    is a near-duplicate or literal-keyword match that the answer LLM picks
    first.

    Args:
        args (SearchUserProfilesArgs): Query, top_k, rerank toggles, refine_with.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count incremented in place.
        llm_client (Any | None): ``LiteLLMClient`` for ``llm_rerank=True``.
        prompt_manager (Any | None): ``PromptManager`` for ``llm_rerank=True``.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing profile projections.
    """
    final_k = _cap_top_k(args.top_k)
    fetch_k = _fetch_k_for_rerank(final_k, args.rerank, args.llm_rerank)

    request = SearchUserProfileRequest(
        query=args.query,
        user_id=ctx.user_id,
        top_k=fetch_k,
    )
    hits = storage.search_user_profile(
        request,
        query_embedding=_maybe_embed_query(storage, args.query),
    )
    ctx.search_count += 1

    hits = _maybe_rerank_hits(
        hits=hits,
        rerank=args.rerank,
        rerank_query=args.refine_with or args.query,
        final_k=final_k,
        llm_rerank=args.llm_rerank,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
    )

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
    args: SearchUserPlaybooksArgs,
    storage: Any,
    ctx: ExtractionCtx,
    llm_client: Any | None = None,
    prompt_manager: Any | None = None,
) -> dict[str, Any]:
    """Search the current user's playbooks and bump search_count.

    Args:
        args (SearchUserPlaybooksArgs): Query, top_k, status filter, rerank toggles.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count and known_ids updated.
        llm_client (Any | None): ``LiteLLMClient`` for ``llm_rerank=True``.
        prompt_manager (Any | None): ``PromptManager`` for ``llm_rerank=True``.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing playbook projections.
    """
    final_k = _cap_top_k(args.top_k)
    fetch_k = _fetch_k_for_rerank(final_k, args.rerank, args.llm_rerank)
    request = SearchUserPlaybookRequest(
        query=args.query,
        user_id=ctx.user_id,
        agent_version=ctx.agent_version,
        top_k=fetch_k,
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
    hits = _maybe_rerank_hits(
        hits=hits,
        rerank=args.rerank,
        rerank_query=args.refine_with or args.query,
        final_k=final_k,
        llm_rerank=args.llm_rerank,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
    )
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
    args: SearchAgentPlaybooksArgs,
    storage: Any,
    ctx: ExtractionCtx,
    llm_client: Any | None = None,
    prompt_manager: Any | None = None,
) -> dict[str, Any]:
    """Search agent-version-scoped playbooks and bump search_count.

    Args:
        args (SearchAgentPlaybooksArgs): Query, top_k, status filter, rerank toggles.
        storage (Any): BaseStorage instance.
        ctx (ExtractionCtx): Per-run state; search_count and known_ids updated.
        llm_client (Any | None): ``LiteLLMClient`` for ``llm_rerank=True``.
        prompt_manager (Any | None): ``PromptManager`` for ``llm_rerank=True``.

    Returns:
        dict[str, Any]: ``{"hits": [...]}`` with LLM-facing agent playbook projections.
    """
    final_k = _cap_top_k(args.top_k)
    fetch_k = _fetch_k_for_rerank(final_k, args.rerank, args.llm_rerank)
    options = SearchOptions(query_embedding=_maybe_embed_query(storage, args.query))
    hits: list[Any] = []
    seen_ids: set[str] = set()
    for playbook_status in _agent_playbook_statuses_from_ctx(ctx):
        request = SearchAgentPlaybookRequest(
            query=args.query,
            agent_version=ctx.agent_version,
            top_k=fetch_k,
            status_filter=[_status_from_str(args.status)],
            playbook_status_filter=playbook_status,
            search_mode=SearchMode.HYBRID,
            threshold=0.4,
        )
        if ctx.extractor_name:
            request.playbook_name = ctx.extractor_name
        for hit in storage.search_agent_playbooks(request, options=options):
            hit_id = str(getattr(hit, "agent_playbook_id", ""))
            if hit_id and hit_id not in seen_ids:
                seen_ids.add(hit_id)
                hits.append(hit)
                if len(hits) >= fetch_k:
                    break
        if len(hits) >= fetch_k:
            break
    ctx.search_count += 1
    hits = _maybe_rerank_hits(
        hits=hits,
        rerank=args.rerank,
        rerank_query=args.refine_with or args.query,
        final_k=final_k,
        llm_rerank=args.llm_rerank,
        llm_client=llm_client,
        prompt_manager=prompt_manager,
    )
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


_COMPRESS_PROMPT_ID = "compress_session_for_query"
# 5s was far too aggressive: under 10-worker benchmark concurrency every
# gpt-5-mini compression call timed out, silently degrading every
# rehydration to raw turns. 30s gives gpt-5-mini room under load while
# still failing fast on genuine model outages (the raw-turns fallback
# kicks in beyond that).
_COMPRESS_LLM_TIMEOUT = 30
_COMPRESS_LLM_MAX_RETRIES = 1


def _format_raw_turns(
    session_ids: list[str],
    by_session: dict[str, list[Any]],
    max_chars_per_session: int,
) -> str:
    """Format role-prefixed raw turns with per-session caps.

    Used both as the compressor's input and as the fallback output when
    compression is unavailable or fails.

    Args:
        session_ids (list[str]): Session ids in the order requested by the agent.
        by_session (dict[str, list[Any]]): Interactions grouped by session id.
        max_chars_per_session (int): Per-session truncation cap (chars).

    Returns:
        str: Concatenated ``=== session <id> ===\\n[role] content`` blocks.
    """
    blocks: list[str] = []
    for sid in session_ids:
        sess_interactions = by_session.get(sid, [])
        if not sess_interactions:
            blocks.append(f"=== session {sid} ===\n(no interactions found)")
            continue
        # Derive the session date from the earliest interaction timestamp.
        # Including this in the header lets the downstream answer LLM resolve
        # relative-time phrases ("yesterday", "last week", "X days ago") in
        # the raw turns against an absolute anchor — without it, the LLM ends
        # up quoting the relative phrase verbatim instead of computing the
        # date (the multi-hop temporal failure pattern).
        timestamps = [
            getattr(i, "created_at", 0)
            for i in sess_interactions
            if getattr(i, "created_at", 0)
        ]
        date_suffix = ""
        if timestamps:
            try:
                date_iso = datetime.fromtimestamp(min(timestamps), tz=UTC).strftime("%Y-%m-%d")
                date_suffix = f" (date: {date_iso})"
            except (OverflowError, OSError, ValueError):
                pass
        lines = [
            f"[{getattr(i, 'role', '?')}] {i.content}"
            for i in sess_interactions
            if getattr(i, "content", None)
        ]
        body = "\n".join(lines)
        if len(body) > max_chars_per_session:
            body = body[:max_chars_per_session] + "…"
        blocks.append(f"=== session {sid}{date_suffix} ===\n{body}")
    return "\n\n".join(blocks)


def _compress_raw_turns(
    raw_turns: str,
    query: str,
    llm_client: Any,
    prompt_manager: Any,
) -> str | None:
    """Run the compress_session_for_query prompt against raw_turns + query.

    Returns the compressed text on success, or ``None`` on any failure
    (timeout, exception, empty/whitespace output). Caller falls back to
    raw_turns when ``None`` is returned.

    Args:
        raw_turns (str): The role-prefixed raw transcript (output of
            ``_format_raw_turns``).
        query (str): The query the compressor scores against.
        llm_client (Any): LiteLLMClient with a ``generate_response`` method.
        prompt_manager (Any): PromptManager with a ``render_prompt`` method.

    Returns:
        str | None: Compressed transcript, or None to indicate fallback.
    """
    # Diagnostic: hard-disable compression to measure its net impact vs raw
    # turns. r60 measured working v1.2.0 cost ~4pp; r65 confirmed v1.3.0
    # near-pass-through still cost. Returning None here forces the raw-turns
    # fallback path — same behavior as the original 5s-timeout era at r45.
    return None

    try:  # noqa  pragma: no cover — disabled compression path
        prompt = prompt_manager.render_prompt(
            _COMPRESS_PROMPT_ID,
            variables={"query": query, "raw_turns": raw_turns},
        )
        result = llm_client.generate_response(
            prompt,
            timeout=_COMPRESS_LLM_TIMEOUT,
            max_retries=_COMPRESS_LLM_MAX_RETRIES,
        )
    except Exception as e:  # noqa: BLE001 — broad: LLM stack raises diverse exception types
        logger.warning("read_session_text compression failed, using raw fallback: %s", e)
        return None

    if not isinstance(result, str) or not result.strip():
        logger.warning("read_session_text compression returned empty, using raw fallback")
        return None
    return result.strip()


def _handle_read_session_text(
    args: ReadSessionTextArgs,
    storage: Any,
    ctx: ExtractionCtx,
    llm_client: Any = None,
    prompt_manager: Any = None,
) -> dict[str, Any]:
    """Return a denoised excerpt of one or more sessions, scored against ``query``.

    Fetches raw interactions for ``session_ids`` and — when ``query`` is
    non-empty and the compression layer is wired — runs an in-tool LLM call
    that compresses the raw turns into the smallest excerpt preserving every
    operand relevant to the query. Falls back to role-prefixed raw turns
    (truncated at ``max_chars_per_session``) on:

    - empty ``query``
    - no ``llm_client`` / ``prompt_manager`` available (test paths or
      backends that don't pass them through ``HandlerBundle``)
    - any compression-LLM exception
    - empty/whitespace compression output

    Args:
        args (ReadSessionTextArgs): Session ids (1–4), query, and per-session char cap.
        storage (Any): BaseStorage instance; must implement
            ``get_interactions_by_request_ids(request_ids: list[str])``.
        ctx (ExtractionCtx): Per-run state (unused for reads, present for consistency).
        llm_client (Any | None): LiteLLMClient for compression. Defaults to None
            so callers that don't enable compression still get raw turns.
        prompt_manager (Any | None): PromptManager for rendering the compression
            prompt. Defaults to None.

    Returns:
        dict[str, Any]: ``{"text": str}`` with compressed-or-raw session text,
            or ``{"error": str}`` when the storage backend doesn't support batch
            fetch or no interactions exist for any of the requested sessions.
    """
    try:
        interactions = storage.get_interactions_by_request_ids(args.session_ids)
    except AttributeError:
        return {"error": "read_session_text requires get_interactions_by_request_ids"}
    if not interactions:
        return {"error": "no interactions found for any of the requested sessions"}

    by_session: dict[str, list[Any]] = {}
    for i in interactions:
        sid = getattr(i, "request_id", None)
        if sid is None:
            continue
        by_session.setdefault(sid, []).append(i)

    # Clamp agent-supplied max_chars_per_session to a 16000 floor: agents
    # have been observed to pick values as low as 3000-4000 inconsistently,
    # which truncates Pattern G (recall-prior-statement) sessions before the
    # answer turn appears. The handler enforces the floor regardless of the
    # arg.
    effective_max_chars = max(args.max_chars_per_session, 16000)
    raw_text = _format_raw_turns(
        args.session_ids, by_session, effective_max_chars
    )

    # Skip compression on empty query or missing wiring; raw turns are still useful.
    if not args.query.strip() or llm_client is None or prompt_manager is None:
        ctx.rehydrated_excerpts.append(raw_text)
        return {"text": raw_text}

    compressed = _compress_raw_turns(raw_text, args.query, llm_client, prompt_manager)
    final_text = raw_text if compressed is None else compressed
    ctx.rehydrated_excerpts.append(final_text)
    return {"text": final_text}


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


def _bundle_handler_with_llm(
    inner: Callable[..., dict[str, Any]],
) -> Callable[[Any, Any], dict[str, Any]]:
    """Adapter variant for handlers that also need ``llm_client`` and
    ``prompt_manager`` from the bundle (e.g., the compression-enabled
    rehydration tool).

    The inner handler signature is
    ``(args, storage, ctx, llm_client=None, prompt_manager=None) -> dict``.
    Both LLM-side fields default to ``None`` so the handler degrades to a
    no-LLM path when callers don't wire them through.

    Args:
        inner (Callable[..., dict[str, Any]]): Handler accepting
            ``(args, storage, ctx, llm_client, prompt_manager)``.

    Returns:
        Callable[[Any, Any], dict[str, Any]]: A 2-arg callable compatible
            with ``Tool.handler``.
    """

    def wrapped(args: Any, bundle: Any) -> dict[str, Any]:
        return inner(
            args,
            bundle.storage,
            bundle.ctx,
            llm_client=getattr(bundle, "llm_client", None),
            prompt_manager=getattr(bundle, "prompt_manager", None),
        )

    return wrapped


_READ_TOOLS = [
    Tool(
        name="search_user_profiles",
        args_model=SearchUserProfilesArgs,
        handler=_bundle_handler_with_llm(_handle_search_user_profiles),
    ),
    Tool(
        name="get_user_profile",
        args_model=GetUserProfileArgs,
        handler=_bundle_handler(_handle_get_user_profile),
    ),
    Tool(
        name="search_user_playbooks",
        args_model=SearchUserPlaybooksArgs,
        handler=_bundle_handler_with_llm(_handle_search_user_playbooks),
    ),
    Tool(
        name="get_user_playbook",
        args_model=GetUserPlaybookArgs,
        handler=_bundle_handler(_handle_get_user_playbook),
    ),
    Tool(
        name="search_agent_playbooks",
        args_model=SearchAgentPlaybooksArgs,
        handler=_bundle_handler_with_llm(_handle_search_agent_playbooks),
    ),
    Tool(
        name="get_agent_playbook",
        args_model=GetAgentPlaybookArgs,
        handler=_bundle_handler(_handle_get_agent_playbook),
    ),
    Tool(
        name="read_session_text",
        args_model=ReadSessionTextArgs,
        handler=_bundle_handler_with_llm(_handle_read_session_text),
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


# ====================================================================
# Multi-stage fallback schema for non-tool-calling models
# ====================================================================
#
# When the search-agent model lacks native tool-calling (e.g.
# minimax/MiniMax-M2.7), `run_tool_loop` drives one structured-output
# call per turn using `SearchAgentTurnPlan` as the response_format. The
# server parses the result, dispatches `next_call` against `SEARCH_TOOLS`,
# appends the tool result to the message history, and loops until
# `next_call.tool == "finish"` or `max_steps` is exhausted. This
# preserves observe-decide-act semantics that single-shot fallback
# (which planned all calls upfront) could not.
#
# The discriminated union mirrors the `args_model` of every tool in
# `SEARCH_TOOLS`. Field names match the existing tool args so we can
# convert each variant directly to the dispatch JSON via
# `model_dump(exclude={"tool"})`.


class _CallSearchUserProfiles(BaseModel):
    """Multi-stage variant: call `search_user_profiles`."""

    tool: Literal["search_user_profiles"]
    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    rerank: bool = False
    refine_with: str | None = None


class _CallSearchUserPlaybooks(BaseModel):
    """Multi-stage variant: call `search_user_playbooks`."""

    tool: Literal["search_user_playbooks"]
    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"
    rerank: bool = False
    refine_with: str | None = None


class _CallSearchAgentPlaybooks(BaseModel):
    """Multi-stage variant: call `search_agent_playbooks`."""

    tool: Literal["search_agent_playbooks"]
    query: Annotated[str, Field(min_length=1)]
    top_k: int = 10
    status: Literal["current", "pending", "archived"] = "current"
    rerank: bool = False
    refine_with: str | None = None


class _CallGetUserProfile(BaseModel):
    """Multi-stage variant: call `get_user_profile`."""

    tool: Literal["get_user_profile"]
    id: Annotated[str, Field(min_length=1)]


class _CallGetUserPlaybook(BaseModel):
    """Multi-stage variant: call `get_user_playbook`."""

    tool: Literal["get_user_playbook"]
    id: Annotated[str, Field(min_length=1)]


class _CallGetAgentPlaybook(BaseModel):
    """Multi-stage variant: call `get_agent_playbook`."""

    tool: Literal["get_agent_playbook"]
    id: Annotated[str, Field(min_length=1)]


class _CallReadSessionText(BaseModel):
    """Multi-stage variant: call `read_session_text`."""

    tool: Literal["read_session_text"]
    session_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    query: str = ""
    max_chars_per_session: int = 16000


class _CallStorageStats(BaseModel):
    """Multi-stage variant: call `storage_stats` (no args)."""

    tool: Literal["storage_stats"]


class _CallFinish(BaseModel):
    """Multi-stage variant: call `finish` to terminate the loop."""

    tool: Literal["finish"]
    answer: str | None = None


_SearchToolCall = Annotated[
    _CallSearchUserProfiles
    | _CallSearchUserPlaybooks
    | _CallSearchAgentPlaybooks
    | _CallGetUserProfile
    | _CallGetUserPlaybook
    | _CallGetAgentPlaybook
    | _CallReadSessionText
    | _CallStorageStats
    | _CallFinish,
    Field(discriminator="tool"),
]


class SearchAgentTurnPlan(BaseModel):
    """One turn of the search agent's multi-stage fallback plan.

    The agent emits one ``SearchAgentTurnPlan`` per turn. The server parses
    it, dispatches ``next_call`` against ``SEARCH_TOOLS``, appends the tool
    result to the message history, and asks for the next turn — until
    ``next_call.tool == "finish"`` or ``max_steps`` is exhausted.

    Used by ``run_tool_loop`` when the configured model lacks native
    tool-calling but should still run a multi-turn observe-decide-act loop
    (e.g. ``minimax/MiniMax-M2.7``).
    """

    reasoning: Annotated[str, Field(min_length=1)]
    next_call: _SearchToolCall


SEARCH_TOOLS = ToolRegistry(
    [
        Tool(
            name="search_user_profiles",
            args_model=SearchUserProfilesArgs,
            handler=_bundle_handler_with_llm(_handle_search_user_profiles),
        ),
        Tool(
            name="get_user_profile",
            args_model=GetUserProfileArgs,
            handler=_bundle_handler(_handle_get_user_profile),
        ),
        # rerank_user_profiles intentionally removed from the agent palette:
        # `search_user_profiles` now does deterministic cross-encoder rerank
        # internally and accepts an optional `refine_with` for two-stage
        # query refinement. The standalone rerank tool required the agent
        # to round-trip profile_ids back through the model, which was both
        # cognitively expensive and a hallucination risk on long lists.
        # The handler `_handle_rerank_user_profiles` is preserved in this
        # module for any non-agent caller that needs explicit rerank.
        Tool(
            name="storage_stats",
            args_model=StorageStatsArgs,
            handler=_bundle_handler(_handle_storage_stats),
        ),
        Tool(
            name="search_user_playbooks",
            args_model=SearchUserPlaybooksArgs,
            handler=_bundle_handler_with_llm(_handle_search_user_playbooks),
        ),
        Tool(
            name="get_user_playbook",
            args_model=GetUserPlaybookArgs,
            handler=_bundle_handler(_handle_get_user_playbook),
        ),
        Tool(
            name="search_agent_playbooks",
            args_model=SearchAgentPlaybooksArgs,
            handler=_bundle_handler_with_llm(_handle_search_agent_playbooks),
        ),
        Tool(
            name="get_agent_playbook",
            args_model=GetAgentPlaybookArgs,
            handler=_bundle_handler(_handle_get_agent_playbook),
        ),
        Tool(
            name="read_session_text",
            args_model=ReadSessionTextArgs,
            handler=_bundle_handler_with_llm(_handle_read_session_text),
        ),
        Tool(
            name="finish",
            args_model=SearchFinishArgs,
            handler=_bundle_handler(_handle_search_finish),
        ),
    ]
)
