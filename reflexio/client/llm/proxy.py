"""Transparent attribute-path proxy that auto-publishes intercepted LLM calls."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .adapters import LLMAdapter, LLMCallContext
from .publisher import Publisher, merge_and_validate

if TYPE_CHECKING:
    from reflexio.models.api_schema.domain.entities import InteractionData

logger = logging.getLogger(__name__)

# Reserved kwarg carrying the per-call ``reflexio={...}`` params dict.
RESERVED_KW = "reflexio"


@dataclass
class WrapContext:
    """Shared, per-wrap configuration (no per-call/per-session state)."""

    adapters: tuple[LLMAdapter, ...]
    defaults: dict[str, Any]
    publisher: Publisher
    namespace_prefixes: frozenset[tuple[str, ...]]
    user_content_extractor: Callable[[LLMCallContext], str | None] | None = None
    publish_filter: Callable[[LLMCallContext], bool] | None = None


def _match_adapter(
    adapters: tuple[LLMAdapter, ...], path: tuple[str, ...], attr: Any
) -> LLMAdapter | None:
    """First adapter (in order) that claims this call path, else ``None``."""
    for adapter in adapters:
        if adapter.is_completion_call(path, attr):
            return adapter
    return None


class _AttrPathProxy:
    """Forwards attribute access to a wrapped client, intercepting completion calls.

    Only known namespace nodes (``adapter.namespace_prefixes``) are re-wrapped and
    matched completion leaves are intercepted; everything else passes through raw,
    so the wrapped client behaves exactly like the original. Note: ``isinstance``
    checks against the concrete SDK class will not see through the proxy.
    """

    __slots__ = ("_target", "_path", "_ctx")

    def __init__(self, target: Any, path: tuple[str, ...], ctx: WrapContext) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_ctx", ctx)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        path = object.__getattribute__(self, "_path")
        ctx = object.__getattribute__(self, "_ctx")
        attr = getattr(target, name)  # AttributeError propagates (preserves hasattr)
        new_path = path + (name,)

        if callable(attr):
            adapter = _match_adapter(ctx.adapters, new_path, attr)
            if adapter is not None:
                return _make_leaf(attr, new_path, ctx, adapter)
        if new_path in ctx.namespace_prefixes:
            return _AttrPathProxy(attr, new_path, ctx)
        return attr

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Supports wrapping a bare module-level callable directly, e.g.
        # ``wrap_llm_client(litellm.completion)(...)``. The path is empty for a
        # bare callable, so derive a single-segment path from the callable's
        # name (e.g. ``("completion",)``) so adapters can still match it.
        target = object.__getattribute__(self, "_target")
        path = object.__getattribute__(self, "_path")
        ctx = object.__getattribute__(self, "_ctx")
        effective_path = path or (getattr(target, "__name__", ""),)
        adapter = _match_adapter(ctx.adapters, effective_path, target)
        if adapter is not None:
            return _make_leaf(target, effective_path, ctx, adapter)(*args, **kwargs)
        return target(*args, **kwargs)

    def __repr__(self) -> str:
        return f"<ReflexioWrapped {object.__getattribute__(self, '_target')!r}>"


def _make_leaf(
    func: Callable[..., Any],
    path: tuple[str, ...],
    ctx: WrapContext,
    adapter: LLMAdapter,
) -> Callable[..., Any]:
    """Wrap the terminal completion callable so it forwards then publishes."""

    def leaf(*args: Any, **kwargs: Any) -> Any:
        # Pop + merge + validate the reflexio params BEFORE calling the provider,
        # so a config contradiction raises at the call site (provider untouched).
        call_reflexio = kwargs.pop(RESERVED_KW, None)
        params = merge_and_validate(ctx.defaults, call_reflexio)

        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return _await_then_publish(result, args, kwargs, path, ctx, adapter, params)
        return _handle_result(result, args, kwargs, path, ctx, adapter, params)

    return leaf


def _build_call_context(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    path: tuple[str, ...],
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> LLMCallContext:
    llmctx = LLMCallContext(
        path=path,
        args=args,
        kwargs=kwargs,
        adapter=adapter,
        source=params.get("source", ""),
        user_id=params.get("user_id", ""),
        session_id=params.get("session_id", ""),
        agent_version=params.get("agent_version", ""),
    )
    user_content = params.get("user_content")
    if not (user_content and user_content.strip()) and ctx.user_content_extractor:
        # A buggy extractor must not lose the turn: fail safe to no user turn
        # (the assistant turn is still published).
        try:
            user_content = ctx.user_content_extractor(llmctx)
        except Exception:
            logger.exception("reflexio wrapper: user_content_extractor raised")
            user_content = None
    llmctx.user_content = (
        user_content if (user_content and user_content.strip()) else None
    )
    return llmctx


def _should_publish(
    params: dict[str, Any], llmctx: LLMCallContext, ctx: WrapContext
) -> bool:
    if not params.get("publish", True):
        return False
    if ctx.publish_filter is not None:
        # A buggy filter must not raise into the call: fail safe to skip publish.
        try:
            if not ctx.publish_filter(llmctx):
                return False
        except Exception:
            logger.exception("reflexio wrapper: publish_filter raised; skipping publish")
            return False
    if not (llmctx.user_id and llmctx.session_id):
        logger.warning("reflexio wrapper: missing user_id/session_id; skipping publish")
        return False
    return True


def _safe_build(
    adapter: LLMAdapter, llmctx: LLMCallContext, response: Any
) -> list[InteractionData]:
    try:
        return adapter.build_interactions(llmctx, response)
    except Exception:
        logger.exception("reflexio wrapper: failed to build interactions")
        return []


def _safe_assemble_stream(
    adapter: LLMAdapter, llmctx: LLMCallContext, chunks: list[Any]
) -> list[InteractionData]:
    try:
        return adapter.assemble_stream(llmctx, chunks)
    except Exception:
        logger.exception("reflexio wrapper: failed to assemble streamed interactions")
        return []


def _handle_result(
    result: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    path: tuple[str, ...],
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> Any:
    # Best-effort: user hooks (user_content_extractor / publish_filter) and
    # batch building must never raise into the caller's LLM call.
    try:
        llmctx = _build_call_context(args, kwargs, path, ctx, adapter, params)
        if not _should_publish(params, llmctx, ctx):
            return result
        if kwargs.get("stream"):
            return _tee_stream(result, llmctx, ctx, adapter, params)
        ctx.publisher.publish(_safe_build(adapter, llmctx, result), params)
    except Exception:
        logger.exception("reflexio wrapper: failed to schedule publish")
    return result


async def _await_then_publish(
    awaitable: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    path: tuple[str, ...],
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> Any:
    result = await awaitable
    # Best-effort: user hooks and batch building must never raise into the caller.
    try:
        llmctx = _build_call_context(args, kwargs, path, ctx, adapter, params)
        if not _should_publish(params, llmctx, ctx):
            return result
        if kwargs.get("stream"):
            return _tee_async_stream(result, llmctx, ctx, adapter, params)
        # Publish on the thread pool (publish_interaction is sync requests).
        ctx.publisher.publish(_safe_build(adapter, llmctx, result), params)
    except Exception:
        logger.exception("reflexio wrapper: failed to schedule publish")
    return result


def _tee_stream(
    stream: Any,
    llmctx: LLMCallContext,
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> Any:
    """Yield chunks unchanged; publish only on normal exhaustion (unless opted in)."""
    chunks: list[Any] = []
    completed = False
    try:
        for chunk in stream:
            chunks.append(chunk)
            yield chunk
        completed = True
    finally:
        _publish_stream(completed, chunks, llmctx, ctx, adapter, params)


async def _tee_async_stream(
    stream: Any,
    llmctx: LLMCallContext,
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> Any:
    chunks: list[Any] = []
    completed = False
    try:
        async for chunk in stream:
            chunks.append(chunk)
            yield chunk
        completed = True
    finally:
        _publish_stream(completed, chunks, llmctx, ctx, adapter, params)


def _publish_stream(
    completed: bool,
    chunks: list[Any],
    llmctx: LLMCallContext,
    ctx: WrapContext,
    adapter: LLMAdapter,
    params: dict[str, Any],
) -> None:
    if completed or params.get("publish_partial_stream", False):
        ctx.publisher.publish(_safe_assemble_stream(adapter, llmctx, chunks), params)
    else:
        logger.info("reflexio wrapper: stream not fully consumed; skipping publish")
