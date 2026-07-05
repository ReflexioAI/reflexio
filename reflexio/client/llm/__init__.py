"""Drop-in wrapper around an arbitrary LLM client that auto-publishes to Reflexio.

Replace your LLM client with ``wrap_llm_client(client, ...)`` and call it exactly
as before. After each completion call the wrapper publishes the turn (the clean
user utterance you supply + the assistant response) to Reflexio in the background.

    from reflexio import wrap_llm_client
    from openai import OpenAI

    client = wrap_llm_client(OpenAI(), reflexio={"source": "my-app"})
    client.chat.completions.create(
        model="gpt-4o",
        messages=[...],
        reflexio={"user_id": "alice", "session_id": "s1",
                  "user_content": "what's the weather in SF?"},
    )
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

# Import the submodule (not the ``reflexio.client`` package) to avoid an import
# cycle: this subpackage is imported from ``reflexio.client.__init__``.
from reflexio.client.client import ReflexioClient

from .adapters import (
    AnthropicAdapter,
    BaseAdapter,
    LLMAdapter,
    LLMCallContext,
    OpenAIChatAdapter,
    OpenAIResponsesAdapter,
    default_adapters,
)
from .proxy import WrapContext, _AttrPathProxy
from .publisher import Publisher, ReflexioParams, to_param_dict, validate_keys


def wrap_llm_client(
    client: Any,
    reflexio: ReflexioParams | dict[str, Any] | None = None,
    *,
    reflexio_client: ReflexioClient | None = None,
    adapters: Sequence[LLMAdapter] | None = None,
    user_content_extractor: Callable[[LLMCallContext], str | None] | None = None,
    publish_filter: Callable[[LLMCallContext], bool] | None = None,
) -> Any:
    """Wrap an LLM client so its completion calls auto-publish to Reflexio.

    The wrapped client is used exactly like the original; pass a ``reflexio={...}``
    dict per call (and/or as wrap-time defaults here) with the publish params.

    Args:
        client: The LLM client/module to wrap (OpenAI, AsyncOpenAI, litellm,
            Anthropic, or an OpenAI-compatible client such as OpenRouter).
        reflexio: Optional default ``ReflexioParams`` merged under each call's
            per-call ``reflexio={...}`` (per-call wins key-by-key). Keys:
            ``user_id``, ``session_id``, ``user_content``, ``source``,
            ``agent_version``, ``publish``, ``publish_partial_stream``,
            ``skip_aggregation``, ``force_extraction``, ``evaluation_only``.
        reflexio_client: Reflexio client for publishing. Defaults to
            ``ReflexioClient()`` (reads ``REFLEXIO_API_KEY`` / ``REFLEXIO_URL``).
        adapters: Custom adapters; prepended to the built-ins (so they win on
            overlapping call paths). Defaults to the built-ins only.
        user_content_extractor: Optional callback to derive the clean user
            utterance from a call when ``user_content`` is not supplied. May call
            ``ctx.adapter.default_user_content(ctx)``.
        publish_filter: Optional callback; return ``False`` to suppress publishing
            for a given call (e.g. internal/subagent calls).

    Returns:
        A transparent proxy around ``client``. (Note: ``isinstance`` checks
        against the original SDK class will not see through the proxy.)
    """
    # Wrap-time defaults may be partial (e.g. just source/agent_version); identity
    # is enforced per call on the merged result. Validate keys only here.
    defaults: dict[str, Any] = to_param_dict(reflexio)
    validate_keys(defaults)

    adapter_list: tuple[LLMAdapter, ...] = (
        (*adapters, *default_adapters()) if adapters else tuple(default_adapters())
    )
    namespace_prefixes: frozenset[tuple[str, ...]] = frozenset().union(
        *(a.namespace_prefixes for a in adapter_list)
    )

    ctx = WrapContext(
        adapters=adapter_list,
        defaults=defaults,
        publisher=Publisher(reflexio_client or ReflexioClient()),
        namespace_prefixes=namespace_prefixes,
        user_content_extractor=user_content_extractor,
        publish_filter=publish_filter,
    )
    return _AttrPathProxy(client, (), ctx)


__all__ = [
    "wrap_llm_client",
    "LLMAdapter",
    "LLMCallContext",
    "BaseAdapter",
    "OpenAIChatAdapter",
    "OpenAIResponsesAdapter",
    "AnthropicAdapter",
    "ReflexioParams",
    "default_adapters",
]
