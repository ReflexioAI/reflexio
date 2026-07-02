"""Embedding concern for ``LiteLLMClient`` — mixin + token-budget helpers (Tier-2.5).

``EmbeddingMixin`` holds the three ``self``-bound embedding methods
(``_resolve_default_embedding_model``, ``get_embedding``, ``get_embeddings``); the
four stateless token-budget helpers (``_get_embedding_limit``,
``_get_embedding_encoding``, ``_reject_cloud_mode``, ``_truncate_for_embedding``),
the module-level ``_TRUNCATION_WARNED_MODELS`` set and the budget constants live
alongside them at module scope.

SINK-1 (patch-where-used): the provider/router names this module references
(``resolve_model_name``, ``get_service_embeddings``, ``should_use_embedding_service``,
``NomicEmbedder``, ``LocalEmbedder``, ``is_chromadb_importable``) are imported HERE,
so tests patch them at ``_litellm_embedding.<name>`` — patching the old facade
namespace would no-op and a no-op would hit the real embedder/137M model/network
(the P0). ``litellm.get_model_info``/``litellm.embedding`` are called via the shared
``litellm`` module attr so the global mock and ``get_model_info`` patch still apply.

SINK-2 (identity): ``_TRUNCATION_WARNED_MODELS`` and the three test-imported budget
functions are re-exported by import binding from the facade — the facade set IS this
set (the test autouse ``.clear()`` fixture mutates it through the facade).

Bodies moved VERBATIM; only the ``self``-typing (per-mixin TYPE_CHECKING stubs of
foreign members, Tier-1b idiom) is added.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import litellm
import tiktoken

from reflexio.server.llm._litellm_types import LiteLLMClientError
from reflexio.server.llm.model_defaults import ModelRole, resolve_model_name
from reflexio.server.llm.providers.embedding_service_provider import (
    EmbeddingUnavailableError,
    embedding_provider_mode,
    get_service_embeddings,
    should_use_embedding_service,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    LocalEmbedder,
)
from reflexio.server.llm.providers.local_embedding_provider import (
    is_chromadb_importable as _is_chromadb_importable,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    NomicEmbedder,
)
from reflexio.server.llm.providers.nomic_embedding_provider import (
    is_nomic_model as _is_nomic_model,
)

if TYPE_CHECKING:
    from reflexio.server.llm._litellm_types import LiteLLMConfig

_LOGGER = logging.getLogger(__name__)

# OpenAI's documented max input length for text-embedding-3-* and ada-002 is
# 8191 tokens. Used as the fallback limit only when a model's name looks
# OpenAI-family but litellm's registry has no entry for it.
_OPENAI_EMBEDDING_FALLBACK_MAX_TOKENS = 8191

# Models whose truncation warning has already been emitted this process. Keeps
# batch backfills of millions of long docs from flooding logs — the first hit
# per model goes to WARNING, everything after to DEBUG.
_TRUNCATION_WARNED_MODELS: set[str] = set()

# Model-name prefixes that route through OpenAI's embedding API (and therefore
# share the 8191-token cap). Anything that does not start with one of these is
# treated as "unknown provider" when litellm has no registry entry.
_OPENAI_EMBEDDING_FAMILY_PREFIXES = ("text-embedding-", "openai/", "azure/")


@lru_cache(maxsize=32)
def _get_embedding_limit(model: str) -> int | None:
    """
    Resolve the maximum input token count for an embedding model.

    Consults ``litellm.get_model_info`` first so provider-specific caps are
    respected (OpenAI ~8191, Cohere 512, Voyage 32000, etc.). When litellm has
    no entry for the model, falls back to the OpenAI 8191 cap only when the
    model name looks OpenAI-family; otherwise returns ``None`` to disable
    truncation for unknown providers (safer than over-truncating their input).

    Args:
        model (str): Embedding model name (e.g. 'text-embedding-3-small',
            'cohere/embed-english-v3.0').

    Returns:
        int | None: Maximum input tokens, or ``None`` when the limit is unknown
            and no safe fallback applies.
    """
    try:
        info = litellm.get_model_info(model)
    except Exception:
        info = None
    if info and info.get("mode") == "embedding":
        max_tokens = info.get("max_input_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            return max_tokens
    if model.startswith(_OPENAI_EMBEDDING_FAMILY_PREFIXES):
        return _OPENAI_EMBEDDING_FALLBACK_MAX_TOKENS
    return None


@lru_cache(maxsize=16)
def _get_embedding_encoding(model: str) -> tiktoken.Encoding:
    """
    Return the tiktoken encoding for an embedding model, falling back to cl100k_base.

    For non-OpenAI providers tiktoken does not know the real tokenizer, so the
    cl100k_base fallback is an approximate proxy for token counting. That is
    acceptable here because we truncate toward the provider's cap with the
    proxy, which tends to over-truncate by a small fraction rather than under-
    truncate and cause upstream 400s.

    Args:
        model (str): Embedding model name (e.g. 'text-embedding-3-small').

    Returns:
        tiktoken.Encoding: Encoder to use for token counting and truncation.
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _reject_cloud_mode(embedding_model: str, mode: str) -> None:
    """
    Raise when a local-only embedding model is configured for cloud mode.

    Args:
        embedding_model (str): The resolved embedding model name.
        mode (str): The resolved embedding provider mode.

    Raises:
        EmbeddingUnavailableError: If ``mode`` is ``"cloud"``.
    """
    if mode == "cloud":
        raise EmbeddingUnavailableError(
            f"Local embedding model {embedding_model!r} cannot use cloud mode"
        )


def _truncate_for_embedding(
    text: str, model: str, max_tokens: int | None = None
) -> str:
    """
    Truncate a string so its token count fits within an embedding model's input limit.

    The token budget is auto-resolved from ``_get_embedding_limit`` by default.
    When the model has no known limit (unknown provider not in litellm's
    registry and not OpenAI-family), returns the text unchanged — over-
    truncating an unknown provider's input is worse than passing it through
    and letting the provider's own error surface.

    Args:
        text (str): Raw input text.
        model (str): Embedding model name, used to pick the tokenizer and the
            per-provider token cap.
        max_tokens (int | None): Override for the resolved budget. Primarily
            used by tests to exercise the truncation path on short strings;
            leave as ``None`` in production callers.

    Returns:
        str: Original text if it already fits (or the model has no known
            limit), otherwise a token-bounded prefix.
    """
    if not text:
        return text
    if max_tokens is None:
        max_tokens = _get_embedding_limit(model)
    if max_tokens is None:
        return text
    encoding = _get_embedding_encoding(model)
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    if model in _TRUNCATION_WARNED_MODELS:
        _LOGGER.debug(
            "Truncating embedding input from %d to %d tokens for model %s",
            len(tokens),
            max_tokens,
            model,
        )
    else:
        _TRUNCATION_WARNED_MODELS.add(model)
        _LOGGER.warning(
            "Truncating embedding input from %d to %d tokens for model %s "
            "(further occurrences will be logged at DEBUG)",
            len(tokens),
            max_tokens,
            model,
        )
    return encoding.decode(tokens[:max_tokens])


class EmbeddingMixin:
    """Embedding dispatch (service / Nomic / local ONNX / litellm) for ``LiteLLMClient``.

    Mixed into ``LiteLLMClient``; the ``self`` members these methods read are
    owned by the client-core ``__init__`` on the facade. The annotation-only
    stubs below (Tier-1b idiom) give pyright the foreign-member types without
    introducing shared class-level mutable state — NEVER assign here.
    """

    # Base-owned attributes these methods read (init'd in the facade ``__init__``).
    config: "LiteLLMConfig"
    # Lazy per-instance cache; ``_resolve_default_embedding_model`` populates it,
    # ``update_config`` invalidates it. Annotation-only (no class default).
    _default_embedding_model: str | None

    if TYPE_CHECKING:
        # Client-core credential resolver (stays on the facade per the split).
        # Declared type-only so pyright can resolve ``self._resolve_api_key(...)``.
        def _resolve_api_key(
            self, model: str | None = ..., for_embedding: bool = ...
        ) -> tuple[str | None, str | None, str | None]: ...

    def _resolve_default_embedding_model(self) -> str:
        """
        Resolve the embedding model to use when callers do not specify one.

        Routes through the same auto-detection chain as the rest of reflexio
        (``resolve_model_name`` for ``ModelRole.EMBEDDING``) so a session that
        has the local ONNX embedder enabled — or any non-OpenAI provider —
        does not silently fall back to ``text-embedding-3-small`` and produce
        OpenAI 401s. Higher-precedence org config and site-var overrides are
        the caller's responsibility to resolve and pass via ``model=``; this
        helper handles only the auto-detect tier.

        Returns:
            str: The auto-detected embedding model name (cached after first call).

        Raises:
            RuntimeError: Propagated from ``resolve_model_name`` when no
                embedding-capable provider is available.
        """
        if self._default_embedding_model is None:
            self._default_embedding_model = resolve_model_name(
                ModelRole.EMBEDDING,
                api_key_config=self.config.api_key_config,
            )
        return self._default_embedding_model

    def get_embedding(
        self, text: str, model: str | None = None, dimensions: int | None = None
    ) -> list[float]:
        """
        Get embedding vector for the given text.

        Args:
            text: The text to get embedding for.
            model: Optional embedding model. When omitted, the model is
                auto-detected via ``resolve_model_name(ModelRole.EMBEDDING)``
                so callers inherit the local-embedder gate and any non-OpenAI
                provider configured for this client.
            dimensions: Optional number of dimensions for the embedding vector.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        embedding_model = model or self._resolve_default_embedding_model()
        mode = embedding_provider_mode(embedding_model)
        if mode == "off":
            raise EmbeddingUnavailableError("Embedding provider is disabled")
        if should_use_embedding_service(embedding_model):
            return get_service_embeddings(
                [text], model=embedding_model, dimensions=dimensions
            )[0]

        # local/nomic-embed-* must stay on the Nomic provider (137M params,
        # 768d Matryoshka-truncated to 512). Falling through to MiniLM would
        # mix embedding models inside existing vector stores.
        if _is_nomic_model(embedding_model):
            _reject_cloud_mode(embedding_model, mode)
            try:
                return NomicEmbedder.get().embed([text])[0]
            except Exception as e:
                raise LiteLLMClientError(
                    f"Nomic embedding generation failed: {str(e)}"
                ) from e

        # local/* models route through the in-process ONNX embedder — no
        # network call, no litellm API, no tiktoken truncation (the embedder
        # applies its own token cap). The dispatch is gated solely on
        # ``chromadb`` being importable; the env-var opt-in (claude-smart's
        # ``CLAUDE_SMART_USE_LOCAL_EMBEDDING``) is enforced earlier in the
        # auto-detection layer (see ``model_defaults._auto_detect_model``).
        if embedding_model.startswith("local/"):
            _reject_cloud_mode(embedding_model, mode)
            if not _is_chromadb_importable():
                raise LiteLLMClientError(
                    f"Embedding model {embedding_model!r} requires chromadb. "
                    "Run `pip install chromadb`."
                )
            try:
                return LocalEmbedder.get().embed([text])[0]
            except Exception as e:
                raise LiteLLMClientError(
                    f"Local embedding generation failed: {str(e)}"
                ) from e

        text = _truncate_for_embedding(text, embedding_model)

        try:
            params = {"model": embedding_model, "input": [text]}
            if dimensions:
                params["dimensions"] = dimensions

            # Resolve and add API key configuration if provided (overrides env vars)
            api_key, api_base, api_version = self._resolve_api_key(
                embedding_model, for_embedding=True
            )
            if api_key:
                params["api_key"] = api_key
            if api_base:
                params["api_base"] = api_base
            if api_version:
                params["api_version"] = api_version

            response = litellm.embedding(
                **params,
                timeout=self.config.timeout,
                num_retries=self.config.max_retries,
            )
            return response.data[0]["embedding"]
        except Exception as e:
            raise LiteLLMClientError(f"Embedding generation failed: {str(e)}") from e

    def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[list[float]]:
        """
        Get embedding vectors for multiple texts in a single API call.

        Args:
            texts: List of texts to get embeddings for.
            model: Optional embedding model. When omitted, the model is
                auto-detected via ``resolve_model_name(ModelRole.EMBEDDING)``
                so callers inherit the local-embedder gate and any non-OpenAI
                provider configured for this client.
            dimensions: Optional number of dimensions for the embedding vectors.

        Returns:
            List of embedding vectors, one per input text, in the same order as input.

        Raises:
            LiteLLMClientError: If embedding generation fails.
        """
        if not texts:
            return []

        embedding_model = model or self._resolve_default_embedding_model()
        mode = embedding_provider_mode(embedding_model)
        if mode == "off":
            raise EmbeddingUnavailableError("Embedding provider is disabled")
        if should_use_embedding_service(embedding_model):
            return get_service_embeddings(
                list(texts), model=embedding_model, dimensions=dimensions
            )

        # See matching short-circuits in get_embedding above.
        if _is_nomic_model(embedding_model):
            _reject_cloud_mode(embedding_model, mode)
            try:
                return NomicEmbedder.get().embed(list(texts))
            except Exception as e:
                raise LiteLLMClientError(
                    f"Nomic batch embedding generation failed: {str(e)}"
                ) from e

        if embedding_model.startswith("local/"):
            _reject_cloud_mode(embedding_model, mode)
            if not _is_chromadb_importable():
                raise LiteLLMClientError(
                    f"Embedding model {embedding_model!r} requires chromadb. "
                    "Run `pip install chromadb`."
                )
            try:
                return LocalEmbedder.get().embed(list(texts))
            except Exception as e:
                raise LiteLLMClientError(
                    f"Local batch embedding generation failed: {str(e)}"
                ) from e

        texts = [_truncate_for_embedding(t, embedding_model) for t in texts]

        try:
            params = {"model": embedding_model, "input": texts}
            if dimensions:
                params["dimensions"] = dimensions

            # Resolve and add API key configuration if provided (overrides env vars)
            api_key, api_base, api_version = self._resolve_api_key(
                embedding_model, for_embedding=True
            )
            if api_key:
                params["api_key"] = api_key
            if api_base:
                params["api_base"] = api_base
            if api_version:
                params["api_version"] = api_version

            response = litellm.embedding(
                **params,
                timeout=self.config.timeout,
                num_retries=self.config.max_retries,
            )
            # Response data may not be in order, sort by index to ensure correct ordering
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            raise LiteLLMClientError(
                f"Batch embedding generation failed: {str(e)}"
            ) from e
