"""
Embedding cache + deterministic fake embedder for retrieval latency benchmarks.

This module controls the cost of embedding generation during a benchmark run.
Retrieval latency measurement should not be dominated by embedder API calls,
but query-side vectors must still be realistic so the vector-search path
computes meaningful cosine similarities.

Two helpers are provided:

1. :class:`QueryEmbedCache` — on-disk cache of real embeddings for the fixed
   query set. Populated once via :class:`LiteLLMClient`, then reused on every
   subsequent run. First-time population requires a configured embedding API
   key (typically ``OPENAI_API_KEY``).
2. :func:`make_fake_document_embedder` — deterministic pseudo-random vector
   generator used during corpus seeding. Document vector *content* does not
   affect query-time latency (only row count does), so fake vectors are a
   sound optimization for a latency benchmark.

Both helpers expose a :func:`patch_storage_get_embedding` style context
manager that swaps out the storage instance's bound ``_get_embedding`` method
for the duration of a ``with`` block.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

logger = logging.getLogger(__name__)

# Track the storage-level canonical dimension so fake document vectors
# always fit the vec0 column width. Query cache entries are populated
# through LiteLLMClient at this same dimension for consistency.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMS = EMBEDDING_DIMENSIONS

_CACHE_DIR = Path.home() / ".cache" / "reflexio-benchmarks"
_DOCUMENT_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "


EmbeddingPurpose = Literal["document", "query"]

EmbeddingFn = Callable[[str, EmbeddingPurpose], list[float]]


def _cache_path(model: str) -> Path:
    """
    Return the on-disk cache path for embeddings of a given model.

    Args:
        model (str): Embedding model identifier (e.g. text-embedding-3-small).

    Returns:
        Path: JSON cache file path. Parent directory is not created here.
    """
    safe = model.replace("/", "_")
    return _CACHE_DIR / f"embeddings-{safe}.json"


def _hash_key(text: str) -> str:
    """
    Return a stable short hash key for a full (prefixed) embedding input.

    Args:
        text (str): The full text that will be passed to the embedding model.

    Returns:
        str: Hex digest suitable for use as a JSON dict key.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class QueryEmbedCache:
    """
    On-disk cache of real embedding vectors for the benchmark query set.

    The cache file is a flat JSON dict mapping SHA256(prefixed_text) → vector.
    Population happens lazily via :meth:`ensure` and only touches the network
    when the cache is missing an entry.

    Attributes:
        model (str): Embedding model used to populate the cache.
        dimensions (int): Expected vector dimensionality.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMS,
    ) -> None:
        """
        Initialize the cache, loading any existing on-disk entries.

        Args:
            model (str): Embedding model name. Must match whatever the live
                storage backend is configured with.
            dimensions (int): Expected embedding dimensionality. Must match
                the storage vector column width.
        """
        self.model = model
        self.dimensions = dimensions
        self._path = _cache_path(model)
        self._data: dict[str, list[float]] = {}
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text())
            except (OSError, json.JSONDecodeError) as err:
                logger.warning(
                    "Failed to load embed cache at %s: %s — starting fresh",
                    self._path,
                    err,
                )
                self._data = {}

    def ensure(self, queries: Iterable[str]) -> None:
        """
        Make sure every query has a real embedding in the cache.

        Populates missing entries via the live embedding API in a single
        batched call. Writes the cache to disk at the end of population so
        subsequent runs skip the network entirely.

        Args:
            queries (Iterable[str]): Raw query strings (no prefix). The
                ``search_query:`` prefix is applied internally.

        Raises:
            RuntimeError: If the embedder cannot be reached and the cache is
                not already populated for every query.
        """
        missing: list[str] = []
        missing_prefixed: list[str] = []
        for q in queries:
            prefixed = _QUERY_PREFIX + q
            if _hash_key(prefixed) not in self._data:
                missing.append(q)
                missing_prefixed.append(prefixed)
        if not missing:
            return

        logger.info(
            "Embed cache miss for %d queries (model=%s) — fetching",
            len(missing),
            self.model,
        )
        client = LiteLLMClient(LiteLLMConfig(model=self.model))
        try:
            vectors = client.get_embeddings(
                missing_prefixed, self.model, self.dimensions
            )
        except Exception as err:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to populate query embed cache for model {self.model}: "
                f"{err}. Set OPENAI_API_KEY (or the appropriate provider key) "
                "and re-run."
            ) from err

        for prefixed, vec in zip(missing_prefixed, vectors, strict=True):
            if len(vec) != self.dimensions:
                raise RuntimeError(
                    f"Embed cache got vector of length {len(vec)}, expected "
                    f"{self.dimensions} for model {self.model}"
                )
            self._data[_hash_key(prefixed)] = list(vec)

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data))

    def lookup(self, text: str, purpose: EmbeddingPurpose) -> list[float]:
        """
        Return a cached embedding for the given raw text.

        Args:
            text (str): Raw text (no prefix).
            purpose (EmbeddingPurpose): ``"document"`` or ``"query"``.

        Returns:
            list[float]: The cached embedding vector.

        Raises:
            KeyError: If the text is not in the cache.
        """
        prefix = _DOCUMENT_PREFIX if purpose == "document" else _QUERY_PREFIX
        key = _hash_key(prefix + text)
        try:
            return self._data[key]
        except KeyError as err:
            raise KeyError(
                f"Query embed cache miss for text={text!r} purpose={purpose}. "
                "Did you forget to call QueryEmbedCache.ensure() for the query set?"
            ) from err


def _deterministic_vector(text: str, dimensions: int) -> list[float]:
    """
    Generate a deterministic pseudo-random unit vector from text.

    Uses SHA256 as a cheap keyed PRNG, converts the digest stream into
    little-endian float32 values, then normalizes to unit length. The
    output is stable across runs and machines.

    Args:
        text (str): Input text (the hash seed).
        dimensions (int): Desired vector length.

    Returns:
        list[float]: Unit-length vector of ``dimensions`` floats.
    """
    # Derive 4*dimensions pseudo-random bytes from chained SHA256 of
    # (text, counter). This is not cryptographic — it just needs to be
    # stable and unbiased enough to exercise cosine similarity paths.
    need = dimensions * 4
    buf = bytearray()
    counter = 0
    while len(buf) < need:
        h = hashlib.sha256(f"{text}:{counter}".encode()).digest()
        buf.extend(h)
        counter += 1
    floats = [
        struct.unpack("<i", bytes(buf[i * 4 : (i + 1) * 4]))[0] / 2**31
        for i in range(dimensions)
    ]
    norm = sum(x * x for x in floats) ** 0.5
    if norm == 0.0:
        return [0.0] * dimensions
    return [x / norm for x in floats]


def make_fake_document_embedder(
    dimensions: int = DEFAULT_EMBEDDING_DIMS,
) -> Callable[[str, EmbeddingPurpose], list[float]]:
    """
    Build a fake ``_get_embedding`` replacement for corpus seeding.

    Returns deterministic hash-derived vectors so seeding is fast (no API
    calls) while preserving vector column width and cosine-similarity
    determinism across runs. Only valid for seeding — query-time retrieval
    should use a real :class:`QueryEmbedCache` so BM25/vector ranking is
    meaningful.

    Args:
        dimensions (int): Vector length to produce. Must match the storage
            backend's vector column width.

    Returns:
        Callable: A bound-method-compatible ``(text, purpose) → vector``
            function ready to patch onto a storage instance.
    """

    def fake(text: str, purpose: EmbeddingPurpose = "document") -> list[float]:
        # Include purpose in the seed so doc and query versions of the same
        # text get different vectors — mirrors how the real embedder treats
        # the search_document/search_query prefixes as distinct inputs.
        return _deterministic_vector(f"{purpose}:{text}", dimensions)

    return fake


def make_cached_query_embedder(
    cache: QueryEmbedCache,
    fallback_dims: int = DEFAULT_EMBEDDING_DIMS,
) -> Callable[[str, EmbeddingPurpose], list[float]]:
    """
    Build a ``_get_embedding`` replacement backed by :class:`QueryEmbedCache`.

    Document lookups fall back to fake deterministic vectors (seeding already
    happened; any ``purpose='document'`` call during the timed loop would be
    an oversight but must not crash). Query lookups raise if the cache is
    empty — that indicates :meth:`QueryEmbedCache.ensure` was skipped.

    Args:
        cache (QueryEmbedCache): Populated cache for the benchmark query set.
        fallback_dims (int): Vector width for any document-side lookups that
            leak into the timed loop.

    Returns:
        Callable: A bound-method-compatible ``(text, purpose) → vector`` fn.
    """

    def cached(text: str, purpose: EmbeddingPurpose = "document") -> list[float]:
        if purpose == "query":
            return cache.lookup(text, "query")
        return _deterministic_vector(f"document:{text}", fallback_dims)

    return cached


@contextmanager
def patch_storage_get_embedding(
    storage: Any,
    replacement: EmbeddingFn,
) -> Iterator[None]:
    """
    Temporarily replace a storage instance's ``_get_embedding`` bound method.

    Used to swap in either the fake seeding embedder or the cached query
    embedder for the duration of a benchmark phase.

    ``storage`` is typed ``Any`` because ``_get_embedding`` is declared on the
    sqlite storage mixins rather than on ``BaseStorage`` itself — a Protocol
    that required the attribute would reject real storage instances.

    Args:
        storage (Any): A BaseStorage-like instance exposing a
            ``_get_embedding`` method. Writing to ``storage._get_embedding``
            goes into the instance dict, so subsequent lookups resolve to the
            plain replacement function (no auto-binding).
        replacement (EmbeddingFn): The ``(text, purpose) → vector`` fn to bind.

    Yields:
        None: During the ``with`` block, ``storage._get_embedding`` routes to
        ``replacement``. On exit, the original bound method is restored.
    """
    original = storage._get_embedding
    storage._get_embedding = replacement
    try:
        yield
    finally:
        storage._get_embedding = original
