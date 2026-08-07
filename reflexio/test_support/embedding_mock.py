"""Deterministic embedding stub for reflexio test suites.

Companion to :mod:`llm_mock`, which does the same for ``litellm.completion``.
Patches the shared embedding dispatch so unit tests get a vector without a
running inference service.

Why this exists: embeddings used to be computed in-process during tests, so
storage writes got real vectors for free. That path was removed when local
inference moved behind a service, and nothing replaced it for tests -- the
suite kept passing only because the SQLite ingest path swallows
``EmbeddingUnavailableError`` and stores an empty vector. So the write
succeeded, the row was silently unsearchable, and any assertion about vector
search or clustering downstream of it was passing vacuously.

The vectors are deterministic but **not semantic**: they are derived from a
hash of the text, so identical text embeds identically and different text
embeds differently, and that is all. Do not use them to assert that related
phrasings cluster together -- for that, place vectors explicitly, as
``test_missing_vector_backfill_integration`` does.

Usage in conftest.py::

    from reflexio.test_support.embedding_mock import patch_embeddings

    @pytest.fixture(autouse=True)
    def _deterministic_embeddings():
        with patch_embeddings():
            yield
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

from reflexio.models.config_schema import EMBEDDING_DIMENSIONS

__all__ = ["deterministic_embedding", "patch_embeddings"]


def deterministic_embedding(text: str, dimensions: int | None = None) -> list[float]:
    """Return a stable unit vector derived from ``text``.

    Args:
        text (str): Text to embed.
        dimensions (int | None): Vector length. Defaults to the project-wide
            ``EMBEDDING_DIMENSIONS``, which is what storage validates against.

    Returns:
        list[float]: A unit-norm vector of length ``dimensions``. The same text
            always yields the same vector; different text yields a different
            one.
    """
    dims = dimensions or EMBEDDING_DIMENSIONS
    # Expand the digest by counter until it covers `dims` floats. Hash-derived
    # rather than PRNG-seeded so the vector does not depend on the interpreter's
    # random implementation, which would make fixtures non-portable.
    raw = b""
    counter = 0
    while len(raw) < dims * 4:
        raw += hashlib.sha256(f"{counter}:{text}".encode()).digest()
        counter += 1
    values = [
        # Map each 4-byte word to [-1, 1); the exact distribution does not
        # matter, only that it is stable and spreads distinct texts apart.
        (struct.unpack_from(">I", raw, i * 4)[0] / 0x7FFFFFFF) - 1.0
        for i in range(dims)
    ]
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:  # pragma: no cover - only reachable if every word is 2^31
        return [1.0] + [0.0] * (dims - 1)
    return [value / norm for value in values]


@contextmanager
def patch_embeddings() -> Iterator[None]:
    """Patch the shared embedding dispatch with deterministic vectors.

    Patches ``LiteLLMClient._embed_texts`` -- the single seam both
    ``get_embedding`` and ``get_embeddings`` route through -- so batch and
    single paths cannot diverge.

    A test that installs its own client (``storage.llm_client = MagicMock()``)
    is unaffected: this patches the class method, and such a test never reaches
    it. A test that wants a real failure can still patch the client to raise.
    """
    from reflexio.server.llm.litellm_client import LiteLLMClient

    def _embed_texts(
        _self: LiteLLMClient,
        texts: list[str],
        _model: str | None,
        dimensions: int | None,
        *,
        batch: bool,  # noqa: ARG001 - signature parity with the real method
    ) -> list[list[float]]:
        return [deterministic_embedding(text, dimensions) for text in texts]

    with patch.object(LiteLLMClient, "_embed_texts", _embed_texts):
        yield
