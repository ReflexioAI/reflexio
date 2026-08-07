"""Guard that the autouse embedding fake is actually in effect.

Without this, the fixture is unfalsifiable. Removing it breaks nothing visible:
the ingest path swallows ``EmbeddingUnavailableError`` and stores an empty
vector, so every other test still passes -- writing rows that are silently
unsearchable. That is the exact state this fixture exists to end, and it is
invisible unless something asserts on the stored vector.

Verified by mutation: disabling the fixture leaves ``test_sqlite_storage``'s 80
tests green, and fails only this module.
"""

from __future__ import annotations

from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.test_support.embedding_mock import deterministic_embedding


def _client() -> LiteLLMClient:
    return LiteLLMClient(LiteLLMConfig(model="test-model"))


def test_embedding_calls_return_a_real_vector_without_a_service() -> None:
    """A plain embed call must yield a usable vector, not [] and not an error.

    No inference service runs in the unit tier, so before the fixture this
    raised EmbeddingUnavailableError at the client and degraded to [] at the
    storage layer.
    """
    vector = _client().get_embedding("a stored document")

    assert vector, "embedding must not be empty -- the fake is not in effect"
    assert len(vector) == EMBEDDING_DIMENSIONS


def test_batch_and_single_paths_agree() -> None:
    """Both public entry points route through one patched seam.

    Patching them separately would let batch and single drift, which is the
    defect the shared ``_embed_texts`` seam exists to prevent.
    """
    client = _client()
    texts = ["alpha", "beta"]

    assert client.get_embeddings(texts) == [client.get_embedding(t) for t in texts]


def test_same_text_embeds_identically_and_different_text_does_not() -> None:
    """Determinism is what makes fixtures reproducible across runs and workers.

    Asserted together with distinctness because a constant vector would satisfy
    determinism alone while making every row identical -- which would quietly
    ruin any clustering assertion built on top.
    """
    client = _client()

    assert client.get_embedding("same") == client.get_embedding("same")
    assert client.get_embedding("same") != client.get_embedding("different")


def test_vectors_are_not_semantic() -> None:
    """Pin the documented limitation so nobody builds similarity tests on it.

    The vectors are hash-derived, so related phrasings are no closer than
    unrelated ones. A test that needs semantic structure must place vectors
    explicitly rather than rely on this fixture.
    """
    near = deterministic_embedding("cancel my order")
    paraphrase = deterministic_embedding("cancel my order please")
    unrelated = deterministic_embedding("what is the weather")

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    # Both are near-orthogonal; the paraphrase enjoys no advantage.
    assert abs(cosine(near, paraphrase)) < 0.5
    assert abs(cosine(near, unrelated)) < 0.5
