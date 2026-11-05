"""Projection cannot erase durable metadata or bypass native embedding failures."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from reflexio.server.services.profile.learning.semantic_embedding import (
    AUDIT_FEATURE_KEYS,
    SemanticEmbeddingStorage,
)


@dataclass
class Profile:
    content: str
    custom_features: dict | None
    expiration_timestamp: int = 1234
    embedding: list = field(default_factory=list)
    expanded_terms: str | None = "supplied terms"

    def model_copy(self, *, deep):
        assert deep
        return deepcopy(self)


def test_semantic_features_survive_and_only_embedding_results_return():
    features = {k: {"nested": "audit"} for k in AUDIT_FEATURE_KEYS}
    features["user_conditions"] = {"project": "alpha", "scope": "historical"}
    p = Profile("Original qualified fact", features)
    before = deepcopy(p)
    seen = []

    def precompute(rows, *, expand_documents):
        assert expand_documents is False
        seen.extend(rows)
        assert rows[0].content == p.content
        assert rows[0].custom_features == {
            "user_conditions": features["user_conditions"]
        }
        rows[0].embedding = [0.25, 0.75]
        rows[0].custom_features["user_conditions"]["project"] = "copy only"
        rows[0].content = "copy only"
        rows[0].expiration_timestamp = 9999
        rows[0].expanded_terms = "native result"

    storage = SimpleNamespace(precompute_profile_embeddings=precompute)
    SemanticEmbeddingStorage(storage).precompute_profile_embeddings([p])
    assert p.custom_features == before.custom_features
    assert p.content == before.content
    assert p.expiration_timestamp == before.expiration_timestamp
    assert p.embedding == [0.25, 0.75]
    assert p.expanded_terms == "native result"
    assert seen[0] is not p


def test_native_failure_does_not_partially_mutate_originals():
    rows = [Profile("one", {"rsi_clauses_v14": [1]}), Profile("two", None)]
    before = deepcopy(rows)

    def fail(projected, *, expand_documents):
        assert expand_documents is False
        projected[0].embedding = [1.0]
        projected[0].expanded_terms = "partial result"
        raise RuntimeError("native embedding failure")

    with pytest.raises(RuntimeError, match="native embedding failure"):
        SemanticEmbeddingStorage(
            SimpleNamespace(precompute_profile_embeddings=fail)
        ).precompute_profile_embeddings(rows)
    assert rows == before
