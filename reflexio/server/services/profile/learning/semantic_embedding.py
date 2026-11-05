"""Candidate-owned embedding projection; original profiles retain their audit state."""

from __future__ import annotations

from typing import Any

# Exact code-owned machine fields only. Unknown/custom user fields stay semantic.
AUDIT_FEATURE_KEYS = frozenset(
    {
        "rsi_clauses_v14",
        "rsi_lifetime_v8",
        "rsi_lifetime_v7",
        "rsi_lifetime_v6",
        "rsi_deadline_ancestors_v8",
    }
)


class SemanticEmbeddingStorage:
    """Project only the compute-side input before native embedding computation.

    Native precomputation owns model policy, admission and embedding execution.
    No copies enter persistence. Mutate original result fields only after the
    complete native computation succeeds; leave all originals intact on failure.
    """

    def __init__(self, storage: Any) -> None:
        self._storage = storage

    def precompute_profile_embeddings(self, profiles: list[Any]) -> None:
        projected = []
        for profile in profiles:
            row = profile.model_copy(deep=True)
            features = {
                key: value
                for key, value in (row.custom_features or {}).items()
                if key not in AUDIT_FEATURE_KEYS
            }
            row.custom_features = features or None
            projected.append(row)
        self._storage.precompute_profile_embeddings(projected, expand_documents=False)
        for original, row in zip(profiles, projected, strict=True):
            original.embedding = row.embedding
            original.expanded_terms = row.expanded_terms
