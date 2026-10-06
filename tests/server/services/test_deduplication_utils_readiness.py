"""resolve_dedup_query_embeddings: degrade on failure, stop on "not ready yet"."""

from unittest.mock import MagicMock

import pytest

from reflexio.server.services.deduplication_utils import resolve_dedup_query_embeddings
from reflexio.server.services.storage.error import ReadinessUnavailableError


def _storage(error: Exception) -> MagicMock:
    storage = MagicMock()
    storage.embedding_model_name = "model"
    storage._get_embedding.side_effect = error
    return storage


def test_an_embedding_failure_degrades_to_no_vectors():
    storage = _storage(RuntimeError("embedding service down"))

    result = resolve_dedup_query_embeddings(
        storage, MagicMock(), ["a", "b"], entity_label="Playbook"
    )

    assert result == [None, None]


def test_not_ready_yet_stops_the_batch():
    storage = _storage(ReadinessUnavailableError("embedding_readiness_unavailable"))

    with pytest.raises(ReadinessUnavailableError):
        resolve_dedup_query_embeddings(
            storage, MagicMock(), ["a"], entity_label="Playbook"
        )
