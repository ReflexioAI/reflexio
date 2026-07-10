"""Contract: iter_interactions_missing_vectors across storage backends.

Any backend that implements missing-vector detection must surface interactions
whose embedding was never persisted (empty embedding column / no vector row) and
exclude interactions that already have a vector — bounded by ``limit``. The
safe base default returns ``[]`` (covered in the integration suite); this
contract runs against every locally-testable backend via the shared fixture.
"""

from __future__ import annotations

import time

import pytest

from reflexio.models.api_schema.service_schemas import Interaction
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _seed(storage: BaseStorage, user_id: str, iid: int, *, embedding: list[float]):
    interaction = Interaction(
        interaction_id=iid,
        user_id=user_id,
        request_id=f"req-{iid}",
        content=f"content {iid}",
        created_at=int(time.time()) + iid,
        embedding=embedding,
    )
    # embeddings_prepared=True writes the provided embedding verbatim (no LLM call),
    # so an empty list reproduces the degraded-ingest state deterministically.
    storage.add_user_interactions_bulk(
        user_id=user_id, interactions=[interaction], embeddings_prepared=True
    )


class TestIterInteractionsMissingVectors:
    def test_detects_only_the_vectorless_rows(self, storage: BaseStorage) -> None:
        uid = "u-missing"
        _seed(storage, uid, 1, embedding=[])  # missing -> detected
        _seed(storage, uid, 2, embedding=[0.1] * 512)  # has vector -> excluded
        _seed(storage, uid, 3, embedding=[])  # missing -> detected

        found = dict(storage.iter_interactions_missing_vectors(100))

        assert set(found) == {1, 3}
        # The returned text is the ingest-path derivation (content + action desc).
        assert found[1] == "content 1\n"
        assert found[3] == "content 3\n"

    def test_respects_the_limit(self, storage: BaseStorage) -> None:
        uid = "u-limit"
        for iid in range(1, 6):
            _seed(storage, uid, iid, embedding=[])

        assert len(storage.iter_interactions_missing_vectors(2)) == 2
        assert len(storage.iter_interactions_missing_vectors(100)) == 5

    def test_empty_when_nothing_missing(self, storage: BaseStorage) -> None:
        _seed(storage, "u-none", 1, embedding=[0.2] * 512)

        assert storage.iter_interactions_missing_vectors(100) == []

    def test_non_positive_limit_returns_empty(self, storage: BaseStorage) -> None:
        _seed(storage, "u-zero", 1, embedding=[])

        assert storage.iter_interactions_missing_vectors(0) == []
