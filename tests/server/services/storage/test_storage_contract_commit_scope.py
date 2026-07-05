"""Contract tests for commit_scope atomicity across all storage backends."""

import pytest

from reflexio.models.api_schema.domain.entities import Request
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def _seed_request(
    storage: BaseStorage, request_id: str = "r1", user_id: str = "u1"
) -> None:
    storage.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            created_at=1000,
            source="api",
            agent_version="v1",
            session_id=request_id,
        )
    )


class TestCommitScopeAtomicity:
    def test_exception_in_scope_rolls_back_all_writes(
        self, storage: BaseStorage
    ) -> None:
        with pytest.raises(RuntimeError, match="boom"), storage.commit_scope():
            _seed_request(storage, "r-atomic", "u-atomic")
            raise RuntimeError("boom")
        assert storage.get_request("r-atomic") is None

    def test_clean_scope_persists_all_writes(self, storage: BaseStorage) -> None:
        with storage.commit_scope():
            _seed_request(storage, "r-clean", "u-clean")
        assert storage.get_request("r-clean") is not None
