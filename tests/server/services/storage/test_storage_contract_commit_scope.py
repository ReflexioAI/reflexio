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

    def test_standalone_write_succeeds_after_failed_scope(
        self, storage: BaseStorage
    ) -> None:
        """After a rolled-back commit_scope, no implicit txn must remain open.

        Regression guard for the Important-1 TOCTOU fix: if own_txn is read
        outside the lock, a stale False value skips BEGIN IMMEDIATE on the next
        standalone call, leaving an implicit transaction open that makes the
        following BEGIN IMMEDIATE raise OperationalError.
        """
        with pytest.raises(RuntimeError), storage.commit_scope():
            _seed_request(storage, "r-fail", "u-fail")
            raise RuntimeError("trigger rollback")
        # The rolled-back request must not be visible.
        assert storage.get_request("r-fail") is None
        # A standalone write after the failed scope must succeed without error.
        _seed_request(storage, "r-after", "u-after")
        assert storage.get_request("r-after") is not None
