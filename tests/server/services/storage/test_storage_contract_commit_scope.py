"""Contract tests for commit_scope atomicity across all storage backends."""

import sqlite3

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

    def test_scope_depth_reset_after_begin_failure(self, storage: BaseStorage) -> None:
        """If BEGIN IMMEDIATE raises, _scope_depth stays 0 so subsequent scopes work.

        Regression guard: _scope_depth was set to 1 *before* BEGIN IMMEDIATE.
        If BEGIN raised, _scope_depth stayed 1 forever — every subsequent
        commit_scope treated itself as nested and never committed (silent data
        loss).  The fix moves _scope_depth = 1 to *after* a successful BEGIN.

        This invariant is SQLite-specific (_scope_depth lives on SQLiteStorage);
        other backends are skipped.
        """
        if not hasattr(storage, "_scope_depth"):
            pytest.skip("_scope_depth regression is SQLite-specific")

        real_conn = storage.conn  # type: ignore[attr-defined]

        class _FailBeginConn:
            """Proxy that raises OperationalError on any BEGIN statement."""

            def execute(self, sql: str, *a: object, **kw: object) -> object:
                if "BEGIN" in sql.upper():
                    raise sqlite3.OperationalError("database is locked (forced)")
                return real_conn.execute(sql, *a, **kw)

            def __getattr__(self, name: str) -> object:
                return getattr(real_conn, name)

        storage.conn = _FailBeginConn()  # type: ignore[assignment]
        try:
            with (
                pytest.raises(sqlite3.OperationalError, match="database is locked"),
                storage.commit_scope(),
            ):
                pass
        finally:
            storage.conn = real_conn  # type: ignore[assignment]  # restore regardless

        # _scope_depth must be back to 0 — not stuck at 1.
        assert storage._scope_depth == 0  # type: ignore[attr-defined]
        # The connection must still be usable: next scope commits normally.
        with storage.commit_scope():
            _seed_request(storage, "r-after-begin-fail", "u-after-begin-fail")
        assert storage.get_request("r-after-begin-fail") is not None
