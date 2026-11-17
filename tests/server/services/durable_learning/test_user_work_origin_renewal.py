"""Heartbeat context and optional native renewal fencing, without providers."""

from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, get_ident
from typing import cast

import pytest

from reflexio.server.services.durable_learning import user_lease
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.services.storage.storage_base._extraction_stream import (
    ExtractionStreamStore,
)


def test_actual_heartbeat_thread_preserves_context_and_storage_identity(monkeypatch):
    origin: ContextVar[object | None] = ContextVar("test_native_origin", default=None)
    expected = object()
    done = Event()
    observed = []
    caller = get_ident()

    class Storage:
        org_id = "org"

        def renew_extraction(self, user_id, token, seconds):
            observed.append((self, origin.get(), get_ident(), user_id, token, seconds))
            # A change belongs to the copied thread context, not its parent.
            origin.set("heartbeat-only")
            done.set()
            return False

    storage = Storage()
    monkeypatch.setattr(
        user_lease.durable_learning_policy, "renewal_interval_seconds", lambda _: 0.001
    )
    binding = origin.set(expected)
    try:
        with user_lease.owned_user_work_context(
            cast(BaseStorage, storage), "user", "token", 60
        ) as owner:
            assert owner.storage is storage
            assert done.wait(2), "The actual heartbeat thread did not run"
        assert origin.get() is expected
    finally:
        origin.reset(binding)
    assert len(observed) == 1
    real_storage, inherited, thread, *arguments = observed[0]
    assert real_storage is storage
    assert inherited is expected
    assert thread != caller
    assert arguments == ["user", "token", 60]


class _SQL:
    """Observe the shared renewal algorithm's transaction/statement ordering."""

    def __init__(self):
        self.events = []
        self.lease_until = 200.0
        self.live = True
        self.committed = False
        self.rolled_back = False

    def table(self, name):
        assert name == "learning_work"
        return name

    def lock(self):
        return " FOR UPDATE"

    def now(self):
        self.events.append("clock")
        return 100.0

    def query(self, sql, params=()):
        if sql.startswith("SELECT lease_until"):
            assert sql.endswith(" FOR UPDATE")
            assert params == ("org", "user", "token")
            self.events.append("W-row-lock")
            return [{"lease_until": self.lease_until}] if self.live else []
        assert sql.startswith("UPDATE learning_work SET lease_until=")
        assert params == (160.0, "org", "user", "token")
        self.events.append("lease-update")
        self.lease_until = params[0]
        return [{"user_id": "user"}]


class _FencedSQL(_SQL):
    def __init__(self, *, valid=True):
        super().__init__()
        self.valid = valid

    def require_learning_operation_renewal_origin(self):
        # Native adapters own root validation. The SDK must call this on the
        # same writer after W admission and before its unchanged lease UPDATE.
        assert self.events == ["W-row-lock", "clock"]
        self.events.append("native-origin-fence")
        if not self.valid:
            raise ValueError("origin_changed")


class _Storage(ExtractionStreamStore):
    org_id = "org"

    def __init__(self, db):
        self.db = db

    @contextmanager
    def _stream_sql(self, *, coordination=False, **_kwargs):
        assert coordination is True
        previous = self.db.lease_until
        try:
            yield self.db
            self.db.events.append("commit")
            self.db.committed = True
        except BaseException:
            self.db.lease_until = previous
            self.db.events.append("rollback")
            self.db.rolled_back = True
            raise


def test_native_origin_guard_runs_after_w_lock_before_same_writer_update():
    db = _FencedSQL()
    assert _Storage(db).renew_extraction("user", "token", 60)
    assert db.events == [
        "W-row-lock",
        "clock",
        "native-origin-fence",
        "lease-update",
        "commit",
    ]
    assert db.committed
    assert db.lease_until == 160


def test_native_origin_failure_prevents_renewal_and_propagates_without_commit():
    db = _FencedSQL(valid=False)
    with pytest.raises(ValueError, match="origin_changed"):
        _Storage(db).renew_extraction("user", "token", 60)
    assert db.events == ["W-row-lock", "clock", "native-origin-fence", "rollback"]
    assert db.rolled_back and not db.committed
    assert db.lease_until == 200


def test_ordinary_adapter_without_origin_hook_keeps_legacy_renewal():
    db = _SQL()
    assert _Storage(db).renew_extraction("user", "token", 60)
    assert db.events == ["W-row-lock", "clock", "lease-update", "commit"]


@pytest.mark.parametrize("missing", [False, True])
def test_expired_or_missing_lease_does_not_invoke_guard_or_update(missing):
    db = _FencedSQL()
    db.live = not missing
    db.lease_until = 99
    assert not _Storage(db).renew_extraction("user", "token", 60)
    assert "native-origin-fence" not in db.events
    assert "lease-update" not in db.events


def test_broken_native_hook_does_not_silently_authorize_renewal():
    class BrokenNativeSQL(_SQL):
        require_learning_operation_renewal_origin = object()

    db = BrokenNativeSQL()
    with pytest.raises(TypeError):
        _Storage(db).renew_extraction("user", "token", 60)
    assert "lease-update" not in db.events
    assert db.rolled_back and not db.committed
