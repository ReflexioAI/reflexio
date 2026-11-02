"""A real interrupted SQLite window needs its actual owner, without a reaper."""

import pytest

from reflexio.server.services.storage.sqlite_storage._extraction_stream import (
    SQLiteStreamSQL,
)
from reflexio.test_support.owned_window_recovery import (
    assert_running_window_recovery,
    install_crashing_model,
    recovery_publish_request,
)
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline
pytestmark = pytest.mark.integration


@pytest.mark.parametrize("parent_state", ["released", "live", "expired", "replacement"])
def test_running_window_restart_needs_no_age_reaper(
    pipeline, monkeypatch, parent_state
):
    engine, worker, client = pipeline
    storage = engine.get_storage()
    calls = install_crashing_model(monkeypatch)
    assert client.post(
        "/api/publish_interaction", json=recovery_publish_request()
    ).json()["success"]
    assert_running_window_recovery(
        storage, worker, calls, monkeypatch, parent_state, _expire_sqlite_parent
    )


def _expire_sqlite_parent(storage, monkeypatch):
    with storage._stream_sql(read_only=True) as db:
        after_expiry = db.now() + 2
    monkeypatch.setattr(SQLiteStreamSQL, "now", lambda _: after_expiry)
