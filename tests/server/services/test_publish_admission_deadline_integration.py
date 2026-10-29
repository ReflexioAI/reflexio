"""A publish that reaches its admission deadline is a retryable capacity refusal.

`check_admission_deadline` runs only before the admission transaction commits,
so a publish that trips it admitted nothing. Real SQLite storage, so "nothing
was admitted" is read back from the database rather than asserted on a mock.
"""

from __future__ import annotations

import datetime
import logging
import tempfile
import time
from datetime import UTC

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.server.services.durable_learning.waiting import (
    AdmissionDeadlineExceededError,
    admission_deadline,
)


@pytest.fixture(autouse=True)
def stop_background_scheduler(monkeypatch):
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )


def _payload(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "user_id": "deadline-user",
        "session_id": "deadline-session",
        "interaction_data_list": [
            {
                "content": "hello",
                "created_at": int(datetime.datetime.now(UTC).timestamp()),
            }
        ],
    }


def _publish_with_deadline_inside_commit_scope(reflexio: Reflexio, request_id: str):
    """Expire the deadline once the publish stream lock is held.

    That is the LAST check before the writes, inside `commit_scope`, so it is
    the case where "nothing was admitted" depends on the scope rolling back.
    """
    storage = reflexio.get_storage()
    original_lock = storage.lock_publish_stream

    def lock_then_expire(*args, **kwargs):
        original_lock(*args, **kwargs)
        admission_deadline.set(time.monotonic() - 1)

    storage.lock_publish_stream = lock_then_expire  # type: ignore[method-assign]
    token = admission_deadline.set(time.monotonic() + 60)
    try:
        return reflexio.publish_interaction(_payload(request_id), defer_learning=True)
    finally:
        admission_deadline.reset(token)
        storage.lock_publish_stream = original_lock  # type: ignore[method-assign]


def test_deadline_propagates_as_capacity_with_nothing_admitted(caplog, worker_id):
    with tempfile.TemporaryDirectory() as tmp_dir:
        reflexio = Reflexio(org_id=f"deadline-{worker_id}", storage_base_dir=tmp_dir)
        caplog.set_level(logging.WARNING)

        with pytest.raises(AdmissionDeadlineExceededError):
            _publish_with_deadline_inside_commit_scope(reflexio, "deadline-req")

        storage = reflexio.get_storage()
        assert storage.get_request("deadline-req") is None
        assert storage.claim_extraction("deadline-worker", 300) is None
        assert any(
            "publish_admission_deadline_exceeded" in r.getMessage()
            and r.levelno == logging.WARNING
            for r in caplog.records
        ), caplog.text
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR], (
            caplog.text
        )


def test_other_publish_failures_still_log_error(caplog, worker_id):
    """The capacity carve-out is by type: a genuine failure stays an ERROR."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        reflexio = Reflexio(org_id=f"failure-{worker_id}", storage_base_dir=tmp_dir)
        storage = reflexio.get_storage()

        def boom(*_args, **_kwargs):
            raise TimeoutError("driver socket timeout")

        storage.lock_publish_stream = boom  # type: ignore[method-assign]
        caplog.set_level(logging.WARNING)

        response = reflexio.publish_interaction(
            _payload("failure-req"), defer_learning=True
        )

        assert response.success is False
        assert any(
            r.levelno == logging.ERROR
            and "Failed to refresh user profile" in r.getMessage()
            for r in caplog.records
        ), caplog.text
