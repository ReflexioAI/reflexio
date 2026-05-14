"""Tests for the singleton stall_state row in SQLite storage."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reflexio.server.services.storage.sqlite_storage._stall_state import (
    StallState,
    clear_stall_state,
    get_stall_state,
    mark_stall_notified,
    upsert_stall_state,
)


def test_default_is_clean(storage):
    """A fresh DB returns stalled=False with all fields default-null."""
    state = get_stall_state(storage.connection)
    assert state.stalled is False
    assert state.reason is None
    assert state.notified_in_cc is False


def test_upsert_then_get_roundtrip(storage):
    now = datetime.now(timezone.utc)
    upsert_stall_state(
        storage.connection,
        reason="billing_error",
        stalled_at=now,
        reset_estimate=None,
        error_message="credit exhausted",
    )
    state = get_stall_state(storage.connection)
    assert state.stalled is True
    assert state.reason == "billing_error"
    assert state.notified_in_cc is False
    assert state.error_message == "credit exhausted"


def test_mark_notified_flips_only_that_field(storage):
    upsert_stall_state(storage.connection, reason="auth_error",
                       stalled_at=datetime.now(timezone.utc),
                       reset_estimate=None, error_message="login")
    mark_stall_notified(storage.connection)
    state = get_stall_state(storage.connection)
    assert state.stalled is True
    assert state.notified_in_cc is True


def test_clear_resets_all_fields_and_notification(storage):
    upsert_stall_state(storage.connection, reason="billing_error",
                       stalled_at=datetime.now(timezone.utc),
                       reset_estimate=None, error_message="x")
    mark_stall_notified(storage.connection)
    clear_stall_state(storage.connection)
    state = get_stall_state(storage.connection)
    assert state.stalled is False
    assert state.reason is None
    assert state.notified_in_cc is False
    assert state.error_message is None


def test_upsert_after_clear_resets_notified_flag(storage):
    """New stall must re-arm notified_in_cc=False so SessionStart fires again."""
    now = datetime.now(timezone.utc)
    upsert_stall_state(storage.connection, reason="billing_error", stalled_at=now,
                       reset_estimate=None, error_message="x")
    mark_stall_notified(storage.connection)
    clear_stall_state(storage.connection)
    upsert_stall_state(storage.connection, reason="auth_error", stalled_at=now,
                       reset_estimate=None, error_message="y")
    state = get_stall_state(storage.connection)
    assert state.notified_in_cc is False
