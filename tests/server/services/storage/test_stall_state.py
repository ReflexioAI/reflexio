"""Tests for the singleton stall_state row in SQLite storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from reflexio.server.services.storage.sqlite_storage._stall_state import _parse_ts


def test_parse_ts_treats_offsetless_timestamp_as_utc():
    parsed = _parse_ts("2026-06-10T23:47:07.50016")

    assert parsed == datetime(2026, 6, 10, 23, 47, 7, 500160, tzinfo=UTC)


def test_default_is_clean(storage):
    """A fresh DB returns stalled=False with all fields default-null."""
    state = storage.get_stall_state()
    assert state.stalled is False
    assert state.reason is None
    assert state.notified_in_cc is False


def test_upsert_then_get_roundtrip(storage):
    now = datetime.now(UTC)
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=now,
        reset_estimate=None,
        error_message="credit exhausted",
    )
    state = storage.get_stall_state()
    assert state.stalled is True
    assert state.reason == "billing_error"
    assert state.notified_in_cc is False
    assert state.error_message == "credit exhausted"


def test_reset_estimate_roundtrip(storage):
    """A non-None reset_estimate persists and parses back to the same datetime."""
    now = datetime.now(UTC)
    reset_time = now + timedelta(hours=6)
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=now,
        reset_estimate=reset_time,
        error_message="quota exceeded",
    )
    state = storage.get_stall_state()
    assert state.reset_estimate == reset_time


def test_mark_notified_flips_only_that_field(storage):
    """mark_stall_notified flips only notified_in_cc, leaves other fields intact."""
    storage.upsert_stall_state(
        reason="auth_error",
        stalled_at=datetime.now(UTC),
        reset_estimate=None,
        error_message="login",
    )
    storage.mark_stall_notified()
    state = storage.get_stall_state()
    assert state.stalled is True
    assert state.notified_in_cc is True


def test_clear_resets_all_fields_and_notification(storage):
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=datetime.now(UTC),
        reset_estimate=None,
        error_message="x",
    )
    storage.mark_stall_notified()
    storage.clear_stall_state()
    state = storage.get_stall_state()
    assert state.stalled is False
    assert state.reason is None
    assert state.notified_in_cc is False
    assert state.error_message is None


def test_upsert_after_clear_resets_notified_flag(storage):
    """New stall must re-arm notified_in_cc=False so SessionStart fires again."""
    now = datetime.now(UTC)
    storage.upsert_stall_state(
        reason="billing_error",
        stalled_at=now,
        reset_estimate=None,
        error_message="x",
    )
    storage.mark_stall_notified()
    storage.clear_stall_state()
    storage.upsert_stall_state(
        reason="auth_error",
        stalled_at=now,
        reset_estimate=None,
        error_message="y",
    )
    state = storage.get_stall_state()
    assert state.notified_in_cc is False
