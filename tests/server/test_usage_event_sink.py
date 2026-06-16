"""Tests for the SQLite usage-event sink (per-entity observability)."""

from typing import Any
from unittest.mock import MagicMock

from reflexio.server.services.usage_event_sink import SqliteUsageEventSink
from reflexio.server.usage_metrics import UsageEvent


def _make_event(
    org_id: str = "test-org",
    event_name: str = "learning_injection",
    event_category: str = "application",
    **overrides: Any,
) -> UsageEvent:
    """Build a ``UsageEvent`` with sensible defaults; kwargs override fields."""
    return UsageEvent(
        org_id=org_id,
        event_name=event_name,
        event_category=event_category,
        user_id=overrides.get("user_id", "u1"),
        request_id=overrides.get("request_id", "r1"),
        session_id=overrides.get("session_id", "s1"),
        pipeline=overrides.get("pipeline", "unified_search"),
        entity_type=overrides.get("entity_type", "playbook"),
        entity_id=overrides.get("entity_id", "42"),
        caller_type=overrides.get("caller_type", "production_agent"),
        count_value=overrides.get("count_value", 1),
        prompt_tokens=overrides.get("prompt_tokens", 10),
        completion_tokens=overrides.get("completion_tokens"),
        billing_input_tokens=overrides.get("billing_input_tokens"),
        platform_llm=overrides.get("platform_llm"),
        platform_storage=overrides.get("platform_storage"),
        duration_ms=overrides.get("duration_ms"),
        error_kind=overrides.get("error_kind"),
        metadata=overrides.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_sink_forwards_event_to_storage_record_usage_event():
    """The sink calls ``storage.record_usage_event(**kwargs)`` with the event fields."""
    storage = MagicMock()
    sink = SqliteUsageEventSink(lambda _: storage)
    event = _make_event(
        entity_type="playbook",
        entity_id="42",
        prompt_tokens=12,
        metadata={"hook_event_name": "UserPromptSubmit"},
    )
    sink(event)
    storage.record_usage_event.assert_called_once()
    kwargs = storage.record_usage_event.call_args.kwargs
    assert kwargs["org_id"] == "test-org"
    assert kwargs["event_name"] == "learning_injection"
    assert kwargs["event_category"] == "application"
    assert kwargs["entity_type"] == "playbook"
    assert kwargs["entity_id"] == "42"
    assert kwargs["prompt_tokens"] == 12
    assert kwargs["caller_type"] == "production_agent"
    assert kwargs["metadata"] == {"hook_event_name": "UserPromptSubmit"}


def test_sink_resolves_storage_lazily():
    """The storage resolver is called on the first event (not at construction)."""
    storage = MagicMock()
    resolver = MagicMock(return_value=storage)
    sink = SqliteUsageEventSink(resolver)
    # No resolver call yet.
    assert resolver.call_count == 0
    sink(_make_event())
    assert resolver.call_count == 1
    sink(_make_event())
    # Subsequent events still go through the resolver (caller may want to
    # re-resolve on each event, e.g. to pick up org switches).
    assert resolver.call_count == 2


# ---------------------------------------------------------------------------
# Resiliency
# ---------------------------------------------------------------------------


def test_sink_silently_drops_when_storage_resolver_returns_none():
    """``None`` storage is the convention for "no storage configured"."""
    sink = SqliteUsageEventSink(lambda _: None)
    # Must not raise.
    sink(_make_event())


def test_sink_swallows_storage_resolver_exception():
    """A broken storage resolver must not break the caller's hot path."""
    def broken_resolver(org_id: str) -> Any:
        raise RuntimeError("storage backend unreachable")
    sink = SqliteUsageEventSink(broken_resolver)
    # Must not raise.
    sink(_make_event())


def test_sink_swallows_storage_write_exception():
    """A storage write failure must not break the caller's hot path."""
    storage = MagicMock()
    storage.record_usage_event.side_effect = RuntimeError("database is locked")
    sink = SqliteUsageEventSink(lambda _: storage)
    # Must not raise.
    sink(_make_event())


def test_sink_serialises_non_jsonable_metadata_to_strings():
    """Metadata values that aren't JSON-serialisable are coerced to strings."""
    storage = MagicMock()
    sink = SqliteUsageEventSink(lambda _: storage)
    event = _make_event(metadata={"hook_event_name": "UserPromptSubmit", "raw": object()})
    sink(event)
    kwargs = storage.record_usage_event.call_args.kwargs
    # The non-serialisable value is coerced to its ``str(...)`` form.
    assert isinstance(kwargs["metadata"]["raw"], str)
    # JSON-serialisable values pass through unchanged.
    assert kwargs["metadata"]["hook_event_name"] == "UserPromptSubmit"


def test_sink_handles_empty_metadata():
    """``metadata=None`` and ``metadata={}`` both produce an empty dict at the storage layer."""
    storage = MagicMock()
    sink = SqliteUsageEventSink(lambda _: storage)
    sink(_make_event(metadata=None))
    assert storage.record_usage_event.call_args.kwargs["metadata"] == {}
    storage.reset_mock()
    sink(_make_event(metadata={}))
    assert storage.record_usage_event.call_args.kwargs["metadata"] == {}


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------


def test_sink_returns_none_on_call():
    """The sink returns ``None`` (it is a fire-and-forget adapter)."""
    storage = MagicMock()
    sink = SqliteUsageEventSink(lambda _: storage)
    assert sink(_make_event()) is None
