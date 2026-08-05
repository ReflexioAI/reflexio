import logging

import pytest

from reflexio.server import usage_metrics
from reflexio.server.usage_metrics import (
    UsageEvent,
    UsageEventDeliveryError,
    UsageEventDeliveryStatus,
)


def test_usage_event_carries_event_key():
    captured = []
    usage_metrics.configure_usage_event_recorder(captured.append)
    try:
        usage_metrics.record_usage_event(
            org_id="7",
            event_name="search_request",
            event_category="application",
            count_value=1,
            event_key="search:abc",
        )
    finally:
        usage_metrics.configure_usage_event_recorder(None)
    assert captured and captured[0].event_key == "search:abc"


def test_ordinary_delivery_is_silent_without_recorder(caplog):
    usage_metrics.configure_usage_event_recorder(None)

    with caplog.at_level(logging.WARNING, logger=usage_metrics.__name__):
        usage_metrics.record_usage_event(
            org_id="7",
            event_name="search_request",
            event_category="application",
        )

    assert caplog.records == []


def test_ordinary_delivery_is_silent_for_legacy_recorder(caplog):
    captured = []
    usage_metrics.configure_usage_event_recorder(captured.append)
    try:
        with caplog.at_level(logging.WARNING, logger=usage_metrics.__name__):
            usage_metrics.record_usage_event(
                org_id="7",
                event_name="search_request",
                event_category="application",
            )
    finally:
        usage_metrics.configure_usage_event_recorder(None)

    assert len(captured) == 1
    assert caplog.records == []


def test_event_key_defaults_none():
    assert UsageEvent(org_id="7", event_name="x", event_category="y").event_key is None


def test_strict_delivery_rejects_legacy_none_recorder_as_unknown():
    captured = []
    usage_metrics.configure_usage_event_recorder(captured.append)
    try:
        with pytest.raises(UsageEventDeliveryError) as exc_info:
            usage_metrics.record_usage_event_strict(
                org_id="7",
                event_name="learnings_generated",
                event_category="learning",
                event_key="learn:profile:1",
            )
    finally:
        usage_metrics.configure_usage_event_recorder(None)

    assert exc_info.value.status is UsageEventDeliveryStatus.UNKNOWN
    assert [event.event_key for event in captured] == ["learn:profile:1"]


def test_strict_delivery_rejects_missing_recorder_as_unknown():
    usage_metrics.configure_usage_event_recorder(None)

    with pytest.raises(UsageEventDeliveryError) as exc_info:
        usage_metrics.record_usage_event_strict(
            org_id="7",
            event_name="learnings_generated",
            event_category="learning",
            event_key="learn:profile:1",
        )

    assert exc_info.value.status is UsageEventDeliveryStatus.UNKNOWN


def test_strict_delivery_accepts_explicit_deployment_exemption():
    usage_metrics.configure_usage_event_recorder(
        usage_metrics.exempt_usage_event_recorder
    )
    try:
        outcome = usage_metrics.record_usage_event_strict(
            org_id="7",
            event_name="learnings_generated",
            event_category="learning",
            event_key="learn:profile:1",
        )
    finally:
        usage_metrics.configure_usage_event_recorder(None)

    assert outcome is UsageEventDeliveryStatus.EXEMPT


@pytest.mark.parametrize(
    "outcome",
    [UsageEventDeliveryStatus.FAILED, UsageEventDeliveryStatus.REJECTED],
)
def test_strict_delivery_raises_for_unaccepted_outcome(outcome):
    usage_metrics.configure_usage_event_recorder(lambda _event: outcome)
    try:
        with pytest.raises(UsageEventDeliveryError) as exc_info:
            usage_metrics.record_usage_event_strict(
                org_id="7",
                event_name="learnings_generated",
                event_category="learning",
                event_key="learn:profile:1",
            )
    finally:
        usage_metrics.configure_usage_event_recorder(None)

    assert exc_info.value.status is outcome


def test_strict_delivery_propagates_recorder_exception_while_ordinary_stays_fail_open():
    def fail(_event):
        raise RuntimeError("sink unavailable")

    usage_metrics.configure_usage_event_recorder(fail)
    try:
        usage_metrics.record_usage_event(
            org_id="7",
            event_name="search_request",
            event_category="application",
            event_key="search:1",
        )
        with pytest.raises(RuntimeError, match="sink unavailable"):
            usage_metrics.record_usage_event_strict(
                org_id="7",
                event_name="learnings_generated",
                event_category="learning",
                event_key="learn:profile:1",
            )
    finally:
        usage_metrics.configure_usage_event_recorder(None)
