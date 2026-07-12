from reflexio.server import usage_metrics
from reflexio.server.usage_metrics import UsageEvent


def test_usage_event_carries_event_key():
    captured = []
    usage_metrics.configure_usage_event_recorder(captured.append)
    try:
        usage_metrics.record_usage_event(
            org_id="7", event_name="search_request", event_category="application",
            count_value=1, event_key="search:abc",
        )
    finally:
        usage_metrics.configure_usage_event_recorder(None)
    assert captured and captured[0].event_key == "search:abc"


def test_event_key_defaults_none():
    assert UsageEvent(org_id="7", event_name="x", event_category="y").event_key is None
