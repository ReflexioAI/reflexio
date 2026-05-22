"""Unit tests for the EvalHealth singleton — counters and serialization."""

from reflexio.server.services.agent_success_evaluation._eval_health import (
    EvalHealth,
    SkipReason,
)


def test_skip_reason_counts_aggregate_per_reason() -> None:
    """EvalHealth aggregates skip counts by reason."""
    health = EvalHealth()
    health.record_skip(SkipReason.ALREADY_EVALUATED)
    health.record_skip(SkipReason.ALREADY_EVALUATED)
    health.record_skip(SkipReason.NO_REQUESTS)

    status = health.get_status()

    assert status["skip_counts"]["already_evaluated"] == 2
    assert status["skip_counts"]["no_requests"] == 1
    assert status["skip_counts"].get("no_interactions", 0) == 0


def test_producer_failure_24h_count_decays() -> None:
    """Producer failures older than 24h fall off the rolling window."""
    health = EvalHealth()
    health.record_producer_failure(at_ts=0)
    health.record_producer_failure(at_ts=1000)
    health.record_producer_failure(at_ts=1000)

    status = health.get_status(now_ts=86_900)

    assert status["producer_failures_24h"] == 2


def test_tick_timestamp_records_latest() -> None:
    """`record_tick` overwrites with the newest monotonic time."""
    health = EvalHealth()
    health.record_tick(monotonic_ts=10.0)
    health.record_tick(monotonic_ts=42.5)

    status = health.get_status()

    assert status["last_tick_monotonic"] == 42.5
