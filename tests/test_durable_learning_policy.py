"""Independent discovery and aggregation timing retain real operator control."""

import os
import subprocess
import sys
import threading

import pytest

from reflexio.durable_learning_policy import validate_discovery_timing_env


@pytest.mark.parametrize(
    "env",
    [
        {},
        {
            "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": " ",
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": "",
        },
        {
            "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": " 2 ",
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": "300.0",
        },
    ],
)
def test_discovery_defaults_and_blank_inputs_preserve_two_second_poll_and_lease(env):
    from reflexio.durable_learning_policy import resolve_discovery_timing

    timing = resolve_discovery_timing(env)
    assert (timing.poll_seconds, timing.lease_seconds) == (2.0, 300)


@pytest.mark.parametrize("poll,lease", [("0.05", "1"), ("10", "60"), ("7200", "900")])
def test_custom_policies_are_independent_and_mapping_authoritative(
    monkeypatch, poll, lease
):
    from reflexio.durable_learning_policy import resolve_discovery_timing

    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_POLL_SECONDS", "invalid-ambient")
    env = {
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": poll,
        "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": lease,
    }
    timing = resolve_discovery_timing(env)
    assert (timing.poll_seconds, timing.lease_seconds) == (float(poll), int(lease))
    assert resolve_discovery_timing({}).poll_seconds == 2.0
    assert env["REFLEXIO_DURABLE_LEARNING_POLL_SECONDS"] == poll


@pytest.mark.parametrize(
    "key",
    [
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS",
        "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS",
    ],
)
@pytest.mark.parametrize(
    "value", ["0", "-1", "nan", "inf", "private-invalid-input", "1e100"]
)
def test_invalid_discovery_timing_fails_with_value_free_errors(key, value):
    with pytest.raises(ValueError, match=key) as error:
        validate_discovery_timing_env({key: value})
    assert "private-invalid-input" not in str(error.value)


@pytest.mark.parametrize("value", ["0.05", "1.5"])
def test_claim_duration_requires_an_integral_number_of_seconds(value):
    with pytest.raises(ValueError, match="LEASE_SECONDS"):
        validate_discovery_timing_env(
            {"REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": value}
        )


def test_discovery_wait_and_epoch_boundaries_are_enforced():
    from reflexio.durable_learning_policy import (
        MAX_DISCOVERY_LEASE_SECONDS,
        resolve_discovery_timing,
    )

    timing = resolve_discovery_timing(
        {
            "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": str(threading.TIMEOUT_MAX),
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": str(MAX_DISCOVERY_LEASE_SECONDS),
        }
    )
    assert timing.lease_seconds / 3 <= threading.TIMEOUT_MAX
    with pytest.raises(ValueError, match="LEASE_SECONDS"):
        resolve_discovery_timing(
            {
                "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": str(
                    MAX_DISCOVERY_LEASE_SECONDS + 1
                )
            }
        )
    with pytest.raises(ValueError, match="POLL_SECONDS"):
        resolve_discovery_timing(
            {"REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": str(threading.TIMEOUT_MAX * 2)}
        )


@pytest.mark.parametrize(
    "value,accepted", [("10", True), ("private-invalid-input", False)]
)
def test_server_import_validates_policy_before_workers_start(tmp_path, value, accepted):
    env = {
        **os.environ,
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": value,
        "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": "900",
        "REFLEXIO_LOG_DIR": str(tmp_path),
        "LOCAL_STORAGE_PATH": str(tmp_path),
    }
    result = subprocess.run(  # noqa: S603 - fixed interpreter and literal import; no shell.
        [sys.executable, "-c", "import reflexio.server"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert (result.returncode == 0) is accepted, result.stderr
    if not accepted:
        assert "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS" in result.stderr
        assert "private-invalid-input" not in result.stderr


_AGGREGATION_TIMING_KEYS = (
    "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS",
    "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS",
)


@pytest.mark.parametrize(
    "values", [{}, {"interval": " ", "wait": ""}, {"interval": "3600", "wait": "1.0"}]
)
def test_aggregation_defaults_preserve_execution_policy(monkeypatch, values):
    from reflexio.durable_learning_policy import validate_aggregation_timing_env
    from reflexio.server.operation_limiter import _timeout_for
    from reflexio.server.services.playbook.aggregation_scheduler import (
        aggregation_min_interval_seconds,
    )

    for name, key in zip(_AGGREGATION_TIMING_KEYS, ("interval", "wait"), strict=True):
        monkeypatch.delenv(name, raising=False)
        if key in values:
            monkeypatch.setenv(name, values[key])
    validate_aggregation_timing_env(
        {name: os.environ.get(name, "") for name in _AGGREGATION_TIMING_KEYS}
    )
    assert aggregation_min_interval_seconds() == 3600
    assert _timeout_for("aggregation") == 1.0


@pytest.mark.parametrize("key", _AGGREGATION_TIMING_KEYS)
@pytest.mark.parametrize("value", ["-1", "nan", "inf", "private-value"])
def test_invalid_aggregation_policy_rejected_before_scheduler_or_admission(
    monkeypatch, key, value
):
    from reflexio.durable_learning_policy import validate_aggregation_timing_env
    from reflexio.server import operation_limiter
    from reflexio.server.services.playbook.aggregation_scheduler import (
        PlaybookAggregationScheduler,
        aggregation_min_interval_seconds,
    )

    for name in _AGGREGATION_TIMING_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        operation_limiter,
        "_state_for",
        lambda *_args: pytest.fail("semaphore allocated before adoption"),
    )
    for operation in (
        lambda: validate_aggregation_timing_env({key: value}),
        aggregation_min_interval_seconds,
        lambda: PlaybookAggregationScheduler(context_provider=lambda: []),
        lambda: operation_limiter._timeout_for("aggregation"),
    ):
        with pytest.raises(ValueError, match=key) as error:
            operation()
        if value == "private-value":
            assert value not in str(error.value)
    # An explicit internal wait cannot bypass migration of stale deployment inputs.
    with (
        pytest.raises(ValueError, match=key),
        operation_limiter.operation_limit("org", "aggregation", timeout_seconds=0),
    ):
        pytest.fail("work started before adoption")


@pytest.mark.parametrize("key", _AGGREGATION_TIMING_KEYS)
def test_aggregation_startup_rejects_custom_timing(tmp_path, key):
    env = {**os.environ, key: "private-value", "LOCAL_STORAGE_PATH": str(tmp_path)}
    result = subprocess.run(  # noqa: S603 - literal import with the current interpreter.
        [sys.executable, "-c", "import reflexio.server"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert key in result.stderr
    assert "private-value" not in result.stderr


@pytest.mark.parametrize("wait", ["0", "0.025", "3.5"])
def test_supported_custom_aggregation_timing_reaches_all_runtime_consumers(
    monkeypatch, wait
):
    from reflexio.durable_learning_policy import validate_aggregation_timing_env
    from reflexio.server.operation_limiter import _timeout_for, operation_limit
    from reflexio.server.services.playbook.aggregation_scheduler import (
        aggregation_min_interval_seconds,
    )

    monkeypatch.setenv("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "17")
    monkeypatch.setenv("REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", wait)
    validate_aggregation_timing_env(dict(os.environ))
    assert aggregation_min_interval_seconds() == 17
    assert _timeout_for("aggregation") == float(wait)
    with operation_limit("custom-policy-org", "aggregation"):
        pass


def test_aggregation_validation_uses_supplied_mapping(monkeypatch):
    from reflexio.durable_learning_policy import (
        aggregation_admission_wait_seconds,
        aggregation_min_interval_seconds,
        validate_aggregation_timing_env,
    )

    monkeypatch.setenv("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", "private-value")
    env = {
        "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS": "29",
        "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS": "0",
    }
    validate_aggregation_timing_env(env)
    assert aggregation_min_interval_seconds(env) == 29
    assert aggregation_admission_wait_seconds(env) == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", str(1 << 63)),
        ("REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS", str(1 << 62)),
        ("REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS", "1e20"),
        (
            "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS",
            str(threading.TIMEOUT_MAX * 2),
        ),
    ],
)
def test_aggregation_settings_reject_consumer_overflow(key, value):
    from reflexio.durable_learning_policy import validate_aggregation_timing_env

    with pytest.raises(ValueError, match=key):
        validate_aggregation_timing_env({key: value})


def test_aggregation_consumer_range_boundaries_are_accepted():
    from reflexio.durable_learning_policy import (
        MAX_AGGREGATION_INTERVAL_SECONDS,
        validate_aggregation_timing_env,
    )

    validate_aggregation_timing_env(
        {
            "REFLEXIO_AGGREGATION_MIN_INTERVAL_SECONDS": str(
                MAX_AGGREGATION_INTERVAL_SECONDS
            ),
            "REFLEXIO_AGGREGATION_CONCURRENCY_TIMEOUT_SECONDS": str(
                threading.TIMEOUT_MAX
            ),
        }
    )
