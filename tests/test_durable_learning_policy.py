"""Retiring discovery overrides cannot silently change a custom deployment."""

import os
import subprocess
import sys

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
            "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS": "300",
        },
    ],
)
def test_retired_defaults_are_accepted_without_becoming_overrides(env):
    validate_discovery_timing_env(env)


@pytest.mark.parametrize(
    "key",
    [
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS",
        "REFLEXIO_DURABLE_LEARNING_LEASE_SECONDS",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "0.05", "10", "nan", "inf", "invalid"])
def test_custom_or_invalid_retired_timing_requires_migration(key, value):
    with pytest.raises(ValueError, match=key):
        validate_discovery_timing_env({key: value})


def test_server_import_rejects_custom_timing_before_workers_start(tmp_path):
    env = {
        **os.environ,
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS": "10",
        "LOCAL_STORAGE_PATH": str(tmp_path),
    }
    result = subprocess.run(  # noqa: S603 - fixed interpreter and literal import, no shell/input code.
        [sys.executable, "-c", "import reflexio.server"],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert (
        "REFLEXIO_DURABLE_LEARNING_POLL_SECONDS is retired; remove it" in result.stderr
    )


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
