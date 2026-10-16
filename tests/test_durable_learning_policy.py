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
    with pytest.raises(ValueError, match=f"{key} is retired; remove it"):
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
