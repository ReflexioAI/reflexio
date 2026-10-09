"""Portable regression/diagnostic for the storage slice of Windows issue #295."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_concurrent_publishes_persist_while_reads_serve():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHON_DOTENV_DISABLED="1", PYTHONPATH=str(root))
    # The child dumps stacks at 20s; the parent kills it at 30s, including if a
    # SQLite worker blocks event-loop/executor teardown after asyncio timeout.
    # subprocess.run implements this on Windows as well as POSIX.
    try:
        completed = subprocess.run(  # noqa: S603 — executable and script are repository-owned
            [sys.executable, str(root / "tests/test_scripts/local_publish_probe.py")],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Publish probe exceeded 30s. Thread stacks:\n{exc.stderr}")
    assert completed.returncode == 0, completed.stderr + completed.stdout
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["persisted_requests"] == report["concurrent_publishes"] == 20
    assert report["concurrent_read_requests"] == 20
    assert report["legacy_root_post_status"] == 405
    assert report["platform"]
    print(json.dumps(report))
