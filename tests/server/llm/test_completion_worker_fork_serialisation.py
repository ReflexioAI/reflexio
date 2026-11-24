"""Concurrent completion-worker starts must not overlap their forks.

filelock (>= 3.30) installs a ``sys.audit`` hook that raises
``RuntimeError("os.fork is unsafe while filelock is changing descriptor
ownership")`` in any thread calling ``os.fork`` while another thread is between
filelock's before-fork and after-fork handlers. Under the ``fork`` start method
every isolated completion forks from the threaded server, so two concurrent
completions failed in production (Sentry PYTHON-FASTAPI-56).

The scenario runs in a fresh interpreter. An at-fork ``before`` hook registered
ahead of the filelock import runs after filelock's own (before-hooks run in
reverse registration order), so it can hold the first fork open exactly while
filelock counts that thread as mid-fork. The second start then deterministically
hits the audit hook unless starts are serialised.
"""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(os, "register_at_fork"), reason="the fork start method needs fork"
)

_SCENARIO = """
import multiprocessing
import os
import sys
import threading

assert "filelock" not in sys.modules, "filelock must import after the hold hook"
in_fork = threading.Event()
resume = threading.Event()


def hold_first_fork():
    if not in_fork.is_set():
        in_fork.set()
        resume.wait()


os.register_at_fork(before=hold_first_fork)

import filelock  # noqa: E402,F401

from reflexio.server.llm._litellm_subprocess import (  # noqa: E402
    start_completion_worker,
)

start = start_completion_worker if sys.argv[1] == "helper" else (lambda p: p.start())
context = multiprocessing.get_context("fork")
first = context.Process(target=os._exit, args=(0,))
second = context.Process(target=os._exit, args=(0,))
holder = threading.Thread(target=start, args=(first,))
holder.start()
assert in_fork.wait(30), "first fork never reached the hold hook"

outcome = []


def start_second():
    try:
        start(second)
        outcome.append("started")
    except RuntimeError as exc:
        outcome.append(f"RuntimeError: {exc}")


contender = threading.Thread(target=start_second)
contender.start()
contender.join(0.5)
blocked = contender.is_alive()
resume.set()
holder.join(30)
contender.join(30)
for process in (first, second):
    if process.pid is not None:
        process.join(30)
sys.stdout.write(f"blocked={blocked} outcome={outcome[0]}\\n")
sys.stdout.flush()
"""


def _run_scenario(mode: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [sys.executable, "-c", _SCENARIO, mode],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


def test_unserialised_concurrent_fork_raises_filelock_guard() -> None:
    """Pins the production failure, so the serialised test is not vacuous."""
    assert _run_scenario("direct") == (
        "blocked=False outcome=RuntimeError: os.fork is unsafe while filelock "
        "is changing descriptor ownership"
    )


def test_start_completion_worker_waits_for_the_fork_in_progress() -> None:
    assert _run_scenario("helper") == "blocked=True outcome=started"
