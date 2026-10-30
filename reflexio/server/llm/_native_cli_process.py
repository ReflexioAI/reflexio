"""Fresh native-launch owner; deliberately imports only the standard library.

SDK hooks remain in their existing completion worker. Native pipes live here so
an unrelated forked SDK worker cannot inherit their writers.
"""

from __future__ import annotations

import logging
import os
import pickle
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _publish(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        temporary.chmod(0o600)
        pickle.dump(value, stream)
    temporary.replace(path)


def _remaining(request: dict[str, Any]) -> float:
    remaining = request["deadline"] - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(request["argv"], request["kwargs"]["timeout"])
    return remaining


@dataclass
class _NativeExecution:
    process: subprocess.Popen[str] | None = None
    exit_confirmed: bool = True


def _run(
    request: dict[str, Any], directory: Path, execution: _NativeExecution
) -> subprocess.CompletedProcess[str]:
    _remaining(request)
    if (directory / "cancelled").exists():
        raise RuntimeError("Native CLI launch cancelled")
    kwargs = dict(request["kwargs"])
    argv = request["argv"]
    input_text = kwargs.pop("input", None)
    timeout = kwargs.pop("timeout")
    kwargs.pop("check", None)
    kwargs.pop("capture_output", None)
    # Reject malformed OS strings before entering uncertain child ownership.
    # OpenClaw carries user prompt text in argv, including potentially NUL.
    os_strings = list(argv)
    if kwargs.get("cwd") is not None:
        os_strings.append(kwargs["cwd"])
    for key, value in kwargs.get("env", {}).items():
        encoded_key = os.fsencode(key)
        if os.name != "nt" and b"=" in encoded_key:
            raise ValueError("illegal environment variable name")
        os_strings.extend((key, value))
    if any(b"\x00" in os.fsencode(value) for value in os_strings):
        raise ValueError("embedded null byte")
    execution.exit_confirmed = False
    try:
        process = subprocess.Popen(  # noqa: S603 — constructed provider argv
            argv,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
    except OSError:
        # Popen reports failed OS creation/exec after collecting its failed child.
        execution.exit_confirmed = True
        raise
    execution.process = process
    try:
        if (directory / "cancelled").exists():
            raise RuntimeError("Native CLI launch completed after cancellation")
        stdout, stderr = process.communicate(input_text, timeout=_remaining(request))
        execution.exit_confirmed = True
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
    except BaseException as exc:
        process.kill()
        process.wait()
        execution.exit_confirmed = True
        if isinstance(exc, subprocess.TimeoutExpired):
            exc.timeout = timeout
        raise
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def native_cli_main(directory_name: str, counter: int, deadline: float) -> None:
    """Publish a terminal receipt only after native exit and stream retirement."""
    directory = Path(directory_name)
    execution = _NativeExecution()
    try:
        with (directory / f"request-{counter}.pickle").open("rb") as stream:
            request = pickle.load(stream)  # noqa: S301 — private attempt files
        (directory / f"request-{counter}.pickle").unlink()
        if request["counter"] != counter:
            raise RuntimeError("Native CLI request file identity mismatch")
        request["deadline"] = min(request["deadline"], deadline)
        result, status = _run(request, directory, execution), "ok"
    except BaseException as exc:
        result, status = exc, "error"
    if not execution.exit_confirmed:
        logger.error(
            "Native CLI exit unconfirmed; retaining launch owner (%s)",
            type(result).__name__,
        )
        # Keep the real handle reachable. Interrupting this wait exits without
        # a terminal receipt, so the parent still retains admission and files.
        threading.Event().wait()
        return
    _publish(directory / f"retired-{counter}.pickle", (counter, status, result))


if __name__ == "__main__":
    native_cli_main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
