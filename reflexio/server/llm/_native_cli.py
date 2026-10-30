"""Parent-owned immediate CLI processes for isolated completion workers.

Only launch/communication/reaping crosses this seam; provider SDK code stays in
its isolated worker. This is not containment of arbitrary CLI descendants.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexio.server.llm._litellm_types import LLMHardTimeoutError
from reflexio.server.llm._provider_concurrency import retain_current_provider_slot

logger = logging.getLogger(__name__)
NATIVE_CLI_REQUEST = "native_cli"


@dataclass(frozen=True)
class NativeCLIContext:
    directory: str
    response_socket: socket.socket


@dataclass
class _WorkerContext:
    context: NativeCLIContext
    queue: Any
    deadline: float
    lock: Any
    counter: int = 0


_worker_context: _WorkerContext | None = None
# Retain resources until worker and native exit are confirmed, even if a reaper
# fails and all caller threads return. This is ownership, not another capacity.
_live_owners: set[NativeCLIOwner] = set()
_live_owners_lock = threading.Lock()


def native_cli_worker(
    context: NativeCLIContext | None, queue: Any, deadline: float
) -> None:
    """Install delegation for the entire worker lifetime, including SDK threads."""
    global _worker_context
    _worker_context = (
        _WorkerContext(context, queue, deadline, threading.Lock()) if context else None
    )


def native_cli_directory() -> str | None:
    """Keep provider output files alive after their isolated worker is killed."""
    return _worker_context.context.directory if _worker_context else None


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if not math.isfinite(remaining) or remaining <= 0:
        raise LLMHardTimeoutError("Native CLI request deadline expired")
    return remaining


def _publish(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as stream:
        temporary.chmod(0o600)
        pickle.dump(value, stream)
    temporary.replace(path)


def run_native_cli(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    """Run the providers' captured-text commands, delegated inside a worker."""
    if (
        kwargs.get("capture_output") is not True
        or kwargs.get("text") is not True
        or kwargs.get("check") is not False
    ):
        raise ValueError("Native CLI launcher requires captured text and check=False")
    worker = _worker_context
    if worker is None:
        return subprocess.run(argv, **kwargs)  # noqa: S603
    timeout = float(kwargs["timeout"])
    if not math.isfinite(timeout) or timeout <= 0:
        raise subprocess.TimeoutExpired(argv, timeout)
    deadline = min(worker.deadline, time.monotonic() + timeout)
    if not worker.lock.acquire(timeout=_remaining(deadline)):
        raise LLMHardTimeoutError("Native CLI request expired waiting for its launcher")
    try:
        _remaining(deadline)
        worker.counter += 1
        counter = worker.counter
        directory = Path(worker.context.directory)
        request = {
            "counter": counter,
            "argv": argv,
            "kwargs": {
                **kwargs,
                "cwd": str(Path.cwd()) if kwargs.get("cwd") is None else kwargs["cwd"],
                "env": dict(os.environ) if kwargs.get("env") is None else kwargs["env"],
            },
            "deadline": deadline,
        }
        _publish(directory / f"request-{counter}.pickle", request)
        worker.queue.put((NATIVE_CLI_REQUEST, counter))
        notification = bytearray()
        while len(notification) < 8:
            worker.context.response_socket.settimeout(_remaining(worker.deadline))
            part = worker.context.response_socket.recv(8 - len(notification))
            if not part:
                raise LLMHardTimeoutError(
                    "Native CLI owner closed before returning a result"
                )
            notification.extend(part)
        if int.from_bytes(notification, "big") != counter:
            raise RuntimeError("Native CLI response identity mismatch")
        with (directory / f"response-{counter}.pickle").open("rb") as stream:
            identity, status, result = pickle.load(stream)  # noqa: S301 — private attempt files
        if identity != counter:
            raise RuntimeError("Native CLI response file identity mismatch")
        (directory / f"response-{counter}.pickle").unlink()
        if status == "error":
            raise result
        if status != "ok" or not isinstance(result, subprocess.CompletedProcess):
            raise RuntimeError("Native CLI response kind mismatch")
        return result
    finally:
        worker.lock.release()


@dataclass
class _NativeTask:
    release: Any
    state: str = "created"
    exit_confirmed: bool = True
    process: subprocess.Popen[str] | None = None


class NativeCLIOwner:
    """Own worker scratch storage and native calls until their actual retirement."""

    def __init__(self, deadline: float):
        self.deadline = deadline
        self._scratch = tempfile.TemporaryDirectory(prefix="reflexio-native-cli-")
        self._sender, receiver = socket.socketpair()
        self.context = NativeCLIContext(self._scratch.name, receiver)
        self._lock = threading.Lock()
        self._closed = False
        self._worker_exited = False
        self._counter = 0
        self._cleaned = False
        self._tasks: dict[int, _NativeTask] = {}
        with _live_owners_lock:
            _live_owners.add(self)

    def create_worker(
        self, context: Any, target: Any, params: dict[str, Any]
    ) -> tuple[Any, Any]:
        """Create the isolated worker and retire all resources on factory failure."""
        result_queue = None
        try:
            result_queue = context.Queue(maxsize=1)
            process = context.Process(
                target=target,
                args=(params, result_queue, self.deadline, self.context),
                daemon=True,
            )
            return process, result_queue
        except BaseException:
            self.close()
            self.worker_exited()
            if result_queue is not None:
                result_queue.close()
            raise

    def worker_started(self) -> None:
        self.context.response_socket.close()

    def close(self) -> None:
        """Revoke launch admission without waiting for native startup or cleanup."""
        with self._lock:
            self._closed = True

    def worker_exited(self) -> None:
        with self._lock:
            self._worker_exited = True
        self._cleanup_if_finished()

    def _cleanup_if_finished(self) -> None:
        with self._lock:
            finished = self._worker_exited and not self._tasks and not self._cleaned
            if finished:
                self._cleaned = True
        if finished:
            self._sender.close()
            self.context.response_socket.close()
            try:
                self._scratch.cleanup()
            except OSError:
                logger.exception("Native CLI scratch cleanup failed; owner retained")
            else:
                with _live_owners_lock:
                    _live_owners.discard(self)

    def dispatch(self, counter: int) -> None:
        _remaining(self.deadline)
        with self._lock:
            if self._closed:
                raise LLMHardTimeoutError("Native CLI owner is closed")
            if type(counter) is not int or counter != self._counter + 1:
                raise RuntimeError("Native CLI request identity mismatch")
            self._counter = counter
            # The caller owns the ContextVar-held lease; a new thread does not.
            task = _NativeTask(retain_current_provider_slot())
            self._tasks[counter] = task
        try:
            thread = threading.Thread(
                target=self._execute,
                args=(counter, task),
                name="reflexio-native-cli",
                daemon=True,
            )
            thread.start()
        except BaseException:
            with self._lock:
                cancelled = task.state == "created"
                if cancelled:
                    task.state = "cancelled"
            if cancelled:
                self._finish(counter, task)
            raise

    def _finish(self, counter: int, task: _NativeTask) -> None:
        with self._lock:
            self._tasks.pop(counter)
        try:
            task.release()
        finally:
            self._cleanup_if_finished()

    def _execute(self, counter: int, task: _NativeTask) -> None:
        with self._lock:
            if task.state == "cancelled":
                return
            task.state = "running"
        try:
            directory = Path(self.context.directory)
            try:
                with (directory / f"request-{counter}.pickle").open("rb") as stream:
                    request = pickle.load(stream)  # noqa: S301 — private attempt files
                (directory / f"request-{counter}.pickle").unlink()
                if request["counter"] != counter:
                    raise RuntimeError("Native CLI request file identity mismatch")
                result = self._run(request, task)
                status = "ok"
            except Exception as exc:
                result, status = exc, "error"
            if not task.exit_confirmed:
                logger.error(
                    "Native CLI exit could not be confirmed; admission retained"
                )
                return
            with self._lock:
                closed = self._closed
            if not closed:
                _publish(
                    directory / f"response-{counter}.pickle", (counter, status, result)
                )
                self._sender.settimeout(_remaining(self.deadline))
                self._sender.sendall(counter.to_bytes(8, "big"))
        except (OSError, LLMHardTimeoutError):
            # The caller can already have expired or killed its worker. Native
            # exit, rather than delivery of the reply, determines its ownership.
            logger.debug(
                "Native CLI reply outlived its completion worker", exc_info=True
            )
        finally:
            if task.exit_confirmed:
                self._finish(counter, task)

    def _run(
        self, request: dict[str, Any], task: _NativeTask
    ) -> subprocess.CompletedProcess[str]:
        deadline = min(self.deadline, request["deadline"])
        try:
            _remaining(deadline)
        except LLMHardTimeoutError as exc:
            raise subprocess.TimeoutExpired(
                request["argv"], request["kwargs"]["timeout"]
            ) from exc
        with self._lock:
            if self._closed:
                raise LLMHardTimeoutError("Native CLI launch cancelled")
        kwargs = dict(request["kwargs"])
        argv = request["argv"]
        input_text = kwargs.pop("input", None)
        timeout = kwargs.pop("timeout")
        kwargs.pop("check", None)
        kwargs.pop("capture_output", None)
        process = subprocess.Popen(  # noqa: S603 — same validated provider argv
            argv,
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs,
        )
        task.process = process
        task.exit_confirmed = False
        try:
            with self._lock:
                closed = self._closed
            if closed:
                raise LLMHardTimeoutError(
                    "Native CLI launch completed after cancellation"
                )
            stdout, stderr = process.communicate(
                input_text, timeout=_remaining(deadline)
            )
            task.exit_confirmed = True
            return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)
        except BaseException as exc:
            process.kill()
            process.wait()
            task.exit_confirmed = True
            if isinstance(exc, LLMHardTimeoutError):
                raise subprocess.TimeoutExpired(argv, timeout) from exc
            if isinstance(exc, subprocess.TimeoutExpired):
                exc.timeout = timeout
            raise
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
