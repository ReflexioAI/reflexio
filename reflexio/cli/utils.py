"""Shared utilities for service process management."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

from reflexio.cli.log_format import format_service_line


@dataclasses.dataclass
class ServiceConfig:
    """Configuration for a single service to launch."""

    name: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None


def get_env_port(name: str, default: int) -> int:
    """Read a port from an environment variable, falling back to default."""
    val = os.environ.get(name)
    if val is not None:
        try:
            return int(val)
        except ValueError:
            print(f"Warning: invalid {name}={val!r}, using default {default}")
    return default


def find_pids_on_port(port: int) -> list[int]:
    """Find process IDs listening on the given port using lsof."""
    try:
        result = subprocess.run(
            ["lsof", "-t", f"-i:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [
                int(p) for p in result.stdout.strip().split("\n") if p.strip().isdigit()
            ]
    except FileNotFoundError:
        pass
    return []


def find_pids_by_pattern(pattern: str) -> list[int]:
    """Find process IDs matching a command pattern using pgrep."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [
                int(p) for p in result.stdout.strip().split("\n") if p.strip().isdigit()
            ]
    except FileNotFoundError:
        pass
    return []


def kill_processes(
    pids: list[int], graceful_timeout: float = 2.0, force: bool = False
) -> None:
    """Kill processes: SIGTERM first, then SIGKILL survivors after timeout."""
    if not pids:
        return

    unique_pids = list(set(pids))
    my_pid = os.getpid()
    unique_pids = [p for p in unique_pids if p != my_pid]

    if not unique_pids:
        return

    sig = signal.SIGKILL if force else signal.SIGTERM
    for pid in unique_pids:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            pass
        except PermissionError:
            print(f"Warning: no permission to kill PID {pid}")

    if force:
        return

    time.sleep(graceful_timeout)

    for pid in unique_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            pass


def get_pidfile_path(ports: dict[str, int]) -> Path:
    """Get a unique pidfile path based on the port combination."""
    port_str = json.dumps(ports, sort_keys=True)
    port_hash = hashlib.md5(port_str.encode()).hexdigest()[:8]  # noqa: S324
    return Path(f"/tmp/reflexio_services_{port_hash}.json")  # noqa: S108


def write_pidfile(pidfile: Path, service_pids: dict[str, int]) -> None:
    """Write service PIDs to the pidfile."""
    pidfile.write_text(json.dumps(service_pids, indent=2))


def read_pidfile(pidfile: Path) -> dict[str, int]:
    """Read service PIDs from the pidfile. Returns empty dict if not found."""
    if not pidfile.exists():
        return {}
    try:
        return json.loads(pidfile.read_text())
    except (json.JSONDecodeError, OSError):  # fmt: skip
        return {}


def remove_pidfile(pidfile: Path) -> None:
    """Remove the pidfile if it exists."""
    with contextlib.suppress(OSError):
        pidfile.unlink(missing_ok=True)


# Patterns that indicate a service is ready to accept requests
_READY_PATTERNS: dict[str, list[str]] = {
    "backend": ["Application startup complete"],
    "frontend": ["Ready in"],
    "docs": ["Ready in"],
}

# Extra env vars to suppress noise from subprocesses at source
_NOISE_SUPPRESSION_ENV: dict[str, dict[str, str]] = {
    "backend": {"LITELLM_LOG": "ERROR"},
    "frontend": {"NODE_NO_WARNINGS": "1"},
    "docs": {"NODE_NO_WARNINGS": "1"},
}


def _stream_output(
    proc: subprocess.Popen[bytes],
    service_name: str,
    lock: threading.Lock,
    ready_event: threading.Event,
) -> None:
    """Read subprocess stdout line by line, prefix, and write to stdout.

    Args:
        proc: The subprocess to read from.
        service_name: Name of the service for prefixing.
        lock: Lock to prevent interleaved partial lines.
        ready_event: Event to set when the service is ready.
    """
    ready_patterns = _READY_PATTERNS.get(service_name, [])
    assert proc.stdout is not None  # noqa: S101
    for raw_line in proc.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        formatted = format_service_line(service_name, line)
        with lock:
            sys.stdout.write(formatted + "\n")
            sys.stdout.flush()
        # Check for ready patterns
        if not ready_event.is_set() and any(p in line for p in ready_patterns):
            ready_event.set()


def run_services(
    services: list[ServiceConfig],
    ports: dict[str, int],
    *,
    on_all_ready: Callable[[dict[str, int]], None] | None = None,
) -> None:
    """Launch services, pipe output with prefixes, and manage lifecycle.

    Each service's stdout/stderr is captured and prefixed with a colored
    service tag (e.g., [backend]). A callback is invoked when all services
    report ready (or after a timeout).

    Args:
        services: List of service configurations to launch.
        ports: Port mapping for pidfile identification.
        on_all_ready: Optional callback invoked with ports when all services are ready.
    """
    processes: dict[str, subprocess.Popen[bytes]] = {}
    threads: list[threading.Thread] = []
    ready_events: dict[str, threading.Event] = {}
    output_lock = threading.RLock()
    pidfile = get_pidfile_path(ports)

    def shutdown(_signum: int | None = None, _frame: object = None) -> None:
        with output_lock:
            sys.stdout.write("\nShutting down services...\n")
            sys.stdout.flush()
        for proc in processes.values():
            with contextlib.suppress(OSError):
                proc.terminate()
        deadline = time.time() + 3.0
        for proc in processes.values():
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
        for t in threads:
            t.join(timeout=1.0)
        remove_pidfile(pidfile)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for svc in services:
        noise_env = _NOISE_SUPPRESSION_ENV.get(svc.name, {})
        merged_env = {**os.environ, **(svc.env or {}), **noise_env}
        with output_lock:
            sys.stdout.write(f"Starting {svc.name}...\n")
            sys.stdout.flush()
        proc = subprocess.Popen(
            svc.command,
            cwd=svc.cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        processes[svc.name] = proc
        ready_event = threading.Event()
        ready_events[svc.name] = ready_event
        t = threading.Thread(
            target=_stream_output,
            args=(proc, svc.name, output_lock, ready_event),
            daemon=True,
        )
        t.start()
        threads.append(t)
        with output_lock:
            sys.stdout.write(f"  {svc.name} started (PID {proc.pid})\n")
            sys.stdout.flush()

    # Write pidfile
    pids = {name: proc.pid for name, proc in processes.items()}
    write_pidfile(pidfile, pids)

    # Wait for all services to be ready (or timeout with shared deadline)
    if on_all_ready:
        deadline = time.monotonic() + 60
        all_ready = all(
            evt.wait(timeout=max(0, deadline - time.monotonic()))
            for evt in ready_events.values()
        )
        if all_ready:
            on_all_ready(ports)

    # Wait for any child to exit
    try:
        while processes:
            for name, proc in list(processes.items()):
                ret = proc.poll()
                if ret is not None:
                    with output_lock:
                        sys.stdout.write(
                            format_service_line(name, f"exited with code {ret}") + "\n"
                        )
                        sys.stdout.flush()
                    del processes[name]
            if processes:
                time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()

    remove_pidfile(pidfile)


def stop_services(
    port_map: dict[str, int],
    process_patterns: dict[str, str],
    force: bool = False,
) -> None:
    """Stop services by port and process pattern.

    Args:
        port_map: Mapping of service name to port number.
        process_patterns: Mapping of service name to pgrep pattern.
        force: If True, send SIGKILL immediately instead of graceful shutdown.
    """
    pidfile = get_pidfile_path(port_map)
    saved_pids = read_pidfile(pidfile)

    for name, port in port_map.items():
        pids_from_port = find_pids_on_port(port)
        pids_from_pattern = (
            find_pids_by_pattern(process_patterns[name])
            if name in process_patterns
            else []
        )
        pids_from_saved = [saved_pids[name]] if name in saved_pids else []

        all_pids = list(set(pids_from_port + pids_from_pattern + pids_from_saved))

        if all_pids:
            kill_processes(all_pids, force=force)
            print(f"Stopped {name} (port {port})")
        else:
            print(f"{name} (port {port}) not running")

    remove_pidfile(pidfile)
