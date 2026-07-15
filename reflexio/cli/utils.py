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
import tempfile
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
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
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


def get_requested_port_conflicts(ports: dict[str, int]) -> list[str]:
    """Return startup-blocking conflicts for requested service ports."""
    conflicts: list[str] = []
    services_by_port: dict[int, list[str]] = {}
    for name, port in ports.items():
        services_by_port.setdefault(port, []).append(name)

    for port, names in sorted(services_by_port.items()):
        if len(names) > 1:
            conflicts.append(
                f"port {port} is assigned to multiple services: {', '.join(sorted(names))}"
            )

    for name, port in sorted(ports.items()):
        pids = find_pids_on_port(port)
        if pids:
            pid_list = ", ".join(str(pid) for pid in sorted(set(pids)))
            conflicts.append(
                f"{name} port {port} is already in use by PID(s): {pid_list}"
            )

    return conflicts


def ensure_requested_ports_available(ports: dict[str, int]) -> None:
    """Exit with a clear message when requested service ports are unavailable."""
    conflicts = get_requested_port_conflicts(ports)
    if not conflicts:
        return

    sys.stdout.write("Error: cannot start Reflexio services because ports conflict.\n")
    for conflict in conflicts:
        sys.stdout.write(f"  - {conflict}\n")
    sys.stdout.write(
        "Stop the existing service with `uv run reflexio services stop` or choose "
        "different BACKEND_PORT/FRONTEND_PORT/DOCS_PORT values.\n"
    )
    sys.stdout.flush()
    sys.exit(1)


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
    return Path(tempfile.gettempdir()) / f"reflexio_services_{port_hash}.json"


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
    "embedding": ["Application startup complete"],
    "frontend": ["Ready in"],
    "docs": ["Ready in"],
}

# Extra env vars to suppress noise from subprocesses at source
_NOISE_SUPPRESSION_ENV: dict[str, dict[str, str]] = {
    "backend": {"LITELLM_LOG": "ERROR"},
    "embedding": {"LITELLM_LOG": "ERROR"},
    "frontend": {"NODE_NO_WARNINGS": "1"},
    "docs": {"NODE_NO_WARNINGS": "1"},
}

# Keep supervisor tuning internal until operational evidence justifies exposing it.
_SERVICE_HEALTHY_WINDOW_SECS = 30.0
_SERVICE_MAX_RAPID_FAILURES = 5
_SERVICE_RESPAWN_DELAY_SECS = 2.0


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


def _wait_for_all_ready(ready_events: dict[str, threading.Event]) -> bool:
    """Wait up to 60 seconds for every service to report readiness."""
    deadline = time.monotonic() + 60
    return all(
        event.wait(timeout=max(0, deadline - time.monotonic()))
        for event in ready_events.values()
    )


def _sleep_while_services_run(
    processes: dict[str, subprocess.Popen[bytes]],
) -> None:
    """Avoid a busy monitor loop while at least one child remains alive."""
    if processes:
        time.sleep(0.5)


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
    service_configs = {svc.name: svc for svc in services}
    recent_failures: dict[str, list[float]] = {}
    output_lock = threading.RLock()
    pidfile = get_pidfile_path(ports)
    shutting_down = False

    def write_current_pidfile() -> None:
        write_pidfile(pidfile, {name: proc.pid for name, proc in processes.items()})

    def start_service(svc: ServiceConfig, *, respawn: bool = False) -> None:
        noise_env = _NOISE_SUPPRESSION_ENV.get(svc.name, {})
        merged_env = {**os.environ, **(svc.env or {}), **noise_env}
        with output_lock:
            action = "Respawning" if respawn else "Starting"
            sys.stdout.write(f"{action} {svc.name}...\n")
            sys.stdout.flush()
        proc = subprocess.Popen(
            svc.command,
            cwd=svc.cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
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
        except (OSError, RuntimeError):
            processes.pop(svc.name, None)
            ready_events.pop(svc.name, None)
            with contextlib.suppress(OSError):
                proc.terminate()
            raise
        with output_lock:
            sys.stdout.write(f"  {svc.name} started (PID {proc.pid})\n")
            sys.stdout.flush()

    def shutdown(_signum: int | None = None, _frame: object = None) -> None:
        nonlocal shutting_down
        shutting_down = True
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

    def handle_service_exit(name: str, ret: int) -> None:
        with output_lock:
            sys.stdout.write(
                format_service_line(name, f"exited with code {ret}") + "\n"
            )
            sys.stdout.flush()
        del processes[name]
        ready_events.pop(name, None)
        write_current_pidfile()

        if ret == 0 or shutting_down:
            recent_failures.pop(name, None)
            return

        now = time.monotonic()
        failures = [
            timestamp
            for timestamp in recent_failures.get(name, [])
            if now - timestamp <= _SERVICE_HEALTHY_WINDOW_SECS
        ]
        failures.append(now)
        recent_failures[name] = failures
        if len(failures) >= _SERVICE_MAX_RAPID_FAILURES:
            with output_lock:
                sys.stdout.write(
                    format_service_line(
                        name,
                        f"marked degraded after {len(failures)} "
                        "rapid failures; will not respawn",
                    )
                    + "\n"
                )
                sys.stdout.flush()
            return

        time.sleep(_SERVICE_RESPAWN_DELAY_SECS)
        if shutting_down:
            return
        try:
            start_service(service_configs[name], respawn=True)
            write_current_pidfile()
        except (OSError, RuntimeError) as exc:
            ready_events.pop(name, None)
            with contextlib.suppress(KeyError, OSError):
                processes.pop(name).terminate()
            with output_lock:
                sys.stdout.write(
                    format_service_line(name, f"respawn failed; marked degraded: {exc}")
                    + "\n"
                )
                sys.stdout.flush()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    ensure_requested_ports_available(ports)

    for svc in services:
        start_service(svc)

    # Write pidfile
    write_current_pidfile()

    # Wait for all services to be ready (or timeout with shared deadline)
    if on_all_ready and _wait_for_all_ready(ready_events):
        on_all_ready(ports)

    # Wait for any child to exit
    try:
        while processes:
            for name, proc in list(processes.items()):
                ret = proc.poll()
                if ret is not None:
                    handle_service_exit(name, ret)
            _sleep_while_services_run(processes)
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
