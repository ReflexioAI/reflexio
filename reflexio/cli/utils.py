"""Shared utilities for service process management."""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

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
    """
    Find process IDs with a TCP socket bound to the given port using lsof.

    Matches listeners and bound-but-not-listening sockets (e.g. an orphaned
    uvicorn --reload worker holding the port in CLOSED state, which blocks
    rebinding but is invisible to a LISTEN-only check). Client connections to
    the port (sockets with a peer address) are excluded so unrelated processes
    that merely connected to the service are never reported or killed.

    Args:
        port (int): TCP port to inspect

    Returns:
        list[int]: Sorted, de-duplicated PIDs holding a socket bound to the port
    """
    return _pids_from_lsof(port) or _pids_from_ss(port)


def _pids_from_lsof(port: int) -> list[int]:
    """Find port holders via ``lsof`` (the only option on macOS)."""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-Fpn", f"-iTCP:{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        result = None

    pids: set[int] = set()
    if result is not None and result.returncode == 0:
        current_pid: int | None = None
        suffix = f":{port}"
        for line in result.stdout.splitlines():
            if line.startswith("p") and line[1:].isdigit():
                current_pid = int(line[1:])
            elif line.startswith("n") and current_pid is not None:
                name = line[1:]
                if "->" not in name and name.endswith(suffix):
                    pids.add(current_pid)
    if pids:
        return sorted(pids)

    return []


def _pids_from_ss(port: int) -> list[int]:
    """Find port holders via ``ss`` — the Linux half of the CLOSED-socket case.

    On Linux ``lsof -iTCP:<port>`` reports nothing for a socket that is bound
    but never ``listen()``ed, even though the kernel still rejects a competing
    bind with EADDRINUSE. ``ss`` shows it as ``UNCONN``, so without this the
    orphaned-worker case ``find_pids_on_port`` exists to diagnose is invisible
    on exactly the platform the servers run on. Absent on macOS, where ``lsof``
    already covers it.
    """
    try:
        result = subprocess.run(
            ["ss", "-tanpH", "sport", "=", f":{port}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []

    pids: set[int] = set()
    for line in result.stdout.splitlines():
        fields = line.split()
        # State Recv-Q Send-Q Local:Port Peer:Port users:((...)). A concrete
        # peer means the row is a connection that merely happens to use this
        # port as its source, not a holder of it — same exclusion as the
        # ``->`` check on the lsof path.
        if len(fields) < 5 or not fields[4].endswith(":*"):
            continue
        pids.update(int(pid) for pid in re.findall(r"pid=(\d+)", line))
    return sorted(pids)


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


def get_stop_request_path(service_name: str, port: int) -> Path:
    """Get the marker path used to communicate an intentional child stop."""
    return Path(tempfile.gettempdir()) / f"reflexio_service_{service_name}_{port}.stop"


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
_SERVICE_READY_TIMEOUT_SECS = 60.0
_SERVICE_READY_POLL_INTERVAL_SECS = 0.1
_SERVICE_MONITOR_POLL_INTERVAL_SECS = 0.5
_SERVICE_HEALTHY_WINDOW_SECS = 30.0
_SERVICE_MAX_RAPID_FAILURES = 5
_SERVICE_RESPAWN_DELAY_SECS = 2.0


class _PollableProcess(Protocol):
    def poll(self) -> int | None: ...


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


def _wait_for_all_ready(
    ready_events: dict[str, threading.Event],
    processes: Mapping[str, _PollableProcess],
) -> bool:
    """Wait for readiness, returning early when a child exits or times out."""
    deadline = time.monotonic() + _SERVICE_READY_TIMEOUT_SECS
    while time.monotonic() < deadline:
        if any(proc.poll() is not None for proc in processes.values()):
            return False
        if all(event.is_set() for event in ready_events.values()):
            return True
        remaining = deadline - time.monotonic()
        time.sleep(min(_SERVICE_READY_POLL_INTERVAL_SECS, remaining))
    return all(event.is_set() for event in ready_events.values())


@dataclasses.dataclass
class _ServiceSupervisor:
    service_configs: dict[str, ServiceConfig]
    pidfile: Path
    stop_request_paths: dict[str, Path]
    processes: dict[str, subprocess.Popen[bytes]] = dataclasses.field(
        default_factory=dict
    )
    threads: dict[str, threading.Thread] = dataclasses.field(default_factory=dict)
    ready_events: dict[str, threading.Event] = dataclasses.field(default_factory=dict)
    recent_failures: dict[str, list[float]] = dataclasses.field(default_factory=dict)
    pending_respawns: dict[str, float] = dataclasses.field(default_factory=dict)
    output_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    shutting_down: bool = False

    def write_output(self, message: str) -> None:
        with self.output_lock:
            sys.stdout.write(message + "\n")
            sys.stdout.flush()

    def write_current_pidfile(self) -> None:
        write_pidfile(
            self.pidfile, {name: proc.pid for name, proc in self.processes.items()}
        )

    def start_service(self, svc: ServiceConfig, *, respawn: bool = False) -> None:
        noise_env = _NOISE_SUPPRESSION_ENV.get(svc.name, {})
        merged_env = {**os.environ, **(svc.env or {}), **noise_env}
        action = "Respawning" if respawn else "Starting"
        self.write_output(f"{action} {svc.name}...")
        proc = subprocess.Popen(
            svc.command,
            cwd=svc.cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            self.processes[svc.name] = proc
            ready_event = threading.Event()
            self.ready_events[svc.name] = ready_event
            thread = threading.Thread(
                target=_stream_output,
                args=(proc, svc.name, self.output_lock, ready_event),
                daemon=True,
            )
            thread.start()
            self.threads[svc.name] = thread
        except (OSError, RuntimeError):
            self.processes.pop(svc.name, None)
            self.ready_events.pop(svc.name, None)
            with contextlib.suppress(OSError):
                proc.terminate()
            raise
        self.write_output(f"  {svc.name} started (PID {proc.pid})")

    def stop_started_services(self) -> None:
        for proc in self.processes.values():
            with contextlib.suppress(OSError):
                proc.terminate()
        deadline = time.time() + 3.0
        for proc in self.processes.values():
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(OSError):
                    proc.kill()
        for thread in self.threads.values():
            thread.join(timeout=1.0)

    def shutdown(self, _signum: int | None = None, _frame: object = None) -> None:
        self.shutting_down = True
        self.write_output("\nShutting down services...")
        self.stop_started_services()
        remove_pidfile(self.pidfile)
        for path in self.stop_request_paths.values():
            remove_pidfile(path)
        sys.exit(0)

    def handle_service_exit(self, name: str, ret: int) -> None:
        self.write_output(format_service_line(name, f"exited with code {ret}"))
        del self.processes[name]
        self.threads.pop(name, None)
        self.ready_events.pop(name, None)
        self.write_current_pidfile()

        stop_requested = self.stop_request_paths[name].exists()
        self.pending_respawns.pop(name, None)
        if ret == 0 or self.shutting_down or stop_requested:
            self.recent_failures.pop(name, None)
            remove_pidfile(self.stop_request_paths[name])
            return

        now = time.monotonic()
        failures = [
            timestamp
            for timestamp in self.recent_failures.get(name, [])
            if now - timestamp <= _SERVICE_HEALTHY_WINDOW_SECS
        ]
        failures.append(now)
        self.recent_failures[name] = failures
        if len(failures) >= _SERVICE_MAX_RAPID_FAILURES:
            self.write_output(
                format_service_line(
                    name,
                    f"marked degraded after {len(failures)} "
                    "rapid failures; will not respawn",
                )
            )
            return

        self.pending_respawns[name] = now + _SERVICE_RESPAWN_DELAY_SECS

    def respawn_due_services(self) -> None:
        if self.shutting_down:
            self.pending_respawns.clear()
            return
        now = time.monotonic()
        for name, respawn_at in list(self.pending_respawns.items()):
            if respawn_at > now:
                continue
            self.pending_respawns.pop(name)
            if self.stop_request_paths[name].exists():
                remove_pidfile(self.stop_request_paths[name])
                continue
            try:
                self.start_service(self.service_configs[name], respawn=True)
                self.write_current_pidfile()
            except (OSError, RuntimeError) as exc:
                self.ready_events.pop(name, None)
                proc = self.processes.pop(name, None)
                if proc is not None:
                    with contextlib.suppress(OSError):
                        proc.terminate()
                self.write_output(
                    format_service_line(name, f"respawn failed; marked degraded: {exc}")
                )


def run_services(
    services: list[ServiceConfig],
    ports: dict[str, int],
    *,
    on_all_ready: Callable[[dict[str, int]], None] | None = None,
) -> None:
    """Launch services, pipe output with prefixes, and manage lifecycle.

    Each service's stdout/stderr is captured and prefixed with a colored
    service tag (e.g., [backend]). Unexpected exits are respawned after a
    bounded delay; failure history expires outside the healthy window, and a
    service is marked degraded after reaching the rapid-failure limit. Clean
    exits, shutdowns, and explicitly requested stops are not respawned. A
    callback is invoked when all services report ready (or after a timeout).

    Args:
        services: List of service configurations to launch.
        ports: Port mapping for pidfile identification.
        on_all_ready: Optional callback invoked with ports when all services are ready.
    """
    pidfile = get_pidfile_path(ports)
    supervisor = _ServiceSupervisor(
        service_configs={svc.name: svc for svc in services},
        pidfile=pidfile,
        stop_request_paths={
            name: get_stop_request_path(name, ports[name]) for name in ports
        },
    )

    signal.signal(signal.SIGINT, supervisor.shutdown)
    signal.signal(signal.SIGTERM, supervisor.shutdown)

    ensure_requested_ports_available(ports)
    for path in supervisor.stop_request_paths.values():
        remove_pidfile(path)

    service_names = {svc.name for svc in services}
    gate_local_embedding = {"embedding", "backend"}.issubset(service_names)

    try:
        if gate_local_embedding:
            embedding = next(svc for svc in services if svc.name == "embedding")
            supervisor.start_service(embedding)
            if not _wait_for_all_ready(
                {"embedding": supervisor.ready_events["embedding"]},
                {"embedding": supervisor.processes["embedding"]},
            ):
                raise RuntimeError(
                    "embedding service did not become ready before backend startup"
                )

        for svc in services:
            if gate_local_embedding and svc.name == "embedding":
                continue
            supervisor.start_service(svc)
        supervisor.write_current_pidfile()
    except (OSError, RuntimeError):
        supervisor.stop_started_services()
        remove_pidfile(pidfile)
        for path in supervisor.stop_request_paths.values():
            remove_pidfile(path)
        raise

    # Wait for all services to be ready (or timeout with shared deadline)
    if on_all_ready and _wait_for_all_ready(
        supervisor.ready_events, supervisor.processes
    ):
        on_all_ready(ports)

    # Wait for any child to exit
    try:
        while supervisor.processes or supervisor.pending_respawns:
            supervisor.respawn_due_services()
            for name, proc in list(supervisor.processes.items()):
                ret = proc.poll()
                if ret is not None:
                    supervisor.handle_service_exit(name, ret)
            if supervisor.processes or supervisor.pending_respawns:
                time.sleep(_SERVICE_MONITOR_POLL_INTERVAL_SECS)
    except KeyboardInterrupt:
        supervisor.shutdown()

    remove_pidfile(pidfile)
    for path in supervisor.stop_request_paths.values():
        remove_pidfile(path)


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
        stop_request_path = get_stop_request_path(name, port)
        stop_request_path.touch()
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
