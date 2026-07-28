"""Tests for service process utility helpers."""

from __future__ import annotations

import io
import os
import shutil
import signal
import socket
import tempfile
import threading
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from reflexio.cli import utils


class _FakeProcess:
    def __init__(
        self, *, pid: int, polls_to_exit: int | None, returncode: int = 0
    ) -> None:
        self.pid = pid
        self.stdout = io.BytesIO()
        self._polls_to_exit = polls_to_exit
        self._returncode = returncode
        self._poll_count = 0
        self.terminated = False

    def poll(self) -> int | None:
        self._poll_count += 1
        if self._polls_to_exit is not None and self._poll_count >= self._polls_to_exit:
            return self._returncode
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return self._returncode

    def kill(self) -> None:
        self.terminated = True


def _patch_run_services_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Path:
    pidfile = tmp_path / "services.json"
    monkeypatch.setattr(utils, "ensure_requested_ports_available", lambda _ports: None)
    monkeypatch.setattr(utils, "get_pidfile_path", lambda _ports: pidfile)
    monkeypatch.setattr(
        utils,
        "get_stop_request_path",
        lambda name, port: tmp_path / f"{name}_{port}.stop",
    )
    monkeypatch.setattr(utils.signal, "signal", lambda *_args: None)
    clock = [0.0]
    monkeypatch.setattr(utils.time, "monotonic", lambda: clock[0])

    def advance_time(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(
        utils.time,
        "sleep",
        advance_time,
    )
    return pidfile


def test_pidfile_path_uses_platform_temp_dir() -> None:
    path = utils.get_pidfile_path({"backend": 8071})

    assert path.parent == Path(tempfile.gettempdir())
    assert str(path).startswith(tempfile.gettempdir())
    assert path.name.startswith("reflexio_services_")


def test_find_pids_on_port_reports_bound_sockets_not_clients(monkeypatch) -> None:
    calls: list[list[str]] = []
    # One listener (123), one orphaned bound-but-not-listening socket (456),
    # and one client merely connected to the port (789) which must be excluded.
    lsof_output = (
        "p123\n"
        "f13\n"
        "n*:8090\n"
        "p456\n"
        "f3\n"
        "n127.0.0.1:8090\n"
        "p789\n"
        "f91\n"
        "n127.0.0.1:55006->127.0.0.1:8090\n"
    )

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> CompletedProcess[str]:
        calls.append(cmd)
        assert capture_output is True
        assert text is True
        assert check is False
        return CompletedProcess(cmd, 0, stdout=lsof_output)

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert utils.find_pids_on_port(8090) == [123, 456]
    assert calls == [["lsof", "-nP", "-Fpn", "-iTCP:8090"]]


def test_find_pids_on_port_ignores_other_ports_with_same_suffix(monkeypatch) -> None:
    def fake_run(cmd: list[str], **_kwargs) -> CompletedProcess[str]:
        return CompletedProcess(cmd, 0, stdout="p123\nn*:18090\n")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert utils.find_pids_on_port(8090) == []


def test_find_pids_on_port_falls_back_to_ss(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs) -> CompletedProcess[str]:
        calls.append(cmd)
        if cmd[0] == "lsof":
            return CompletedProcess(cmd, 1, stdout="")
        return CompletedProcess(
            cmd,
            0,
            stdout=(
                "CLOSE 0 0 127.0.0.1:8090 0.0.0.0:* "
                'users:(("python",pid=456,fd=7),("python",pid=456,fd=8))\n'
                "ESTAB 0 0 127.0.0.1:8090 127.0.0.1:55006 "
                'users:(("client",pid=789,fd=9))\n'
            ),
        )

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert utils.find_pids_on_port(8090) == [456]
    assert calls == [
        ["lsof", "-nP", "-Fpn", "-iTCP:8090"],
        ["ss", "-tanpH", "sport", "=", ":8090"],
    ]


@pytest.mark.skipif(
    shutil.which("lsof") is None and shutil.which("ss") is None,
    reason="neither lsof nor ss is available",
)
def test_find_pids_on_port_detects_bound_socket_without_listen() -> None:
    # Regression: an orphaned process can hold a port bound without listening
    # (e.g. a leaked uvicorn --reload worker); it must still be detected.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        assert os.getpid() in utils.find_pids_on_port(port)


def test_requested_port_conflicts_detect_duplicate_service_ports(
    monkeypatch,
) -> None:
    monkeypatch.setattr(utils, "find_pids_on_port", lambda _port: [])

    conflicts = utils.get_requested_port_conflicts(
        {"frontend": 8090, "docs": 8090, "backend": 8091}
    )

    assert conflicts == ["port 8090 is assigned to multiple services: docs, frontend"]


def test_requested_port_conflicts_detect_occupied_ports(monkeypatch) -> None:
    monkeypatch.setattr(
        utils,
        "find_pids_on_port",
        lambda port: [123, 456] if port == 8090 else [],
    )

    conflicts = utils.get_requested_port_conflicts({"frontend": 8090, "docs": 8092})

    assert conflicts == ["frontend port 8090 is already in use by PID(s): 123, 456"]


@pytest.mark.unit
def test_run_services_respawns_unexpected_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Respawn a service after an unexpected non-zero exit."""
    pidfile = _patch_run_services_environment(monkeypatch, tmp_path)
    pidfile_writes: list[tuple[Path, dict[str, int]]] = []
    original_write_pidfile = utils.write_pidfile

    def record_pidfile(path: Path, pids: dict[str, int]) -> None:
        original_write_pidfile(path, pids)
        pidfile_writes.append((path, pids.copy()))

    monkeypatch.setattr(utils, "write_pidfile", record_pidfile)
    started: list[_FakeProcess] = []

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        proc = _FakeProcess(
            pid=1000 + len(started),
            polls_to_exit=1,
            returncode=1 if not started else 0,
        )
        started.append(proc)
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [utils.ServiceConfig(name="backend", command=["fake-backend"])],
        {"backend": 8071},
    )

    assert len(started) == 2
    assert started[0].pid != started[1].pid
    assert (pidfile, {"backend": started[1].pid}) in pidfile_writes


@pytest.mark.unit
def test_run_services_cleans_up_when_initial_service_start_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Terminate earlier children when a later initial launch fails."""
    pidfile = _patch_run_services_environment(monkeypatch, tmp_path)
    started: list[_FakeProcess] = []

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        if started:
            raise OSError("simulated initial spawn failure")
        proc = _FakeProcess(pid=1000, polls_to_exit=None)
        started.append(proc)
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    with pytest.raises(OSError, match="simulated initial spawn failure"):
        utils.run_services(
            [
                utils.ServiceConfig(name="backend", command=["backend"]),
                utils.ServiceConfig(name="embedding", command=["embedding"]),
            ],
            {"backend": 8071, "embedding": 8072},
        )

    assert started[0].terminated is True
    assert pidfile.exists() is False


@pytest.mark.unit
def test_wait_for_all_ready_returns_when_process_exits() -> None:
    """Do not wait for the full readiness timeout after a child exits."""
    ready_event = threading.Event()
    process = _FakeProcess(pid=1000, polls_to_exit=1, returncode=1)

    assert (
        utils._wait_for_all_ready({"backend": ready_event}, {"backend": process})
        is False
    )
    assert process._poll_count == 1


@pytest.mark.unit
def test_run_services_calls_on_all_ready_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Invoke the readiness callback once after all services are ready."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(utils, "_wait_for_all_ready", lambda _events, _processes: True)
    started: list[_FakeProcess] = []
    ready_calls: list[dict[str, int]] = []

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        proc = _FakeProcess(pid=1000 + len(started), polls_to_exit=1, returncode=0)
        started.append(proc)
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [utils.ServiceConfig(name="backend", command=["backend"])],
        {"backend": 8071},
        on_all_ready=lambda ports: ready_calls.append(ports),
    )

    assert ready_calls == [{"backend": 8071}]


@pytest.mark.unit
def test_run_services_does_not_respawn_an_explicitly_stopped_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Honor a stop request written by a separate stop-services command."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(utils, "_wait_for_all_ready", lambda _events, _processes: True)
    started: list[_FakeProcess] = []

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        proc = _FakeProcess(pid=1000 + len(started), polls_to_exit=1, returncode=1)
        started.append(proc)
        return proc

    def request_stop(_ports: dict[str, int]) -> None:
        utils.get_stop_request_path("backend", 8071).touch()

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [utils.ServiceConfig(name="backend", command=["backend"])],
        {"backend": 8071},
        on_all_ready=request_stop,
    )

    assert len(started) == 1


def test_stop_services_writes_intent_before_killing_saved_pids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pidfile = tmp_path / "services.json"
    pidfile.write_text('{"backend": 1234}')
    stop_requests: list[list[int]] = []
    stop_request_path = tmp_path / "backend_8071.stop"
    monkeypatch.setattr(utils, "get_pidfile_path", lambda _ports: pidfile)
    monkeypatch.setattr(
        utils,
        "get_stop_request_path",
        lambda _name, _port: stop_request_path,
    )
    monkeypatch.setattr(utils, "find_pids_on_port", lambda _port: [])
    monkeypatch.setattr(utils, "find_pids_by_pattern", lambda _pattern: [])

    def record_stop(pids: list[int], **_kwargs: object) -> None:
        assert stop_request_path.exists()
        stop_requests.append(pids)

    monkeypatch.setattr(
        utils,
        "kill_processes",
        record_stop,
    )

    utils.stop_services({"backend": 8071}, {"backend": "backend"})

    assert stop_requests == [[1234]]
    assert stop_request_path.exists()
    utils.remove_pidfile(stop_request_path)


@pytest.mark.unit
def test_run_services_gives_up_on_crash_loop_and_keeps_sibling_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Give up on one crash-looping service while supervising its sibling."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    started: list[tuple[str, _FakeProcess]] = []

    def fake_popen(command, **_kwargs) -> _FakeProcess:
        name = command[0]
        proc = _FakeProcess(
            pid=1000 + len(started),
            polls_to_exit=1 if name == "crash" else 10,
            returncode=1 if name == "crash" else 0,
        )
        started.append((name, proc))
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [
            utils.ServiceConfig(name="crash", command=["crash"]),
            utils.ServiceConfig(name="sibling", command=["sibling"]),
        ],
        {"crash": 8071, "sibling": 8072},
    )

    assert [name for name, _proc in started] == [
        "crash",
        "sibling",
        "crash",
        "crash",
        "crash",
        "crash",
    ]
    sibling = next(proc for name, proc in started if name == "sibling")
    assert sibling._poll_count == 10
    assert sibling.terminated is False
    assert "marked degraded after 5 rapid failures" in capsys.readouterr().out


@pytest.mark.unit
def test_run_services_does_not_respawn_during_shutdown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Do not restart children after the supervisor receives SIGTERM."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    handlers: dict[signal.Signals, object] = {}
    started: list[_FakeProcess] = []

    def capture_signal(signum, handler) -> None:
        handlers[signum] = handler

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        proc = _FakeProcess(pid=1000 + len(started), polls_to_exit=None)
        started.append(proc)
        return proc

    def request_shutdown(_seconds: float) -> None:
        handler = handlers[signal.SIGTERM]
        assert callable(handler)
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(utils.signal, "signal", capture_signal)
    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(utils.time, "sleep", request_shutdown)

    with pytest.raises(SystemExit):
        utils.run_services(
            [utils.ServiceConfig(name="backend", command=["backend"])],
            {"backend": 8071},
        )

    assert len(started) == 1
    assert started[0].terminated is True


@pytest.mark.unit
def test_run_services_does_not_respawn_clean_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Treat a clean exit as intentional and do not restart the service."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    started: list[_FakeProcess] = []

    def fake_popen(*_args, **_kwargs) -> _FakeProcess:
        proc = _FakeProcess(pid=1000 + len(started), polls_to_exit=1, returncode=0)
        started.append(proc)
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [utils.ServiceConfig(name="backend", command=["backend"])],
        {"backend": 8071},
    )

    assert len(started) == 1


@pytest.mark.unit
def test_run_services_drops_failed_respawn_without_stopping_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Drop a child when respawn fails while keeping its sibling supervised."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    started: list[tuple[str, _FakeProcess]] = []
    popen_calls = 0

    def fake_popen(command, **_kwargs) -> _FakeProcess:
        nonlocal popen_calls
        popen_calls += 1
        if popen_calls == 3:
            raise OSError("simulated spawn failure")
        name = command[0]
        proc = _FakeProcess(
            pid=1000 + len(started),
            polls_to_exit=1 if name == "crash" else 3,
            returncode=1 if name == "crash" else 0,
        )
        started.append((name, proc))
        return proc

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [
            utils.ServiceConfig(name="crash", command=["crash"]),
            utils.ServiceConfig(name="sibling", command=["sibling"]),
        ],
        {"crash": 8071, "sibling": 8072},
    )

    assert popen_calls == 3
    assert [name for name, _proc in started] == ["crash", "sibling"]
    sibling = next(proc for name, proc in started if name == "sibling")
    assert sibling._poll_count == 3
    assert sibling.terminated is False
    assert "respawn failed; marked degraded: simulated spawn failure" in (
        capsys.readouterr().out
    )
