"""Tests for service process utility helpers."""

from __future__ import annotations

import io
import signal
import tempfile
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
    monkeypatch.setattr(utils.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("REFLEXIO_SERVICE_HEALTHY_SECS", "30")
    monkeypatch.setenv("REFLEXIO_SERVICE_MAX_FAILS", "2")
    monkeypatch.setenv("REFLEXIO_SERVICE_RESPAWN_DELAY", "0")
    return pidfile


def test_pidfile_path_uses_platform_temp_dir() -> None:
    path = utils.get_pidfile_path({"backend": 8071})

    assert path.parent == Path(tempfile.gettempdir())
    assert str(path).startswith(tempfile.gettempdir())
    assert path.name.startswith("reflexio_services_")


def test_find_pids_on_port_queries_tcp_listeners_only(monkeypatch) -> None:
    calls: list[list[str]] = []

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
        return CompletedProcess(cmd, 0, stdout="123\nnot-a-pid\n456\n")

    monkeypatch.setattr(utils.subprocess, "run", fake_run)

    assert utils.find_pids_on_port(8090) == [123, 456]
    assert calls == [["lsof", "-nP", "-t", "-iTCP:8090", "-sTCP:LISTEN"]]


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
    _patch_run_services_environment(monkeypatch, tmp_path)
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

    assert [name for name, _proc in started] == ["crash", "sibling", "crash"]
    sibling = next(proc for name, proc in started if name == "sibling")
    assert sibling._poll_count == 10
    assert sibling.terminated is False
    assert "marked degraded after 2 rapid failures" in capsys.readouterr().out


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
def test_run_services_rejects_nonfinite_supervisor_env_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fall back safely when supervisor float settings are non-finite."""
    _patch_run_services_environment(monkeypatch, tmp_path)
    monkeypatch.setenv("REFLEXIO_SERVICE_HEALTHY_SECS", "nan")
    monkeypatch.setenv("REFLEXIO_SERVICE_RESPAWN_DELAY", "inf")
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
        [utils.ServiceConfig(name="backend", command=["backend"])],
        {"backend": 8071},
    )

    output = capsys.readouterr().out
    assert len(started) == 2
    assert "invalid REFLEXIO_SERVICE_HEALTHY_SECS='nan'" in output
    assert "invalid REFLEXIO_SERVICE_RESPAWN_DELAY='inf'" in output


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
