"""Tests for service process utility helpers."""

from __future__ import annotations

import tempfile
from email.message import Message
from pathlib import Path
from subprocess import CompletedProcess
from urllib.error import HTTPError

import pytest

from reflexio.cli import utils


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


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_service_port_healthy_uses_service_health_path(monkeypatch) -> None:
    calls: list[tuple[str, float]] = []

    def fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        calls.append((request.full_url, timeout))  # type: ignore[attr-defined]
        return _FakeResponse(200)

    monkeypatch.setattr(utils.urllib.request, "urlopen", fake_urlopen)

    assert utils.service_port_healthy("backend", 8061, timeout=0.25)
    assert calls == [("http://127.0.0.1:8061/health", 0.25)]


def test_service_port_healthy_rejects_unhealthy_status(monkeypatch) -> None:
    def fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        raise HTTPError(
            request.full_url,  # type: ignore[attr-defined]
            503,
            "Service unavailable",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr(utils.urllib.request, "urlopen", fake_urlopen)

    assert not utils.service_port_healthy("backend", 8061)


def test_service_port_healthy_does_not_guess_without_health_path(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not probe unknown health path")
        ),
    )

    assert not utils.service_port_healthy("docs", 8062)
    assert not utils.service_port_healthy("frontend", 8063)


def test_service_port_healthy_handles_connection_refused(monkeypatch) -> None:
    def fake_urlopen(request: object, *, timeout: float) -> _FakeResponse:
        raise ConnectionRefusedError("refused")

    monkeypatch.setattr(utils.urllib.request, "urlopen", fake_urlopen)

    assert not utils.service_port_healthy("backend", 8061)


def test_skip_healthy_services_filters_only_healthy_requested_services(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        utils,
        "service_port_healthy",
        lambda name, _port: name == "backend",
    )
    backend = utils.ServiceConfig("backend", ["backend"])
    docs = utils.ServiceConfig("docs", ["docs"])

    services, ports, skipped = utils.skip_healthy_services(
        [backend, docs],
        {"backend": 8061, "docs": 8062},
    )

    assert services == [docs]
    assert ports == {"docs": 8062}
    assert skipped == ["backend"]


def test_skip_healthy_services_probes_health_at_filter_time(
    monkeypatch,
) -> None:
    calls: list[tuple[str, int]] = []

    def fake_service_port_healthy(name: str, port: int) -> bool:
        calls.append((name, port))
        return False

    monkeypatch.setattr(
        utils,
        "service_port_healthy",
        fake_service_port_healthy,
    )
    backend = utils.ServiceConfig("backend", ["backend"])

    services, ports, skipped = utils.skip_healthy_services(
        [backend],
        {"backend": 8061},
    )

    assert services == [backend]
    assert ports == {"backend": 8061}
    assert skipped == []
    assert calls == [("backend", 8061)]


def test_skip_healthy_services_handles_service_without_port_entry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        utils,
        "service_port_healthy",
        lambda _name, _port: (_ for _ in ()).throw(
            AssertionError("should not probe missing port")
        ),
    )
    orphan = utils.ServiceConfig("orphan", ["orphan"])

    services, ports, skipped = utils.skip_healthy_services([orphan], {})

    assert services == [orphan]
    assert ports == {}
    assert skipped == []


def test_run_services_skip_if_running_returns_without_conflict_or_spawn(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(utils, "service_port_healthy", lambda _name, _port: True)
    monkeypatch.setattr(
        utils,
        "ensure_requested_ports_available",
        lambda _ports: (_ for _ in ()).throw(AssertionError("should not check ports")),
    )
    monkeypatch.setattr(
        utils.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not spawn")
        ),
    )

    utils.run_services(
        [utils.ServiceConfig("backend", ["backend"])],
        {"backend": 8061},
        skip_if_running=True,
    )

    assert capsys.readouterr().out == "Services already running: backend\n"


def test_run_services_skip_if_running_spawns_only_unhealthy_service(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        utils,
        "service_port_healthy",
        lambda name, _port: name == "backend",
    )
    monkeypatch.setattr(utils, "ensure_requested_ports_available", lambda _ports: None)
    monkeypatch.setattr(utils, "write_pidfile", lambda _pidfile, _data: None)
    monkeypatch.setattr(utils, "remove_pidfile", lambda _pidfile: None)
    monkeypatch.setattr(utils.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(utils.time, "sleep", lambda _seconds: None)
    spawned: list[str] = []

    class _FakeStdout:
        def __iter__(self):
            return iter(())

    class _FakeProc:
        pid = 1234
        stdout = _FakeStdout()

        def poll(self) -> int:
            return 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def fake_popen(cmd: list[str], **_kwargs: object) -> _FakeProc:
        spawned.append(cmd[0])
        return _FakeProc()

    monkeypatch.setattr(utils.subprocess, "Popen", fake_popen)

    utils.run_services(
        [
            utils.ServiceConfig("backend", ["backend"]),
            utils.ServiceConfig("docs", ["docs"]),
        ],
        {"backend": 8061, "docs": 8062},
        skip_if_running=True,
    )

    assert spawned == ["docs"]
    assert "Services already running: backend" in capsys.readouterr().out


def test_run_services_skip_if_running_rejects_duplicate_ports_before_health(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(utils, "service_port_healthy", lambda _name, _port: True)
    monkeypatch.setattr(
        utils.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("should not spawn")
        ),
    )

    with pytest.raises(SystemExit):
        utils.run_services(
            [
                utils.ServiceConfig("backend", ["backend"]),
                utils.ServiceConfig("docs", ["docs"]),
            ],
            {"backend": 8061, "docs": 8061},
            skip_if_running=True,
        )

    assert (
        "port 8061 is assigned to multiple services: backend, docs"
        in capsys.readouterr().out
    )
