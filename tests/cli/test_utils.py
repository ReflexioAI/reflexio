"""Tests for service process utility helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path
from subprocess import CompletedProcess

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
