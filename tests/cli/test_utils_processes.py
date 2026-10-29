"""Process, pidfile and port-probe behaviour of ``reflexio.cli.utils``.

These tests use real child processes and real executables placed on PATH
(standing in for ``lsof``/``ss``/``pgrep`` where the host tool's output
would be nondeterministic), so signal delivery, process discovery and
pidfile handling run exactly as they do for ``reflexio services``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from reflexio.cli import utils

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX signal and process semantics"
)

_IGNORE_TERM = (
    "import signal, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "print('ready', flush=True)\n"
    "time.sleep(60)\n"
)


def _spawn(code: str, *args: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen(  # noqa: S603 - current interpreter, test-literal code.
        [sys.executable, "-c", code, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _spawn_ready(code: str, *args: str) -> subprocess.Popen[bytes]:
    proc = _spawn(code, *args)
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == b"ready"
    return proc


@pytest.fixture
def children():
    procs: list[subprocess.Popen[bytes]] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
        if proc.stdout is not None:
            proc.stdout.close()


def _fake_tool(bin_dir: Path, name: str, script: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(f"#!/bin/sh\n{script}\n")
    path.chmod(0o755)


class TestGetEnvPort:
    def test_unset_uses_default(self, monkeypatch) -> None:
        monkeypatch.delenv("REFLEXIO_TEST_PORT", raising=False)
        assert utils.get_env_port("REFLEXIO_TEST_PORT", 8061) == 8061

    def test_valid_value_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("REFLEXIO_TEST_PORT", "9123")
        assert utils.get_env_port("REFLEXIO_TEST_PORT", 8061) == 9123

    def test_invalid_value_warns_and_uses_default(self, monkeypatch, capsys) -> None:
        monkeypatch.setenv("REFLEXIO_TEST_PORT", "eighty")
        assert utils.get_env_port("REFLEXIO_TEST_PORT", 8061) == 8061
        assert (
            "Warning: invalid REFLEXIO_TEST_PORT='eighty', using default 8061"
            in capsys.readouterr().out
        )


class TestPortProbeToolsUnavailable:
    def test_no_lsof_or_ss_reports_no_holders(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
        assert utils.find_pids_on_port(8061) == []

    def test_failing_ss_reports_no_holders(self, monkeypatch, tmp_path) -> None:
        bin_dir = tmp_path / "bin"
        _fake_tool(bin_dir, "lsof", "exit 1")
        _fake_tool(bin_dir, "ss", "echo 'ss: permission denied' >&2; exit 2")
        monkeypatch.setenv("PATH", str(bin_dir))
        assert utils.find_pids_on_port(8061) == []

    def test_ss_rows_without_a_peer_column_are_ignored(
        self, monkeypatch, tmp_path
    ) -> None:
        bin_dir = tmp_path / "bin"
        _fake_tool(bin_dir, "lsof", "exit 1")
        _fake_tool(
            bin_dir,
            "ss",
            "echo 'garbled pid=111'\n"
            "echo 'LISTEN 0 128 127.0.0.1:8061 0.0.0.0:* users:((\"py\",pid=222,fd=3))'",
        )
        monkeypatch.setenv("PATH", str(bin_dir))
        assert utils.find_pids_on_port(8061) == [222]


class TestFindPidsByPattern:
    def test_finds_a_running_process_by_its_command_line(self, children) -> None:
        marker = f"reflexio-pgrep-{uuid.uuid4().hex}"
        proc = _spawn_ready(
            "import time; print('ready', flush=True); time.sleep(60)", marker
        )
        children.append(proc)
        if shutil.which("pgrep") is None:
            pytest.skip("pgrep is unavailable")
        assert proc.pid in utils.find_pids_by_pattern(marker)

    def test_no_match_returns_empty(self) -> None:
        if shutil.which("pgrep") is None:
            pytest.skip("pgrep is unavailable")
        assert utils.find_pids_by_pattern(f"no-such-{uuid.uuid4().hex}") == []

    def test_missing_pgrep_returns_empty(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
        assert utils.find_pids_by_pattern("anything") == []


class TestKillProcesses:
    def test_graceful_stop_terminates_a_cooperative_process(self, children) -> None:
        proc = _spawn_ready("import time; print('ready', flush=True); time.sleep(60)")
        children.append(proc)
        utils.kill_processes([proc.pid, proc.pid], graceful_timeout=0.2)
        assert proc.wait(timeout=10) == -signal.SIGTERM

    def test_term_ignoring_process_is_killed_after_the_grace_period(
        self, children
    ) -> None:
        proc = _spawn_ready(_IGNORE_TERM)
        children.append(proc)
        utils.kill_processes([proc.pid], graceful_timeout=0.3)
        assert proc.wait(timeout=10) == -signal.SIGKILL

    def test_force_sends_sigkill_immediately(self, children) -> None:
        proc = _spawn_ready(_IGNORE_TERM)
        children.append(proc)
        started = time.monotonic()
        utils.kill_processes([proc.pid], graceful_timeout=30, force=True)
        assert proc.wait(timeout=10) == -signal.SIGKILL
        assert time.monotonic() - started < 10

    def test_never_signals_itself_and_tolerates_missing_pids(self, children) -> None:
        gone = _spawn("pass")
        children.append(gone)
        gone.wait(timeout=10)
        # Our own PID is filtered out; an already-reaped PID is ignored.
        utils.kill_processes([os.getpid()])
        utils.kill_processes([gone.pid], graceful_timeout=0)

    def test_empty_list_is_a_noop(self) -> None:
        utils.kill_processes([])

    @pytest.mark.skipif(
        not hasattr(os, "geteuid") or os.geteuid() == 0,
        reason="needs an unprivileged user so signalling PID 1 is refused",
    )
    def test_permission_denied_is_reported_not_raised(self, capsys) -> None:
        utils.kill_processes([1], graceful_timeout=0)
        assert "Warning: no permission to kill PID 1" in capsys.readouterr().out


class TestPidfiles:
    def test_round_trip_and_removal(self, tmp_path) -> None:
        pidfile = tmp_path / "pids.json"
        utils.write_pidfile(pidfile, {"backend": 101, "docs": 202})
        assert utils.read_pidfile(pidfile) == {"backend": 101, "docs": 202}
        utils.remove_pidfile(pidfile)
        assert not pidfile.exists()
        utils.remove_pidfile(pidfile)  # already gone: no error

    def test_missing_pidfile_reads_empty(self, tmp_path) -> None:
        assert utils.read_pidfile(tmp_path / "absent.json") == {}

    def test_corrupt_pidfile_reads_empty(self, tmp_path) -> None:
        pidfile = tmp_path / "pids.json"
        pidfile.write_text("{not json")
        assert utils.read_pidfile(pidfile) == {}

    def test_stop_request_path_is_per_service_and_port(self) -> None:
        a = utils.get_stop_request_path("backend", 8061)
        b = utils.get_stop_request_path("backend", 8062)
        assert a != b
        assert a.parent == Path(tempfile.gettempdir())
        assert a.name == "reflexio_service_backend_8061.stop"


class TestStopServicesNothingRunning:
    def test_reports_not_running_and_leaves_stop_intent(
        self, monkeypatch, tmp_path, capsys
    ) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        bin_dir = tmp_path / "bin"
        _fake_tool(bin_dir, "lsof", "exit 1")
        _fake_tool(bin_dir, "ss", "exit 0")
        _fake_tool(bin_dir, "pgrep", "exit 1")
        monkeypatch.setenv("PATH", str(bin_dir))

        utils.stop_services({"docs": 8062}, {"docs": "next dev"})

        assert "docs (port 8062) not running" in capsys.readouterr().out
        assert (tmp_path / "reflexio_service_docs_8062.stop").exists()


class TestStreamOutput:
    def test_blank_lines_are_skipped_and_ready_fires_once(self, capsys) -> None:
        proc = _spawn(
            "print('booting'); print(''); print('Application startup complete');"
            " print('Application startup complete again')"
        )
        ready = threading.Event()
        try:
            utils._stream_output(proc, "backend", threading.Lock(), ready)
        finally:
            proc.wait(timeout=10)
            assert proc.stdout is not None
            proc.stdout.close()
        out = capsys.readouterr().out.splitlines()
        assert ready.is_set()
        assert len(out) == 3
        assert all(line.startswith("[backend ]") for line in out)
        assert out[0].endswith("booting")


def _supervisor(tmp_path: Path, configs: dict[str, utils.ServiceConfig]):
    return utils._ServiceSupervisor(
        service_configs=configs,
        pidfile=tmp_path / "services.json",
        stop_request_paths={name: tmp_path / f"{name}.stop" for name in configs},
    )


class TestSupervisorRespawn:
    def test_future_respawn_waits(self, tmp_path, capsys) -> None:
        sup = _supervisor(
            tmp_path, {"backend": utils.ServiceConfig("backend", ["true"])}
        )
        sup.pending_respawns["backend"] = time.monotonic() + 60
        sup.respawn_due_services()
        assert "backend" in sup.pending_respawns
        assert sup.processes == {}
        assert capsys.readouterr().out == ""

    def test_shutdown_cancels_pending_respawns(self, tmp_path) -> None:
        sup = _supervisor(
            tmp_path, {"backend": utils.ServiceConfig("backend", ["true"])}
        )
        sup.pending_respawns["backend"] = 0.0
        sup.shutting_down = True
        sup.respawn_due_services()
        assert sup.pending_respawns == {}
        assert sup.processes == {}

    def test_stop_request_cancels_a_due_respawn(self, tmp_path) -> None:
        sup = _supervisor(
            tmp_path, {"backend": utils.ServiceConfig("backend", ["true"])}
        )
        stop = tmp_path / "backend.stop"
        stop.touch()
        sup.pending_respawns["backend"] = 0.0
        sup.respawn_due_services()
        assert sup.processes == {}
        assert sup.pending_respawns == {}
        assert not stop.exists()

    def test_failed_respawn_marks_service_degraded(self, tmp_path, capsys) -> None:
        missing = str(tmp_path / "no-such-binary")
        sup = _supervisor(
            tmp_path, {"backend": utils.ServiceConfig("backend", [missing])}
        )
        sup.pending_respawns["backend"] = 0.0
        sup.respawn_due_services()
        out = capsys.readouterr().out
        assert "Respawning backend..." in out
        assert "respawn failed; marked degraded" in out
        assert sup.processes == {}
        assert sup.ready_events == {}
        assert sup.pending_respawns == {}

    def test_repeated_shutdown_request_does_not_raise_again(self, tmp_path) -> None:
        sup = _supervisor(tmp_path, {})
        with pytest.raises(KeyboardInterrupt):
            sup.request_shutdown(signal.SIGTERM)
        sup.request_shutdown(signal.SIGTERM)  # already shutting down
        assert sup.shutting_down


class TestSupervisorStop:
    def test_term_ignoring_child_is_killed(self, tmp_path, children) -> None:
        stubborn = _IGNORE_TERM.replace("'ready'", "'Application startup complete'")
        sup = _supervisor(
            tmp_path,
            {
                "backend": utils.ServiceConfig(
                    "backend", [sys.executable, "-c", stubborn]
                )
            },
        )
        sup.start_service(sup.service_configs["backend"])
        proc = sup.processes["backend"]
        children.append(proc)
        # The readiness line is printed only after SIGTERM is ignored.
        assert sup.ready_events["backend"].wait(timeout=10)
        sup.stop_started_services()
        assert proc.wait(timeout=10) == -signal.SIGKILL

    def test_thread_start_failure_terminates_the_child(
        self, tmp_path, monkeypatch, children
    ) -> None:
        sup = _supervisor(
            tmp_path,
            {
                "backend": utils.ServiceConfig(
                    "backend", [sys.executable, "-c", "import time; time.sleep(60)"]
                )
            },
        )
        spawned: list[subprocess.Popen[bytes]] = []
        real_popen: Any = subprocess.Popen

        def recording_popen(*args, **kwargs) -> subprocess.Popen[bytes]:
            proc: subprocess.Popen[bytes] = real_popen(*args, **kwargs)
            spawned.append(proc)
            children.append(proc)
            return proc

        def refuse_start(self) -> None:
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(utils.subprocess, "Popen", recording_popen)
        monkeypatch.setattr(utils.threading.Thread, "start", refuse_start)
        with pytest.raises(RuntimeError, match="can't start new thread"):
            sup.start_service(sup.service_configs["backend"])
        assert sup.processes == {}
        assert sup.ready_events == {}
        assert spawned[0].wait(timeout=10) == -signal.SIGTERM
