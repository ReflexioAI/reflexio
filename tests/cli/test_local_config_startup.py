"""Run the real startup preparation before strict discovery and process launch."""

import json
import os
import socket
from pathlib import Path

import pytest

from reflexio.cli import run_services
from reflexio.cli.bootstrap_config import default_config_path
from reflexio.config_document import decode_config_document
from reflexio.config_file import LocalConfigFile


@pytest.fixture
def saved_local_config(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("REFLEXIO_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("REFLEXIO_DEFAULT_ORG_ID", "startup-owner")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("REFLEXIO_STORAGE", raising=False)
    monkeypatch.setattr(run_services, "load_reflexio_env", lambda: None)
    fixture = Path(__file__).parents[1] / "fixtures/local_config_generation0.json"
    data = json.loads(fixture.read_text())
    data["storage_config"]["db_path"] = str(tmp_path / "owned.sqlite")
    path = default_config_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize(
    "listener_host,requested_host",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("0.0.0.0", "0.0.0.0"),  # noqa: S104 - short-lived owned socket, no accepts
        ("127.0.0.1", "localhost"),
    ],
)
@pytest.mark.parametrize("launcher", ["cli", "module"])
def test_occupied_backend_never_prepares_when_pid_tools_cannot_see_it(
    saved_local_config, monkeypatch, listener_host, requested_host, launcher
):
    original = saved_local_config.read_bytes()
    monkeypatch.setattr("reflexio.cli.utils.find_pids_on_port", lambda _port: [])
    monkeypatch.setattr(
        run_services, "run_services", lambda *_args: pytest.fail("must refuse launch")
    )
    with socket.socket() as listener:
        listener.bind((listener_host, 0))
        listener.listen()
        args = run_services._build_run_services_parser().parse_args(
            [
                "--only",
                "backend",
                "--backend-host",
                requested_host,
                "--backend-port",
                str(listener.getsockname()[1]),
                "--no-reload",
                "--workers",
                "1",
            ]
        )
        with pytest.raises(SystemExit):
            if launcher == "cli":
                run_services.execute(args)
            else:
                from reflexio.server.__main__ import main

                monkeypatch.setattr(
                    "reflexio.server.__main__.uvicorn.run",
                    lambda *_args, **_kwargs: pytest.fail("must refuse launch"),
                )
                main(
                    ["--host", requested_host, "--port", str(listener.getsockname()[1])]
                )
    assert saved_local_config.read_bytes() == original
    assert not (saved_local_config.parent / ".config-migrations").exists()


@pytest.mark.parametrize("tagged_sqlite", [False, True])
@pytest.mark.parametrize("explicit_storage", [None, "sqlite"])
def test_ordinary_restart_preserves_database_and_unrelated_raw_fields(
    saved_local_config,
    monkeypatch,
    explicit_storage,
    tagged_sqlite,
):
    if not tagged_sqlite:
        original = json.loads(saved_local_config.read_text())
        original["storage_config"].pop("type")
        saved_local_config.write_text(json.dumps(original))
    before = decode_config_document(saved_local_config.read_text())
    launches = []
    monkeypatch.setattr(
        run_services, "run_services", lambda *args: launches.append(args)
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = run_services._build_run_services_parser().parse_args(
        [
            "--only",
            "backend",
            "--backend-port",
            str(port),
            "--no-reload",
            "--workers",
            "1",
        ]
    )
    run_services.execute(args, storage=explicit_storage)
    logical = LocalConfigFile(saved_local_config).read_document()
    assert logical["storage_config"] == before["storage_config"]
    expected = dict(before)
    optimizer = dict(expected["playbook_optimizer_config"])
    optimizer.pop("optimize_agent_playbooks")
    optimizer.pop("auto_update_pending_agent_playbooks")
    optimizer["agent_playbooks_enabled"] = False
    enabled = optimizer.pop("enabled")
    limit = optimizer.pop("max_metric_calls")
    optimizer["metric_call_budget"] = limit if enabled else 0
    expected["playbook_optimizer_config"] = optimizer
    floor = dict(expected["retrieval_floor"])
    assert floor.pop("enabled") is False
    assert floor.pop("pool_size") == 30
    floor["candidate_pool_size"] = 0
    expected["retrieval_floor"] = floor
    # Generation-zero disabled audit retention must remain disabled after its
    # explicit age-owned policy conversion, just like the other retired fields.
    audit = dict(expected["governance_retention"])
    assert audit.pop("audit_events_retention_enabled") is False
    assert audit.pop("audit_events_retention_days") == 365
    audit["audit_events_max_age_days"] = 0
    expected["governance_retention"] = audit
    assert logical == expected
    first = saved_local_config.read_bytes()
    run_services.execute(args, storage=explicit_storage)
    assert saved_local_config.read_bytes() == first
    assert len(launches) == 2


def test_real_server_upgrades_before_serving_and_keeps_owned_dataset(
    saved_local_config,
    tmp_path,
):
    """Exercise the installed entrypoint, strict reader and real HTTP response."""
    import os
    import sqlite3
    import subprocess
    import sys
    import time
    import urllib.request

    from cryptography.fernet import Fernet

    original = saved_local_config.read_bytes()
    document = decode_config_document(original.decode())
    database = document["storage_config"]["db_path"]
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE upgrade_identity (value TEXT)")
        connection.execute("INSERT INTO upgrade_identity VALUES ('same-owned-dataset')")
    empty_env = tmp_path / "empty.env"
    empty_env.touch()
    env = {
        key: value
        for key, value in os.environ.items()
        if key in {"PATH", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT"}
    }
    env.update(
        HOME=str(tmp_path),
        USERPROFILE=str(tmp_path),
        REFLEXIO_LOG_DIR=str(tmp_path),
        REFLEXIO_ENV_FILE=str(empty_env),
        REFLEXIO_DEFAULT_ORG_ID="startup-owner",
        LOCAL_STORAGE_PATH=str(tmp_path / "sqlite"),
        REFLEXIO_STORAGE="sqlite",
        PYTHONPATH=str(Path(__file__).parents[2]),
        FERNET_KEYS=Fernet.generate_key().decode(),
        FERNET_REQUIRED="true",
        REFLEXIO_NATIVE_COMPUTE_THREADS="2",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OPENAI_API_KEY="synthetic-no-provider-access",
        LITELLM_LOCAL_MODEL_COST_MAP="true",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        REFLEXIO_EMBEDDING_SERVICE_URL="http://127.0.0.1:9",
    )
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # No provider call or model download may escape the isolated server.
    wrapper = """import sys, runpy, ipaddress

def audit(event, args):
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and not ipaddress.ip_address(address[0]).is_loopback:
            raise RuntimeError("Upgrade regression refuses non-loopback network")

sys.addaudithook(audit)
sys.argv = ["reflexio.server", "--port", sys.argv[1], "--workers", "1"]
runpy.run_module("reflexio.server", run_name="__main__")
"""
    log = tmp_path / "server.log"
    with log.open("w") as output:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and source, isolated env
            [sys.executable, "-c", wrapper, str(port)],
            env=env,
            cwd=tmp_path,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                assert process.poll() is None, log.read_text()[-8000:]
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        assert response.status == 200
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                pytest.fail(f"Server readiness timed out: {log.read_text()[-8000:]}")
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/get_config", timeout=10
            ) as response:
                config = json.load(response)
            optimizer = config["playbook_optimizer_config"]
            assert optimizer["agent_playbooks_enabled"] is False
            assert optimizer["user_playbooks_enabled"] is True
            assert optimizer["metric_call_budget"] == 7
            assert "optimize_agent_playbooks" not in optimizer
            assert "auto_update_pending_agent_playbooks" not in optimizer
            assert config["storage_config"]["db_path"] == database
            assert "_reflexio_config_file" not in config
            with sqlite3.connect(database) as connection:
                assert connection.execute(
                    "SELECT value FROM upgrade_identity"
                ).fetchone() == ("same-owned-dataset",)
            backups = list(
                (
                    saved_local_config.parent
                    / ".config-migrations"
                    / saved_local_config.name
                ).glob("*.json")
            )
            assert len(backups) == 1
            assert backups[0].read_bytes() == original
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX reuse semantics; Windows requires exclusive address ownership",
)
def test_recently_closed_connections_do_not_block_a_valid_restart(monkeypatch):
    from reflexio.cli.utils import ensure_requested_ports_available

    monkeypatch.setattr("reflexio.cli.utils.find_pids_on_port", lambda _port: [])
    with socket.socket() as previous:
        previous.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        previous.bind(("127.0.0.1", 0))
        previous.listen()
        port = previous.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port)) as client:
            accepted, _address = previous.accept()
            # The server actively closes first, retaining TIME_WAIT on its port.
            accepted.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
            accepted.close()
    # Uvicorn's ordinary address reuse can actually bind and listen again.
    with socket.socket() as replacement:
        replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        replacement.bind(("127.0.0.1", port))
        replacement.listen()
    ensure_requested_ports_available({"backend": port})


@pytest.mark.parametrize(
    "host,ipv4,available",
    [("localhost", True, True), ("localhost", False, False), ("::1", False, False)],
)
def test_unavailable_ipv6_requires_a_verified_applicable_address(
    host, ipv4, available, monkeypatch
):
    import errno

    from reflexio.cli import utils

    original_socket = socket.socket
    addresses: list[
        tuple[int, int, int, str, tuple[str, int] | tuple[str, int, int, int]]
    ] = [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", 0, 0, 0))]
    if ipv4:
        addresses.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0)))

    def socket_with_disabled_ipv6(family, *args):
        if family == socket.AF_INET6:
            raise OSError(errno.EAFNOSUPPORT, "IPv6 disabled")
        return original_socket(family, *args)

    monkeypatch.setattr(socket, "getaddrinfo", lambda *_args, **_kwargs: addresses)
    monkeypatch.setattr(socket, "socket", socket_with_disabled_ipv6)
    monkeypatch.setattr(utils, "find_pids_on_port", lambda _port: [])
    assert (
        bool(utils.get_requested_port_conflicts({"backend": 0}, host=host))
        is not available
    )
