"""CLI admission survives a busy execution slot over real TCP and SQLite."""

import json
import socket
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
import uvicorn
from fastapi import FastAPI

from reflexio import ReflexioClient
from reflexio.cli.commands import agent_playbooks as command
from reflexio.models.config_schema import (
    Config,
    PlaybookAggregatorConfig,
    StorageConfigSQLite,
    UserPlaybookExtractorConfig,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.operation_limiter import operation_limit
from reflexio.server.routes import playbooks
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.playbook import aggregation_operations
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def test_cli_waits_for_its_queued_operation_behind_busy_aggregation(
    tmp_path, monkeypatch, runner, app
):
    org = "busy-cli"
    config = DefaultConfigurator(org_id=org, base_dir=str(tmp_path))
    config.set_config(
        Config(
            storage_config=StorageConfigSQLite(db_path=str(tmp_path / "data.db")),
            user_playbook_extractor_config=UserPlaybookExtractorConfig(
                extraction_definition_prompt="Test instructions",
                aggregation_config=PlaybookAggregatorConfig(min_cluster_size=2),
            ),
        )
    )
    context = RequestContext(org_id=org, configurator=config)
    storage = context.storage
    assert isinstance(storage, SQLiteStorage)
    facade = SimpleNamespace(get_storage=lambda: storage, request_context=context)
    monkeypatch.setattr(playbooks.publisher_api, "get_reflexio", lambda **_: facade)
    # An empty corpus is a genuine committed no-op, not a fake successful run.
    llm = MagicMock()
    monkeypatch.setattr(
        aggregation_operations, "create_generation_litellm_client", lambda _: llm
    )
    occupied, release, released, submitted, polled = (
        threading.Event() for _ in range(5)
    )
    failures = []
    notifications = 0

    def notify(_context):
        nonlocal notifications
        notifications += 1
        (submitted if notifications == 1 else polled).set()

    monkeypatch.setattr(
        playbooks, "ensure_local_playbook_aggregation_scheduler", notify
    )
    api = FastAPI()
    api.include_router(playbooks.router)
    api.dependency_overrides[playbooks.default_get_org_id] = lambda: org
    monkeypatch.setattr(playbooks.limiter, "enabled", False)  # Only HTTP rate limits.
    api.state.limiter = playbooks.limiter
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(api, log_level="warning"))
    http_thread = threading.Thread(
        target=lambda: server.run(sockets=[sock]), daemon=True
    )

    def hold_execution_slot():
        try:
            with operation_limit(org, "aggregation"):
                occupied.set()
                assert release.wait(20)
        except Exception as exc:
            failures.append(exc)
        finally:
            released.set()

    def consume_after_poll():
        try:
            assert submitted.wait(20) and polled.wait(20)
            queued = storage.next_playbook_aggregation_operation()
            assert queued is not None and queued.status == "queued"
            assert occupied.is_set() and not released.is_set()
            release.set()
            assert released.wait(5)
            assert aggregation_operations.run_explicit_operation(
                context, owner="test-worker"
            )
        except Exception as exc:
            failures.append(exc)
            release.set()

    holder = threading.Thread(target=hold_execution_slot, daemon=True)
    worker = threading.Thread(target=consume_after_poll, daemon=True)
    sdk = ReflexioClient(
        api_key="test", url_endpoint=f"http://127.0.0.1:{sock.getsockname()[1]}"
    )
    monkeypatch.setattr(command, "get_client", lambda _: sdk)
    monkeypatch.setattr(
        command,
        "time",
        SimpleNamespace(
            monotonic=time.monotonic,
            sleep=lambda seconds: time.sleep(min(seconds, 0.05)),
        ),
    )
    monkeypatch.setattr(command, "_AGGREGATION_WAIT_SECONDS", 15.0, raising=False)
    http_thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.01)
        assert server.started
        holder.start()
        assert occupied.wait(5)
        busy = requests.post(
            sdk.base_url + "/api/run_playbook_aggregation",
            json={"agent_version": "v1"},
            timeout=5,
        )
        assert busy.status_code == 503
        assert busy.json()["detail"] == "aggregation concurrency limit reached"
        worker.start()
        result = runner.invoke(
            app,
            [
                "--json",
                "agent-playbooks",
                "aggregate",
                "--agent-version",
                "v1",
                "--wait",
            ],
        )
        assert result.exit_code == 0, result.output
        worker.join(timeout=10)
        assert not failures, failures
        assert not worker.is_alive()
        output = json.loads(result.stdout)
        receipt = storage.get_playbook_aggregation_operation(
            output["meta"]["operation_id"]
        )
        assert receipt is not None and receipt.status == "succeeded"
        assert receipt.result is not None and receipt.result.playbooks_generated == 0
        assert output["data"]["success"] and output["meta"]["status"] == "succeeded"
        assert (
            storage.conn.execute(
                "SELECT COUNT(*) FROM playbook_aggregation_operations"
            ).fetchone()[0]
            == 1
        )
        llm.generate_content.assert_not_called()
        llm.generate_content_structured.assert_not_called()
    finally:
        release.set()
        submitted.set()
        polled.set()
        if holder.ident is not None:
            holder.join(timeout=5)
        if worker.ident is not None:
            worker.join(timeout=5)
        sdk.session.close()
        server.should_exit = True
        http_thread.join(timeout=5)
        sock.close()
        storage.conn.close()
    assert (
        not http_thread.is_alive() and not holder.is_alive() and not worker.is_alive()
    )
