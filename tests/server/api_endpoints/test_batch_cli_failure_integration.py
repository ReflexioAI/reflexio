"""Real HTTP/SDK/CLI must expose a partial batch and retain pending output."""

import json
import socket
import threading
import time
from unittest.mock import MagicMock

import pytest
import uvicorn
from typer.testing import CliRunner

from reflexio import ReflexioClient
from reflexio.cli.app import create_app as create_cli
from reflexio.lib._operations import OperationsMixin
from reflexio.models.api_schema.service_schemas import Status, UserPlaybook
from reflexio.server.api import create_app
from reflexio.server.services.playbook.service import PlaybookGenerationService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("json_mode", [False, True])
def test_failed_batch_over_real_http_stops_cli_promotion(
    tmp_path, monkeypatch, json_mode
):
    storage = SQLiteStorage(org_id="batch-http", db_path=str(tmp_path / "batch.db"))
    context = MagicMock()
    context.org_id = "batch-http"
    context.storage = storage
    service = PlaybookGenerationService(llm_client=MagicMock(), request_context=context)
    release_generation = threading.Event()

    def select_users(_request):
        assert release_generation.wait(timeout=10)
        return ["failed", "valid"]

    monkeypatch.setattr(service, "_get_rerun_user_ids", select_users)
    storage.upsert_operation_state(
        "playbook_generation::batch-http::progress",
        {
            "service_name": "playbook_generation",
            "status": "completed",
            "started_at": int(time.time()),
            "processed_users": 2,
            "request_params": {"operation_id": "previous-attempt"},
            "stats": {"total_generated": 99},
        },
    )
    monkeypatch.setattr(service, "_pre_process_rerun", lambda _request: None)

    def generate(request):
        if request.user_id == "failed":
            raise RuntimeError("invalid model evidence")
        storage.save_user_playbooks(
            [
                UserPlaybook(
                    user_id="valid",
                    agent_version="v1",
                    request_id="valid-run",
                    content="Use the supplied source evidence",
                    status=Status.PENDING,
                )
            ],
            skip_embedding=True,
        )

    monkeypatch.setattr(service, "run", generate)
    promotions = []
    facade = MagicMock(request_context=context, _is_storage_configured=lambda: True)
    facade._get_storage.return_value = storage
    facade.get_storage.return_value = storage
    facade.rerun_playbook_generation = service.run_rerun
    facade.get_operation_status = lambda request: OperationsMixin.get_operation_status(
        facade, request
    )
    facade.upgrade_user_playbooks = lambda _request: promotions.append(True)
    monkeypatch.setattr(
        "reflexio.server.cache.reflexio_cache.get_reflexio", lambda **_kwargs: facade
    )
    make_request = ReflexioClient._make_request
    observed_pending = []

    def observe_status(client, method, path, **kwargs):
        response = make_request(client, method, path, **kwargs)
        if path == "/api/get_operation_status":
            state = response.get("operation_status")
            if state and state["status"] == "in_progress":
                assert state["request_params"]["operation_id"] != "previous-attempt"
                if not observed_pending:
                    # A second HTTP admission must reject immediately while selection is paused.
                    busy = make_request(
                        client,
                        "POST",
                        "/api/rerun_playbook_generation",
                        json={
                            "agent_version": "v1",
                            "_operation_id": "previous-attempt",
                        },
                    )
                    assert busy["success"] is False
                    current = service._create_state_manager().get_progress()
                    assert current is not None
                    assert (
                        current["request_params"]["operation_id"]
                        == state["request_params"]["operation_id"]
                    )
                observed_pending.append(state)
                release_generation.set()
        return response

    monkeypatch.setattr(ReflexioClient, "_make_request", observe_status)
    poll = ReflexioClient._poll_operation_status

    def bounded_poll(client, *args, **kwargs):
        # Keep actual HTTP polling; bound the deterministic fixture's wait.
        return poll(client, *args, **kwargs, poll_interval=0.02, max_wait=10)

    monkeypatch.setattr(ReflexioClient, "_poll_operation_status", bounded_poll)
    app = create_app(get_org_id=lambda: "batch-http")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        command = (["--json"] if json_mode else []) + [
            "--server-url",
            f"http://127.0.0.1:{listener.getsockname()[1]}",
            "user-playbooks",
            "regenerate",
            "--agent-version",
            "v1",
            "--wait",
        ]
        result = CliRunner().invoke(create_cli(), command)
        assert result.exit_code == 1, result.output
        assert "Generation failed for 1 user(s)" in result.output
        assert "regeneration complete" not in result.output
        if json_mode:
            assert json.loads(result.output)["ok"] is False
        assert observed_pending, "Client must wait for its admitted attempt"
        assert not promotions
        state = service._create_state_manager().get_progress()
        assert state is not None
        assert state["status"] == "failed"
        assert state["processed_users"] == state["failed_users"] == 1
        assert state["stats"]["total_generated"] == 1
        pending = storage.get_user_playbooks(
            agent_version="v1", status_filter=[Status.PENDING], limit=10
        )
        assert len(pending) == 1
        assert pending[0].user_id == "valid"
    finally:
        release_generation.set()
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive()
