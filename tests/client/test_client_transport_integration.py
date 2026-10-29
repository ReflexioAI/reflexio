"""Client transport behaviour against a scripted loopback HTTP server.

Both transports (``requests`` for blocking calls, ``aiohttp`` for native-async
and fire-and-forget calls) must turn a misconfigured endpoint into a clear
``ReflexioAPIError`` instead of following a redirect or crashing on a non-JSON
body. These tests send real HTTP to a stub server whose replies are scripted
per request, so the guards are exercised exactly as a wrong ``REFLEXIO_URL``
would trigger them.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest
import requests
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from reflexio.client import ReflexioClient
from reflexio.client.client import ReflexioAPIError
from reflexio.models.api_schema.service_schemas import OperationStatus

pytestmark = pytest.mark.integration


@dataclass
class Reply:
    status: int = 200
    body: bytes = b"{}"
    content_type: str | None = "application/json"
    headers: dict[str, str] = field(default_factory=dict)


def json_reply(payload: dict, status: int = 200) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


@dataclass
class StubServer:
    url: str
    replies: deque[Reply] = field(default_factory=deque)
    default: Reply = field(default_factory=Reply)
    seen: list[tuple[str, str, dict]] = field(default_factory=list)

    def client(self) -> ReflexioClient:
        return ReflexioClient(api_key="stub-key", url_endpoint=self.url, timeout=10)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture()
def stub() -> Iterator[StubServer]:
    holder: dict[str, StubServer] = {}

    async def handle(request: Request) -> Response:
        server = holder["server"]
        server.seen.append(
            (request.method, request.url.path, dict(request.headers.items()))
        )
        reply = server.replies.popleft() if server.replies else server.default
        headers = dict(reply.headers)
        if reply.content_type is not None:
            headers["content-type"] = reply.content_type
        return Response(content=reply.body, status_code=reply.status, headers=headers)

    app = Starlette(
        routes=[
            Route(
                "/{path:path}",
                handle,
                methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            )
        ]
    )
    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=_free_port(), lifespan="off", log_level="error"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        assert time.monotonic() < deadline, "stub server did not start"
        time.sleep(0.02)
    port = server.servers[0].sockets[0].getsockname()[1]
    holder["server"] = StubServer(url=f"http://127.0.0.1:{port}")
    try:
        yield holder["server"]
    finally:
        server.should_exit = True
        thread.join(timeout=10)


_SEARCH_OK = {"success": True, "profiles": [], "user_playbooks": []}


# ---------------------------------------------------------------- blocking path


def test_blocking_call_sends_bearer_auth_and_decodes_json(stub):
    stub.replies.append(json_reply({"success": True, "org_id": "org-1"}))
    who = stub.client().whoami()
    assert who.org_id == "org-1"
    method, path, headers = stub.seen[-1]
    assert (method, path) == ("GET", "/api/whoami")
    assert headers["authorization"] == "Bearer stub-key"


def test_blocking_call_treats_empty_success_body_as_empty_object(stub):
    stub.replies.append(Reply(status=204, body=b"", content_type=None))
    # set_config returns the decoded body as-is, so the empty 204 is visible.
    assert stub.client().set_config({"storage_config": None}) == {}
    method, path, _ = stub.seen[-1]
    assert (method, path) == ("POST", "/api/set_config")


def test_blocking_call_rejects_non_json_content_type(stub):
    stub.replies.append(Reply(body=b"<html>marketing</html>", content_type="text/html"))
    with pytest.raises(ReflexioAPIError) as error:
        stub.client().whoami()
    message = str(error.value)
    assert "Content-Type=text/html" in message
    assert "<html>marketing" in message


def test_blocking_call_rejects_malformed_json_body(stub):
    stub.replies.append(Reply(body=b"{not json"))
    with pytest.raises(ReflexioAPIError, match="not valid JSON"):
        stub.client().whoami()


def test_blocking_call_never_follows_redirects(stub):
    stub.replies.append(Reply(status=302, body=b"", headers={"location": "/elsewhere"}))
    with pytest.raises(requests.TooManyRedirects):
        stub.client().whoami()
    assert [path for _, path, _ in stub.seen] == ["/api/whoami"]


def test_blocking_call_raises_http_error_status(stub):
    stub.replies.append(json_reply({"detail": "nope"}, status=500))
    with pytest.raises(requests.HTTPError):
        stub.client().whoami()


# ------------------------------------------------------------------ async path


def test_async_call_decodes_json_with_auth(stub):
    stub.replies.append(json_reply(_SEARCH_OK))
    result = asyncio.run(stub.client().search_async(query="deploys"))
    assert result.success
    method, path, headers = stub.seen[-1]
    assert (method, path) == ("POST", "/api/search")
    assert headers["authorization"] == "Bearer stub-key"


def test_async_call_rejects_redirect_without_following(stub):
    stub.replies.append(Reply(status=307, body=b"", headers={"location": "/elsewhere"}))
    with pytest.raises(ReflexioAPIError, match="Unexpected redirect"):
        asyncio.run(stub.client().search_async(query="q"))
    assert [path for _, path, _ in stub.seen] == ["/api/search"]


def test_async_call_rejects_non_json_content_type(stub):
    stub.replies.append(Reply(body=b"<html>hi</html>", content_type="text/html"))
    with pytest.raises(ReflexioAPIError, match="Content-Type=text/html"):
        asyncio.run(stub.client().search_async(query="q"))


def test_async_call_rejects_malformed_json_body(stub):
    stub.replies.append(Reply(body=b"[unterminated"))
    with pytest.raises(ReflexioAPIError, match="not valid JSON"):
        asyncio.run(stub.client().search_async(query="q"))


def test_async_call_treats_empty_body_as_empty_object(stub):
    from pydantic import ValidationError

    stub.replies.append(Reply(status=200, body=b"", content_type=None))
    # The empty body decodes to {} (no JSON error); {} then fails validation
    # for the response's required ``success`` field.
    with pytest.raises(ValidationError, match="success"):
        asyncio.run(stub.client().search_async(query="q"))


def test_fire_and_forget_inside_running_loop_uses_that_loop(stub):
    stub.default = json_reply({"success": True})
    client = stub.client()

    async def submit_and_drain() -> None:
        assert client.delete_request("req-9") is None
        # The task was scheduled on this loop; give it time to run.
        deadline = time.monotonic() + 5
        while not stub.seen and time.monotonic() < deadline:
            await asyncio.sleep(0.02)

    asyncio.run(submit_and_drain())
    method, path, _ = stub.seen[-1]
    assert (method, path) == ("DELETE", "/api/delete_request")


# -------------------------------------------------------------- status polling


def _operation(operation_id: str, status: str, started_at: int, **extra) -> dict:
    return {
        "success": True,
        "operation_status": {
            "service_name": "profile_generation",
            "status": status,
            "started_at": started_at,
            "request_params": {"operation_id": operation_id},
            "stats": extra.pop("stats", {}),
            **extra,
        },
    }


def test_polling_survives_a_transient_error_before_completion(stub):
    now = int(time.time())
    stub.replies.extend(
        [
            json_reply({"detail": "warming up"}, status=503),
            json_reply(_operation("op-1", "completed", now)),
        ]
    )
    status = stub.client()._poll_operation_status(
        "profile_generation", poll_interval=0.01, max_wait=5, operation_id="op-1"
    )
    assert status.operation_status is not None
    assert status.operation_status.status == OperationStatus.COMPLETED
    assert len(stub.seen) == 2


def test_polling_gives_up_when_errors_persist(stub):
    stub.default = json_reply({"detail": "down"}, status=503)
    with pytest.raises(TimeoutError, match="did not complete"):
        stub.client()._poll_operation_status(
            "profile_generation", poll_interval=0.01, max_wait=0.05
        )


def test_polling_ignores_an_operation_started_before_submission(stub):
    now = int(time.time())
    stub.default = json_reply(_operation("older", "completed", now - 120))
    with pytest.raises(TimeoutError):
        stub.client()._poll_operation_status(
            "profile_generation",
            poll_interval=0.01,
            max_wait=0.05,
            min_started_at=now,
        )


@pytest.mark.parametrize(
    ("kind", "call"),
    [
        (
            "profile",
            lambda c: c.rerun_profile_generation(user_id="u", wait_for_response=True),
        ),
        (
            "playbook",
            lambda c: c.rerun_playbook_generation(
                agent_version="v1", wait_for_response=True
            ),
        ),
    ],
)
def test_rejected_rerun_is_returned_without_polling(stub, kind, call):
    stub.replies.append(
        json_reply({"success": False, "msg": f"{kind} attempt already running"})
    )
    response = call(stub.client())
    assert response.success is False
    assert response.msg == f"{kind} attempt already running"
    # Only the submission was sent; nothing polled for a rejected attempt.
    assert len(stub.seen) == 1
