"""No network, credentials, model, storage or service processes in these tests."""

import gzip
import traceback
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from reflexio.server import search_runtime
from reflexio.server.env_utils import env_str
from reflexio.server.services.retrieval import judge_client as jc
from reflexio.server.services.storage.error import ReadinessUnavailableError

pytestmark = pytest.mark.unit
MODEL = "jev-1.14.0"
URL = "https://api.typesafe.ai/v1/systemone"
PAYLOAD = {"model": MODEL, "state": {"request": "Synthetic task"}, "questions": {}}
REPLY = {"model": MODEL, "answers": {}, "usage": {"input_tokens": 1}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    jc.close_judge_client()
    # An entirely fake mapping prevents every test from reading a real key.
    env = {"REFLEXIO_SEARCH_JUDGE": "jev", "TYPESAFE_API_KEY": "unit-key"}
    monkeypatch.setattr(
        jc, "env_str", lambda name, default="": env_str(name, default, env=env)
    )
    monkeypatch.setattr(jc, "monotonic", lambda: 0.0)
    yield env
    jc.close_judge_client()


@pytest.fixture
def wire(monkeypatch):
    state: dict[str, Any] = {
        "active": False,
        "requests": [],
        "constructors": [],
        "clients": [],
        "leases": [],
        "handler": None,
    }
    original_client = httpx.Client

    @contextmanager
    def lease(provider, *, max_wait):
        state["leases"].append((provider, max_wait))
        state["active"] = True
        try:
            if state.get("on_admit"):
                state["on_admit"]()
            yield
        finally:
            state["active"] = False
            state["released_after_requests"] = len(state["requests"])

    def handler(request):
        assert state["active"]
        state["requests"].append(request)
        if state["handler"]:
            return state["handler"](request)
        return httpx.Response(200, json=REPLY)

    def factory(**kwargs):
        state["constructors"].append(dict(kwargs))
        client = original_client(transport=httpx.MockTransport(handler), **kwargs)
        state["clients"].append(client)
        return client

    monkeypatch.setattr(jc, "named_provider_slot", lease)
    monkeypatch.setattr(
        jc,
        "httpx",
        SimpleNamespace(Client=factory, Limits=httpx.Limits, Request=httpx.Request),
    )
    return state


@pytest.mark.parametrize("mode", [None, "", "   ", "off", " off ", "jev", " jev "])
def test_mode_is_explicit_and_never_reads_key_or_creates_client(monkeypatch, mode):
    accessed = []

    class FakeEnv(dict):
        def get(self, key, default=None):
            accessed.append(key)
            assert key == "REFLEXIO_SEARCH_JUDGE"
            return super().get(key, default)

    env = FakeEnv() if mode is None else FakeEnv(REFLEXIO_SEARCH_JUDGE=mode)
    monkeypatch.setattr(
        jc, "env_str", lambda name, default="": env_str(name, default, env=env)
    )
    monkeypatch.setattr(
        jc, "_client", lambda: pytest.fail("mode must not construct HTTP client")
    )
    assert jc.judge_mode() == ("jev" if mode and mode.strip() == "jev" else "off")
    assert accessed == ["REFLEXIO_SEARCH_JUDGE"]


@pytest.mark.parametrize("mode", ["other", "JEV", "secret-mode-text"])
def test_unknown_mode_is_safe_typed_refusal(isolated, mode):
    isolated["REFLEXIO_SEARCH_JUDGE"] = mode
    with pytest.raises(ReadinessUnavailableError) as caught:
        jc.judge_mode()
    assert mode not in str(caught.value)


def test_disabled_call_cannot_read_key_or_bootstrap_client(monkeypatch, wire):
    class DisabledEnv(dict):
        def get(self, key, default=None):
            assert key == "REFLEXIO_SEARCH_JUDGE"
            return "off"

    monkeypatch.setattr(
        jc,
        "env_str",
        lambda name, default="": env_str(name, default, env=DisabledEnv()),
    )
    with pytest.raises(ReadinessUnavailableError):
        jc.call_jev(PAYLOAD)
    assert not wire["constructors"] and not wire["leases"] and not wire["requests"]


@pytest.mark.parametrize("key", [None, "", "   "])
def test_missing_key_is_explicit_without_http(isolated, wire, key):
    if key is None:
        isolated.pop("TYPESAFE_API_KEY")
    else:
        isolated["TYPESAFE_API_KEY"] = key
    with pytest.raises(ReadinessUnavailableError, match="TYPESAFE_API_KEY"):
        jc.call_jev(PAYLOAD)
    assert not wire["constructors"] and not wire["requests"]


def test_exact_frozen_url_gzip_headers_and_direct_client_reuse(wire):
    payload = deepcopy(PAYLOAD)
    payload["state"]["request"] = "Synthetic unicode é task"
    untouched = deepcopy(payload)
    assert jc.call_jev(payload) == REPLY
    assert jc.call_jev(payload) == REPLY
    assert payload == untouched
    assert len(wire["constructors"]) == 1 and len(wire["requests"]) == 2
    policy = wire["constructors"][0]
    assert policy["trust_env"] is False and policy["follow_redirects"] is False
    assert policy["event_hooks"] == {"request": [search_runtime.http_request_deadline]}
    for request in wire["requests"]:
        assert str(request.url) == URL and request.method == "POST"
        assert request.headers["authorization"] == "Bearer unit-key"
        assert request.headers["content-type"] == "application/json"
        assert request.headers["content-encoding"] == "gzip"
        expected = httpx.Request("POST", URL, json=payload).content
        assert request.content == gzip.compress(expected, compresslevel=1, mtime=0)
        assert gzip.decompress(request.content) == expected
        assert all(
            0 < seconds <= 0.7 for seconds in request.extensions["timeout"].values()
        )
    assert wire["leases"] == [("typesafe", 0.7), ("typesafe", 0.7)]
    assert wire["released_after_requests"] == 2 and not wire["active"]


def test_close_is_idempotent_and_future_call_gets_new_client(wire):
    jc.call_jev(PAYLOAD)
    first = wire["clients"][0]
    jc.close_judge_client()
    jc.close_judge_client()
    assert first.is_closed
    jc.call_jev(PAYLOAD)
    assert len(wire["clients"]) == 2 and not wire["clients"][1].is_closed


def test_pid_change_discards_stale_pool(wire, monkeypatch):
    jc.call_jev(PAYLOAD)
    first = wire["clients"][0]
    monkeypatch.setattr(jc, "os", SimpleNamespace(getpid=lambda: 1234567))
    jc.call_jev(PAYLOAD)
    assert first.is_closed and len(wire["clients"]) == 2


def test_after_fork_never_inherits_a_locked_constructor(wire):
    jc.call_jev(PAYLOAD)
    inherited = wire["clients"][0]
    parent_lock = jc._client_lock
    parent_lock.acquire()
    try:
        jc._after_fork()
        assert jc._client_instance is None and jc._client_pid is None
        assert jc._client_lock is not parent_lock
        jc.call_jev(PAYLOAD)
        assert len(wire["clients"]) == 2
    finally:
        parent_lock.release()
        inherited.close()


@pytest.mark.parametrize("status", [301, 302, 307, 400, 401, 429, 500, 503])
def test_non2xx_refuses_without_redirect_retry_or_fallback(wire, status):
    wire["handler"] = lambda _: httpx.Response(
        status,
        headers={"location": "https://unsafe.invalid/"},
        text="private-provider-body",
    )
    with pytest.raises(ReadinessUnavailableError) as caught:
        jc.call_jev(PAYLOAD)
    assert len(wire["requests"]) == 1 and len(wire["leases"]) == 1
    text = "".join(traceback.format_exception(caught.value))
    assert "private-provider-body" not in text and "unit-key" not in text


@pytest.mark.parametrize(
    "body",
    [
        b"not-json",
        b"[]",
        b"null",
        b'{"model":"other","answers":{}}',
        b'{"model":"jev-1.14.0","answers":[]}',
        b'{"model":"jev-1.14.0","answers":{},"answers":{}}',
        b'{"model":"jev-1.14.0","answers":{"a":{"x":1,"x":2}}}',
        b'{"model":"jev-1.14.0","answers":{},"n":NaN}',
        b'{"model":"jev-1.14.0","answers":{},"n":Infinity}',
        b'{"model":"jev-1.14.0","answers":{},"n":1e309}',
    ],
)
def test_strict_json_and_envelope_errors_are_safe_readiness_failures(wire, body):
    wire["handler"] = lambda _: httpx.Response(200, content=body)
    with pytest.raises(ReadinessUnavailableError) as caught:
        jc.call_jev(PAYLOAD)
    assert len(wire["requests"]) == 1
    assert "unit-key" not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "payload", [{"model": "another"}, {"model": MODEL, "n": float("nan")}]
)
def test_invalid_request_never_dispatches(wire, payload):
    with pytest.raises(ReadinessUnavailableError):
        jc.call_jev(payload)
    assert not wire["requests"]


def test_provider_exception_body_is_not_chained_or_logged(wire, caplog):
    def fail(_):
        raise RuntimeError("private-provider-body unit-key Synthetic task")

    wire["handler"] = fail
    with pytest.raises(ReadinessUnavailableError) as caught:
        jc.call_jev(PAYLOAD)
    text = "".join(traceback.format_exception(caught.value)) + caplog.text
    assert "private-provider-body" not in text and "unit-key" not in text
    assert "Synthetic task" not in text and len(wire["requests"]) == 1


def test_deadline_before_admission_never_reads_key_or_constructs_client(
    wire, monkeypatch
):
    def expired():
        raise search_runtime.SearchDeadlineError()

    monkeypatch.setattr(search_runtime, "checkpoint", expired)
    with pytest.raises(search_runtime.SearchDeadlineError):
        jc.call_jev(PAYLOAD)
    assert not wire["requests"] and not wire["constructors"]


def test_component_budget_includes_provider_queue(wire, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(jc, "monotonic", lambda: clock[0])
    wire["on_admit"] = lambda: clock.__setitem__(0, 0.71)
    with pytest.raises(search_runtime.SearchDeadlineError):
        jc.call_jev(PAYLOAD)
    assert not wire["requests"] and not wire["constructors"]


def test_caller_budget_is_not_reset_by_queue_and_encoding(wire, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(jc, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        search_runtime, "remaining", lambda default=0.7: min(default, 0.35 - clock[0])
    )
    wire["on_admit"] = lambda: clock.__setitem__(0, 0.1)
    compress = gzip.compress

    def encode(*args, **kwargs):
        clock[0] += 0.1
        return compress(*args, **kwargs)

    monkeypatch.setattr(jc.gzip, "compress", encode)
    assert jc.call_jev(PAYLOAD) == REPLY
    timeout = wire["requests"][0].extensions["timeout"]
    assert all(0 < value <= 0.15 for value in timeout.values())


def test_deadline_after_http_keeps_lease_until_physical_call_returns(wire, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(jc, "monotonic", lambda: clock[0])

    def late(_):
        clock[0] = 0.71
        assert wire["active"]
        return httpx.Response(200, json=REPLY)

    wire["handler"] = late
    with pytest.raises(search_runtime.SearchDeadlineError):
        jc.call_jev(PAYLOAD)
    assert len(wire["requests"]) == 1 and wire["released_after_requests"] == 1
    assert not wire["active"]


def test_deadline_during_json_parse_cannot_accept_late_reply(wire, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(jc, "monotonic", lambda: clock[0])
    decode = jc._decode

    def delayed(response):
        assert wire["active"]
        clock[0] = 0.71
        return decode(response)

    monkeypatch.setattr(jc, "_decode", delayed)
    with pytest.raises(search_runtime.SearchDeadlineError):
        jc.call_jev(PAYLOAD)
    assert len(wire["requests"]) == 1


def test_literal_deployment_contract_matches_runtime_mode_and_key_names():
    assert jc.SEARCH_JUDGE_CONTRACT == {
        "mode_env": jc._MODE_ENV,
        "default": "off",
        "enabled": "jev",
        "required_env_secrets": (jc._KEY_ENV,),
    }
