"""Publish diagnostics use one policy throughout an admitted request."""

import asyncio
from types import SimpleNamespace

import pytest

from reflexio.server import publish_timing as timing


@pytest.fixture(autouse=True)
def isolated_policy(monkeypatch):
    monkeypatch.delenv("REFLEXIO_PUBLISH_TIMING_MIN_MS", raising=False)
    monkeypatch.setenv(timing.ENV_MIN_MS, "0")
    monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "0")
    timing.reset_for_tests()
    yield
    timing.reset_for_tests()


def test_request_keeps_threshold_and_next_request_adopts_edit(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        timing,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], monotonic=lambda: clock[0]),
    )
    monkeypatch.setenv(timing.ENV_MIN_MS, "2000")
    with timing.collect():
        monkeypatch.setenv(timing.ENV_MIN_MS, "0")
        clock[0] = 1.0
        assert timing.emit(org_id="org", request_id="before-edit") is False
    with timing.collect():
        assert timing.emit(org_id="org", request_id="after-edit") is True


def test_request_keeps_throttle_and_next_request_adopts_edit(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(
        timing,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], monotonic=lambda: clock[0]),
    )
    monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "60")
    with timing.collect():
        assert timing.emit(org_id="org", request_id="first") is True
    clock[0] = 1.0
    with timing.collect():
        monkeypatch.setenv(timing.ENV_INTERVAL_SECONDS, "0")
        assert timing.emit(org_id="org", request_id="before-edit") is False
    with timing.collect():
        assert timing.emit(org_id="org", request_id="after-edit") is True


@pytest.mark.parametrize(
    ("env", "threshold", "interval"),
    [
        ({}, None, 60),
        ({"REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "0"}, None, 60),
        ({"REFLEXIO_PUBLISH_TIMING_ENABLED": "true"}, 2000, 60),
        (
            {
                "REFLEXIO_PUBLISH_TIMING_ENABLED": "true",
                "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "-1",
            },
            2000,
            60,
        ),
        (
            {
                "REFLEXIO_PUBLISH_TIMING_ENABLED": "true",
                "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "bad",
            },
            2000,
            60,
        ),
        (
            {
                "REFLEXIO_PUBLISH_TIMING_ENABLED": "true",
                "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "0",
            },
            0,
            60,
        ),
        (
            {timing.ENV_MIN_MS: "off", "REFLEXIO_PUBLISH_TIMING_ENABLED": "true"},
            None,
            60,
        ),
        ({timing.ENV_MIN_MS: "250"}, 250, 60),
        ({timing.ENV_MIN_MS: "0", timing.ENV_INTERVAL_SECONDS: "0"}, 0, 0),
        (
            {timing.ENV_MIN_MS: "250", "REFLEXIO_PUBLISH_TIMING_ENABLED": "false"},
            None,
            60,
        ),
        (
            {
                timing.ENV_MIN_MS: "250",
                "REFLEXIO_PUBLISH_TIMING_ENABLED": "true",
                "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS": "1",
            },
            250,
            60,
        ),
        ({timing.ENV_MIN_MS: " ", "REFLEXIO_PUBLISH_TIMING_ENABLED": "true"}, 2000, 60),
        *[
            ({timing.ENV_MIN_MS: "0", timing.ENV_INTERVAL_SECONDS: value}, 0, 60)
            for value in ("bad", "nan", "inf", "-1")
        ],
    ],
)
def test_policy_mapping_is_authoritative(env, threshold, interval):
    # The autouse fixture deliberately supplies conflicting process values.
    if any(
        name in env
        for name in (
            "REFLEXIO_PUBLISH_TIMING_ENABLED",
            "REFLEXIO_PUBLISH_TIMING_THRESHOLD_MS",
        )
    ):
        with pytest.raises(ValueError, match="Removed environment names"):
            timing.resolve_publish_timing_policy(env)
        return
    assert timing.resolve_publish_timing_policy(env) == timing.PublishTimingPolicy(
        threshold, interval
    )


@pytest.mark.parametrize("value", ["-1", "1.5", "https://private.example/secret"])
def test_invalid_canonical_threshold_names_setting_without_value(value):
    with pytest.raises(ValueError) as error:
        timing.resolve_publish_timing_policy({timing.ENV_MIN_MS: value})
    assert timing.ENV_MIN_MS in str(error.value)
    assert value not in str(error.value)


@pytest.mark.parametrize("canonical", ["off", "0"])
@pytest.mark.parametrize("legacy", ["1", "0", "on", "ture", "private-secret"])
def test_canonical_input_does_not_mask_invalid_legacy_opt_in(canonical, legacy):
    with pytest.raises(ValueError, match="Removed environment names") as error:
        timing.resolve_publish_timing_policy(
            {timing.ENV_MIN_MS: canonical, "REFLEXIO_PUBLISH_TIMING_ENABLED": legacy}
        )
    assert "REFLEXIO_PUBLISH_TIMING_ENABLED" in str(error.value)
    assert "private-secret" not in str(error.value)


def test_http_and_handler_share_admission_snapshot_and_next_request_adopts_edit(
    monkeypatch, caplog
):
    monkeypatch.delenv("REFLEXIO_PUBLISH_TIMING_ENABLED", raising=False)
    monkeypatch.setenv(timing.ENV_MIN_MS, "0")
    sent = []
    captured = []

    async def app(_scope, _receive, send):
        outer = timing._http_scope.get()
        monkeypatch.setenv(timing.ENV_MIN_MS, "off")
        with timing.collect():
            inner = timing._scope.get()
            if outer is not None and outer.policy.threshold_ms is not None:
                assert inner is not None
                assert inner.policy is outer.policy
                captured.append(inner.policy)
                timing.handler_started(
                    org_id="org", request_id="req", wait_for_response=False
                )
                assert timing.emit(org_id="org", request_id="req")
            else:
                assert inner is None
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    middleware = timing.PublishHttpTimingMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/publish_interaction",
        "root_path": "",
    }
    asyncio.run(middleware(scope, receive, send))
    asyncio.run(middleware(scope, receive, send))
    assert len(captured) == 1
    assert [
        message["body"] for message in sent if message["type"] == "http.response.body"
    ] == [b"ok", b"ok"]
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == timing._timing_logger.name
    ]
    assert sum(message.startswith("event=publish_timing ") for message in messages) == 1
    assert (
        sum(message.startswith("event=publish_http_timing ") for message in messages)
        == 1
    )


def test_disabled_http_snapshot_survives_enable_edit_until_next_request(monkeypatch):
    monkeypatch.delenv("REFLEXIO_PUBLISH_TIMING_ENABLED", raising=False)
    monkeypatch.setenv(timing.ENV_MIN_MS, "off")
    admitted = []

    async def app(_scope, _receive, send):
        monkeypatch.setenv(timing.ENV_MIN_MS, "0")
        outer = timing._http_scope.get()
        assert outer is not None
        timing.http_org_resolved("org")
        timing.handler_started(org_id="org", request_id="req", wait_for_response=True)
        with timing.http_phase("dependency"):
            pass
        if outer.policy.threshold_ms is None:
            assert outer.org_id == "unknown"
            assert outer.request_id == "unknown"
            assert outer.handler_started is None
            assert outer.wait_for_response is False
            assert outer.phases == {}
        with timing.collect():
            admitted.append(timing.emit(org_id="org", request_id="req"))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        pass

    middleware = timing.PublishHttpTimingMiddleware(app)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/publish_interaction",
        "root_path": "",
    }
    asyncio.run(middleware(scope, receive, send))
    assert timing._http_scope.get() is None
    asyncio.run(middleware(scope, receive, send))
    assert admitted == [False, True]
