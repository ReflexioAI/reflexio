"""add() forwarding to mem0 and best-effort publish to Reflexio."""

import types

import pytest


def _client(wrapped_cls, reflexio_mock):
    return wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)


def test_add_forwards_and_returns_mem0_result(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.add("I love jazz", user_id="u1", metadata={"k": "v"})
    assert result is client.add_result
    call = client.calls[0]
    assert call[0] == "add"
    assert call[1] == "I love jazz"
    assert call[3] == {"user_id": "u1", "metadata": {"k": "v"}}


def test_publish_maps_ids_and_flags(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("hello", user_id="u1", agent_id="support-bot", run_id="run-42")
    kwargs = reflexio_mock.publish_interaction.call_args.kwargs
    assert kwargs["user_id"] == "u1"
    assert kwargs["session_id"].startswith("mem0-run-v1-")
    assert kwargs["agent_version"] == "support-bot"
    assert kwargs["source"] == "mem0"
    assert kwargs["wait_for_response"] is False
    assert kwargs["interactions"] == [{"role": "User", "content": "hello"}]


def test_publish_maps_ids_from_options_filters(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "hello",
        types.SimpleNamespace(
            filters={"user_id": "u-opt", "agent_id": "bot-opt", "run_id": "run-opt"}
        ),
    )
    kwargs = reflexio_mock.publish_interaction.call_args.kwargs
    assert kwargs["user_id"] == "u-opt"
    assert kwargs["agent_version"] == "bot-opt"
    assert kwargs["session_id"].startswith("mem0-run-v1-")


def test_add_kwargs_filters_override_options_filters(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "hello",
        types.SimpleNamespace(
            filters={
                "AND": [
                    {"user_id": "u1", "agent_id": "a1", "run_id": "r1"},
                    {"user_id": "u2", "agent_id": "a2", "run_id": "r2"},
                ]
            }
        ),
        filters={"user_id": "u-kw", "agent_id": "bot-kw", "run_id": "run-kw"},
    )
    kwargs = reflexio_mock.publish_interaction.call_args.kwargs
    assert kwargs["user_id"] == "u-kw"
    assert kwargs["agent_version"] == "bot-kw"
    assert kwargs["session_id"].startswith("mem0-run-v1-")


def test_add_top_level_identity_kwargs_override_filters(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "hello",
        filters={
            "AND": [
                {"user_id": "u1", "agent_id": "a1", "run_id": "r1"},
                {"user_id": "u2", "agent_id": "a2", "run_id": "r2"},
            ]
        },
        user_id="u-direct",
        agent_id="bot-direct",
        run_id="run-direct",
    )
    kwargs = reflexio_mock.publish_interaction.call_args.kwargs
    assert kwargs["user_id"] == "u-direct"
    assert kwargs["agent_version"] == "bot-direct"
    assert kwargs["session_id"].startswith("mem0-run-v1-")


@pytest.mark.parametrize("identity", ["user_id", "agent_id", "app_id", "run_id"])
def test_conflicting_filter_identities_skip_publish(
    identity, wrapped_cls, reflexio_mock
):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "hello",
        filters={
            "user_id": "u1",
            "AND": [{identity: "first"}, {identity: "second"}],
        },
    )
    reflexio_mock.publish_interaction.assert_not_called()


def test_fallback_session_id_is_stable_and_scoped_to_user_and_agent(
    wrapped_cls, reflexio_mock
):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("a", user_id="u1", agent_id="a1")
    client.add("b", user_id="u1", agent_id="a1")
    client.add("c", user_id="u2", agent_id="a1")
    client.add("d", user_id="u1", agent_id="a2")
    sessions = [
        c.kwargs["session_id"] for c in reflexio_mock.publish_interaction.call_args_list
    ]
    assert sessions[0] == sessions[1]
    assert sessions[0].startswith("mem0-run-v1-")
    assert len(set(sessions)) == 3

    other = _client(wrapped_cls, reflexio_mock)
    other.add("e", user_id="u1", agent_id="a1")
    assert (
        reflexio_mock.publish_interaction.call_args.kwargs["session_id"] != sessions[0]
    )


def test_app_id_isolates_user_agent_and_explicit_run(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "first", user_id="same-user", agent_id="same-agent", app_id="app-a", run_id="r"
    )
    first = reflexio_mock.publish_interaction.call_args.kwargs
    client.add(
        "second", user_id="same-user", agent_id="same-agent", app_id="app-b", run_id="r"
    )
    second = reflexio_mock.publish_interaction.call_args.kwargs
    assert first["user_id"].startswith("mem0-user-v1-")
    assert first["agent_version"].startswith("mem0-agent-v1-")
    assert first["user_id"] != second["user_id"]
    assert first["agent_version"] != second["agent_version"]
    assert first["session_id"] != second["session_id"]


def test_scope_encoding_is_delimiter_safe(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("a", user_id="c雪", agent_id="d", app_id="a\0b", run_id="e")
    first = reflexio_mock.publish_interaction.call_args.kwargs
    client.add("b", user_id="b\0c雪", agent_id="d", app_id="a", run_id="e")
    second = reflexio_mock.publish_interaction.call_args.kwargs
    assert first["user_id"] != second["user_id"]
    assert first["session_id"] != second["session_id"]


def test_reused_run_id_is_isolated_across_users(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("a", user_id="u1", agent_id="agent", run_id="same-run")
    first = reflexio_mock.publish_interaction.call_args.kwargs["session_id"]
    client.add("b", user_id="u2", agent_id="agent", run_id="same-run")
    second = reflexio_mock.publish_interaction.call_args.kwargs["session_id"]
    assert first != second


def test_agent_version_defaults_when_no_agent_id(
    wrapped_cls, reflexio_mock, monkeypatch
):
    monkeypatch.delenv("REFLEXIO_AGENT_VERSION", raising=False)
    client = _client(wrapped_cls, reflexio_mock)
    client.add("hello", user_id="u1")
    assert (
        reflexio_mock.publish_interaction.call_args.kwargs["agent_version"]
        == "agent-v0"
    )


@pytest.mark.parametrize(
    ("messages", "expected"),
    [
        (
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
            [{"role": "User", "content": "hi"}, {"role": "Assistant", "content": "yo"}],
        ),
        ({"role": "tool", "content": "ran"}, [{"role": "Tool", "content": "ran"}]),
        (["plain string"], [{"role": "User", "content": "plain string"}]),
        (
            [
                {"role": "system", "content": "you are a bot"},
                {"role": "user", "content": ""},
                {"role": "user", "content": "kept"},
                {"role": "user", "content": {"type": "image"}},
            ],
            [{"role": "User", "content": "kept"}],
        ),
    ],
)
def test_message_normalization(wrapped_cls, reflexio_mock, messages, expected):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(messages, user_id="u1")
    assert (
        reflexio_mock.publish_interaction.call_args.kwargs["interactions"] == expected
    )


def test_timestamp_from_options_becomes_created_at(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("hi", types.SimpleNamespace(timestamp=1720000000), user_id="u1")
    interactions = reflexio_mock.publish_interaction.call_args.kwargs["interactions"]
    assert interactions == [{"role": "User", "content": "hi", "created_at": 1720000000}]


def test_no_user_id_skips_publish(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.add("hi", agent_id="bot", run_id="r1")
    assert result is client.add_result
    reflexio_mock.publish_interaction.assert_not_called()


def test_all_messages_filtered_skips_publish(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add([{"role": "system", "content": "x"}], user_id="u1")
    reflexio_mock.publish_interaction.assert_not_called()


def test_reflexio_publish_error_is_swallowed(wrapped_cls, reflexio_mock, caplog):
    reflexio_mock.publish_interaction.side_effect = RuntimeError("reflexio down")
    client = _client(wrapped_cls, reflexio_mock)
    with caplog.at_level("WARNING"):
        result = client.add("hi", user_id="u1")
    assert result is client.add_result
    assert "Reflexio publish failed" in caplog.text


def test_mem0_error_propagates_and_no_publish(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.raise_on_add = ValueError("bad messages")
    with pytest.raises(ValueError, match="bad messages"):
        client.add("hi", user_id="u1")
    reflexio_mock.publish_interaction.assert_not_called()


def test_default_reflexio_client_failure_degrades(wrapped_cls, monkeypatch):
    import reflexio.mem0._wrapper as wrapper_module

    monkeypatch.setenv("REFLEXIO_URL", "http://localhost:1")
    monkeypatch.setattr(
        wrapper_module,
        "ReflexioClient",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no env")),
    )
    client = wrapped_cls(api_key="mk")
    assert client.reflexio.configured is False
    assert client.add("hi", user_id="u1") is client.add_result


def test_unconfigured_env_yields_none_client(wrapped_cls, monkeypatch):
    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    client = wrapped_cls(api_key="mk")
    assert client.reflexio.configured is False
    # Pure pass-through: mem0 result comes back, nothing else happens.
    assert client.add("hi", user_id="u1") is client.add_result
    assert "reflexio" not in client.search("q", filters={"user_id": "u1"})


@pytest.mark.parametrize("environment", ["key", "url"])
def test_environment_configuration_creates_five_second_client(
    environment, wrapped_cls, monkeypatch
):
    import reflexio.mem0._wrapper as wrapper_module

    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    monkeypatch.setenv(
        "REFLEXIO_API_KEY" if environment == "key" else "REFLEXIO_URL",
        "test-key" if environment == "key" else "http://localhost:8081",
    )
    created = types.SimpleNamespace(timeout=5.0)
    constructor_calls = []

    def construct(**kwargs):
        constructor_calls.append(kwargs)
        return created

    monkeypatch.setattr(wrapper_module, "ReflexioClient", construct)
    client = wrapped_cls(api_key="mk")
    assert client.reflexio.configured is True
    assert constructor_calls == [{"timeout": 5.0}]


def test_wrapper_timeout_overrides_created_client_timeout(wrapped_cls, monkeypatch):
    import reflexio.mem0._wrapper as wrapper_module

    monkeypatch.setenv("REFLEXIO_URL", "http://localhost:8081")
    created = types.SimpleNamespace(timeout=1.25)
    constructor_calls = []

    def construct(**kwargs):
        constructor_calls.append(kwargs)
        return created

    monkeypatch.setattr(wrapper_module, "ReflexioClient", construct)
    client = wrapped_cls(api_key="mk", reflexio_timeout=1.25)
    assert client.reflexio.configured is True
    assert constructor_calls == [{"timeout": 1.25}]


@pytest.mark.parametrize("client_fixture", ["wrapped_cls", "async_wrapped_cls"])
@pytest.mark.parametrize(
    ("constructor_kwargs", "expected_reflexio_kwargs"),
    [
        (
            {"reflexio_api_key": "rflx-test"},
            {"api_key": "rflx-test", "timeout": 5.0},
        ),
        (
            {
                "reflexio_api_key": "rflx-test",
                "reflexio_url_endpoint": "http://localhost:8081",
                "reflexio_timeout": 1.25,
            },
            {
                "api_key": "rflx-test",
                "url_endpoint": "http://localhost:8081",
                "timeout": 1.25,
            },
        ),
        (
            {"reflexio_url_endpoint": "http://localhost:8081"},
            {"url_endpoint": "http://localhost:8081", "timeout": 5.0},
        ),
    ],
)
def test_inline_reflexio_configuration_constructs_sync_and_async_clients(
    client_fixture,
    constructor_kwargs,
    expected_reflexio_kwargs,
    request,
    monkeypatch,
):
    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    client_class = request.getfixturevalue(client_fixture)
    import reflexio.mem0._wrapper as wrapper_module

    created = types.SimpleNamespace(
        timeout=constructor_kwargs.get("reflexio_timeout", 5.0)
    )
    constructor_calls = []

    def construct(**kwargs):
        constructor_calls.append(kwargs)
        return created

    monkeypatch.setattr(wrapper_module, "ReflexioClient", construct)
    client = client_class(api_key="mk", **constructor_kwargs)

    assert client.reflexio.configured is True
    assert constructor_calls == [expected_reflexio_kwargs]


def test_inline_key_without_url_preserves_reflexio_sdk_default(
    wrapped_cls, monkeypatch
):
    from reflexio.client.client import BACKEND_URL

    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)

    client = wrapped_cls(api_key="mk", reflexio_api_key="rflx-test")

    assert client._reflexio_client is not None
    assert client._reflexio_client.api_key == "rflx-test"
    assert client._reflexio_client.base_url == BACKEND_URL


@pytest.mark.parametrize(
    "inline_configuration",
    [
        {"reflexio_api_key": "rflx-test"},
        {"reflexio_url_endpoint": "http://localhost:8081"},
        {"reflexio_timeout": 1.0},
    ],
)
def test_inline_configuration_and_injected_client_are_mutually_exclusive(
    wrapped_cls, reflexio_mock, inline_configuration
):
    with pytest.raises(ValueError, match="cannot be combined"):
        wrapped_cls(
            api_key="mk",
            reflexio_client=reflexio_mock,
            **inline_configuration,
        )


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), "5"])
def test_invalid_timeout_rejected(wrapped_cls, timeout):
    with pytest.raises(ValueError, match="finite positive"):
        wrapped_cls(api_key="mk", reflexio_timeout=timeout)


def test_publish_success_false_is_logged_and_mem0_result_returned(
    wrapped_cls, reflexio_mock, caplog
):
    reflexio_mock.publish_interaction.return_value.success = False
    client = _client(wrapped_cls, reflexio_mock)
    with caplog.at_level("WARNING"):
        result = client.add("hi", user_id="u1")
    assert result is client.add_result
    assert "Best-effort Reflexio publish failed" in caplog.text


def test_timestamp_kwargs_overrides_options(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add(
        "hi",
        types.SimpleNamespace(timestamp=1720000000),
        user_id="u1",
        timestamp=1730000000,
    )
    interactions = reflexio_mock.publish_interaction.call_args.kwargs["interactions"]
    assert interactions == [{"role": "User", "content": "hi", "created_at": 1730000000}]


def test_role_none_defaults_to_user(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add([{"role": None, "content": "hi"}], user_id="u1")
    interactions = reflexio_mock.publish_interaction.call_args.kwargs["interactions"]
    assert interactions == [{"role": "User", "content": "hi"}]


def test_repeat_failures_warn_once_then_debug(wrapped_cls, reflexio_mock, caplog):
    reflexio_mock.publish_interaction.side_effect = RuntimeError("reflexio down")
    reflexio_mock.search.side_effect = RuntimeError("reflexio down")
    client = _client(wrapped_cls, reflexio_mock)
    with caplog.at_level("DEBUG", logger="reflexio.mem0._wrapper"):
        client.add("hi", user_id="u1")
        client.add("hi again", user_id="u1")
        client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    failures = [
        r
        for r in caplog.records
        if "failed" in r.getMessage() and "Reflexio" in r.getMessage()
    ]
    assert [r.levelname for r in failures] == ["WARNING", "DEBUG", "DEBUG"]
