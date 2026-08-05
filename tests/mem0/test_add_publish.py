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
    assert kwargs["session_id"] == "run-42"
    assert kwargs["agent_version"] == "support-bot"
    assert kwargs["source"] == "mem0"
    assert kwargs["wait_for_response"] is False
    assert kwargs["interactions"] == [{"role": "User", "content": "hello"}]


def test_session_id_stable_per_instance_without_run_id(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.add("a", user_id="u1")
    client.add("b", user_id="u1")
    sessions = [
        c.kwargs["session_id"] for c in reflexio_mock.publish_interaction.call_args_list
    ]
    assert sessions[0] == sessions[1]
    assert sessions[0].startswith("mem0-")

    other = _client(wrapped_cls, reflexio_mock)
    other.add("c", user_id="u1")
    assert (
        reflexio_mock.publish_interaction.call_args.kwargs["session_id"] != sessions[0]
    )


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

    monkeypatch.setattr(
        wrapper_module,
        "ReflexioClient",
        lambda **_: (_ for _ in ()).throw(RuntimeError("no env")),
    )
    client = wrapped_cls(api_key="mk")
    assert client._reflexio is None
    assert client.add("hi", user_id="u1") is client.add_result
