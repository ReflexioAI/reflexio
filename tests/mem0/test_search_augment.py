"""Opt-in Reflexio search augmentation without changing normal mem0 search."""

import types

import pytest


def _client(wrapped_cls, reflexio_mock):
    return wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)


def test_search_defaults_to_exact_mem0_result(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search("jazz", filters={"user_id": "u1"})
    assert result is client.search_result
    assert client.calls[-1] == (
        "search",
        "jazz",
        None,
        {"filters": {"user_id": "u1"}},
    )
    reflexio_mock.search.assert_not_called()


def test_opt_in_search_adds_one_namespace_to_shallow_copy(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search(
        "jazz",
        filters={"user_id": "u1", "agent_id": "support-bot"},
        include_reflexio=True,
    )
    assert result is not client.search_result
    assert result["results"] is client.search_result["results"]
    assert "reflexio" not in client.search_result
    assert client.calls[-1] == (
        "search",
        "jazz",
        None,
        {"filters": {"user_id": "u1", "agent_id": "support-bot"}},
    )
    assert result["reflexio"] == {
        "status": "ok",
        "reason": None,
        "profiles": [{"profile_id": "p1", "content": "prefers jazz"}],
        "user_playbooks": [{"user_playbook_id": 7, "content": "greet by name"}],
        "agent_playbooks": [{"agent_playbook_id": 3, "content": "confirm order"}],
    }
    reflexio_mock.search.assert_called_once_with(
        query="jazz", user_id="u1", agent_version="support-bot", top_k=5
    )


def test_user_and_agent_from_and_clause(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search(
        "q",
        filters={"AND": [{"agent_id": "a1"}, {"user_id": "u2"}]},
        include_reflexio=True,
    )
    assert reflexio_mock.search.call_args.kwargs["user_id"] == "u2"
    assert reflexio_mock.search.call_args.kwargs["agent_version"] == "a1"


def test_filters_from_options_and_kwargs_precedence(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search(
        "q",
        types.SimpleNamespace(
            filters={
                "AND": [
                    {"user_id": "u1", "agent_id": "a1"},
                    {"user_id": "u2", "agent_id": "a2"},
                ]
            }
        ),
        filters={"user_id": "u-kw", "agent_id": "a-kw"},
        include_reflexio=True,
    )
    assert reflexio_mock.search.call_args.kwargs["user_id"] == "u-kw"
    assert reflexio_mock.search.call_args.kwargs["agent_version"] == "a-kw"


def test_search_defaults_agent_version_without_agent_id(
    wrapped_cls, reflexio_mock, monkeypatch
):
    monkeypatch.delenv("REFLEXIO_AGENT_VERSION", raising=False)
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert reflexio_mock.search.call_args.kwargs["agent_version"] == "agent-v0"


@pytest.mark.parametrize(
    ("query", "filters", "reason"),
    [
        ("", {"user_id": "u1"}, "empty_query"),
        ("   ", {"user_id": "u1"}, "empty_query"),
        ("q", None, "missing_user_id"),
        ("q", {"agent_id": "a1"}, "missing_user_id"),
        ("q", {"user_id": {"in": ["u1", "u2"]}}, "unsupported_identity_filter"),
        (
            "q",
            {"OR": [{"user_id": "u1"}, {"user_id": "u2"}]},
            "unsupported_identity_filter",
        ),
    ],
)
def test_skipped_search_has_stable_envelope(
    query, filters, reason, wrapped_cls, reflexio_mock
):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search(query, filters=filters, include_reflexio=True)["reflexio"]
    assert result == {
        "status": "skipped",
        "reason": reason,
        "profiles": [],
        "user_playbooks": [],
        "agent_playbooks": [],
    }
    reflexio_mock.search.assert_not_called()


@pytest.mark.parametrize("identity", ["user_id", "agent_id", "app_id", "run_id"])
def test_conflicting_filter_identities_skip_augmentation(
    identity, wrapped_cls, reflexio_mock
):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search(
        "q",
        filters={
            "user_id": "u1",
            "AND": [{identity: "first"}, {identity: "second"}],
        },
        include_reflexio=True,
    )
    assert result["reflexio"]["status"] == "skipped"
    assert result["reflexio"]["reason"] == "conflicting_identity"
    reflexio_mock.search.assert_not_called()


def test_reflexio_failure_is_observable_without_mutating_mem0(
    wrapped_cls, reflexio_mock, caplog
):
    reflexio_mock.search.side_effect = RuntimeError("secret endpoint failure")
    client = _client(wrapped_cls, reflexio_mock)
    with caplog.at_level("WARNING"):
        result = client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert result["results"] is client.search_result["results"]
    assert result["reflexio"] == {
        "status": "error",
        "reason": "request_failed",
        "profiles": [],
        "user_playbooks": [],
        "agent_playbooks": [],
    }
    assert "Best-effort Reflexio search failed" in caplog.text
    assert "secret endpoint failure" not in caplog.text


def test_unsuccessful_response_is_not_reported_as_empty_success(
    wrapped_cls, reflexio_mock
):
    reflexio_mock.search.return_value.success = False
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert result["reflexio"]["status"] == "error"
    assert result["reflexio"]["reason"] == "reflexio_rejected"


def test_unconfigured_opt_in_is_explicit(wrapped_cls, monkeypatch):
    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    client = wrapped_cls(api_key="mk")
    result = client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert result["reflexio"]["status"] == "skipped"
    assert result["reflexio"]["reason"] == "not_configured"


def test_namespace_collision_raises_without_reflexio_call(wrapped_cls, reflexio_mock):
    from reflexio.mem0 import ReflexioNamespaceCollisionError

    client = _client(wrapped_cls, reflexio_mock)
    client.search_result["reflexio"] = {"owned_by": "mem0"}
    with pytest.raises(
        ReflexioNamespaceCollisionError, match="reserved 'reflexio' key"
    ):
        client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert client.search_result["reflexio"] == {"owned_by": "mem0"}
    reflexio_mock.search.assert_not_called()


def test_mem0_search_error_propagates_without_reflexio_call(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.raise_on_search = ValueError("bad query")
    with pytest.raises(ValueError, match="bad query"):
        client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    reflexio_mock.search.assert_not_called()


def test_get_all_is_not_augmented(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.get_all(filters={"user_id": "u1"})
    assert result is client.get_all_result
    assert "reflexio" not in result
    reflexio_mock.search.assert_not_called()
