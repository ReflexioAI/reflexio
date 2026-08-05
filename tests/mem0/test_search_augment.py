"""search() augmentation with Reflexio sibling keys; get_all() untouched."""

import copy
import types


def _client(wrapped_cls, reflexio_mock):
    return wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)


def test_search_adds_sibling_keys_and_keeps_mem0_payload(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    original = copy.deepcopy(client.search_result)
    result = client.search("jazz", filters={"user_id": "u1"})
    assert result["results"] == original["results"]
    assert result["reflexio_profiles"] == [
        {"profile_id": "p1", "content": "prefers jazz"}
    ]
    assert result["reflexio_user_playbooks"] == [
        {"user_playbook_id": 7, "content": "greet by name"}
    ]
    assert result["reflexio_agent_playbooks"] == [
        {"agent_playbook_id": 3, "content": "confirm order"}
    ]
    reflexio_mock.search.assert_called_once_with(query="jazz", user_id="u1", top_k=5)


def test_user_id_from_and_clause(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters={"AND": [{"agent_id": "a1"}, {"user_id": "u2"}]})
    assert reflexio_mock.search.call_args.kwargs["user_id"] == "u2"


def test_filters_from_options_model(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", types.SimpleNamespace(filters={"user_id": "u3"}))
    assert reflexio_mock.search.call_args.kwargs["user_id"] == "u3"


def test_filters_kwargs_override_options(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search(
        "q",
        types.SimpleNamespace(filters={"user_id": "u-opt"}),
        filters={"user_id": "u-kw"},
    )
    assert reflexio_mock.search.call_args.kwargs["user_id"] == "u-kw"


def test_empty_query_skips_augmentation(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    for query in ("", "   "):
        result = client.search(query, filters={"user_id": "u1"})
        assert "reflexio_profiles" not in result
    reflexio_mock.search.assert_not_called()


def test_no_plain_user_id_skips_augmentation(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    for filters in (None, {"agent_id": "a1"}, {"user_id": {"in": ["u1", "u2"]}}):
        result = client.search("q", filters=filters)
        assert "reflexio_profiles" not in result
    reflexio_mock.search.assert_not_called()


def test_reflexio_failure_leaves_payload_untouched(wrapped_cls, reflexio_mock, caplog):
    reflexio_mock.search.side_effect = RuntimeError("reflexio down")
    client = _client(wrapped_cls, reflexio_mock)
    original = copy.deepcopy(client.search_result)
    with caplog.at_level("WARNING"):
        result = client.search("q", filters={"user_id": "u1"})
    assert result == original
    assert "Reflexio search failed" in caplog.text


def test_get_all_is_not_augmented(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    result = client.get_all(filters={"user_id": "u1"})
    assert result is client.get_all_result
    assert "reflexio_profiles" not in result
    reflexio_mock.search.assert_not_called()
