"""A mirrored search must correlate with the publish that follows it.

Reflexio records a user-playbook exposure for every search that serves one, and
joins those rows to interactions by ``session_id`` (or ``request_id``). A
search sent with neither is recorded and then permanently unjoinable.

The mem0 wrapper used to be a clean instance of that defect, and an especially
avoidable one: ``_prepare_publish`` derived a ``session_id`` from
``(namespace, resolved_user, agent_version, run_id)`` and sent it, while
``_prepare_search`` resolved the *same* identities off the *same* client,
dropped ``run_id``, and returned only ``(user_id, agent_version)``. The session
was in hand on both paths and forwarded on exactly one.

The assertion that matters is not "search sends a session_id" -- any constant
would satisfy that -- but that search and publish on one client resolve to the
SAME session_id. That equality is what makes the exposure joinable; a unique
id per call would be just as useless as none.
"""

import pytest


def _client(wrapped_cls, reflexio_mock):
    return wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)


def _search_session(reflexio_mock) -> str:
    return reflexio_mock.search.call_args.kwargs["session_id"]


def _publish_session(reflexio_mock) -> str:
    return reflexio_mock.publish_interaction.call_args.kwargs["session_id"]


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param({"user_id": "u1"}, id="user-only"),
        pytest.param({"user_id": "u1", "agent_id": "a1"}, id="user-and-agent"),
        pytest.param({"user_id": "u1", "app_id": "storefront"}, id="scoped-user"),
        pytest.param(
            {"user_id": "u1", "agent_id": "a1", "app_id": "storefront"},
            id="fully-scoped",
        ),
    ],
)
def test_search_and_publish_resolve_the_same_session(
    wrapped_cls, reflexio_mock, filters
):
    """The join key must match across the two calls of one turn."""
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters=filters, include_reflexio=True)
    client.add([{"role": "user", "content": "hello"}], **filters)

    assert _search_session(reflexio_mock) == _publish_session(reflexio_mock)


def test_a_run_id_scopes_the_search_session_the_same_way_it_scopes_publish(
    wrapped_cls, reflexio_mock
):
    """``run_id`` was the field the search path dropped outright."""
    client = _client(wrapped_cls, reflexio_mock)
    scoped = {"user_id": "u1", "agent_id": "a1", "run_id": "run-42"}
    client.search("q", filters=scoped, include_reflexio=True)
    client.add([{"role": "user", "content": "hello"}], **scoped)
    with_run = _search_session(reflexio_mock)
    assert with_run == _publish_session(reflexio_mock)

    reflexio_mock.search.reset_mock()
    client.search(
        "q", filters={"user_id": "u1", "agent_id": "a1"}, include_reflexio=True
    )
    # A different run is a different session; dropping run_id would collapse
    # every run of one user+agent into one bucket.
    assert _search_session(reflexio_mock) != with_run


def test_different_users_do_not_share_a_search_session(wrapped_cls, reflexio_mock):
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    first = _search_session(reflexio_mock)
    client.search("q", filters={"user_id": "u2"}, include_reflexio=True)
    assert _search_session(reflexio_mock) != first


def test_repeated_searches_in_one_session_keep_one_session_id(
    wrapped_cls, reflexio_mock
):
    """Stability is the point: a per-call id correlates nothing."""
    client = _client(wrapped_cls, reflexio_mock)
    filters = {"user_id": "u1", "agent_id": "a1"}
    client.search("first", filters=filters, include_reflexio=True)
    first = _search_session(reflexio_mock)
    client.search("second", filters=filters, include_reflexio=True)
    assert _search_session(reflexio_mock) == first


def test_a_plain_search_still_makes_no_reflexio_call(wrapped_cls, reflexio_mock):
    """Correlation must not have turned the opt-in into an always-on call."""
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters={"user_id": "u1"})
    reflexio_mock.search.assert_not_called()


def test_a_skipped_search_sends_nothing_rather_than_a_bare_session(
    wrapped_cls, reflexio_mock
):
    """No user id means no resolvable session, so no request at all."""
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search("q", filters={"agent_id": "a1"}, include_reflexio=True)
    assert result["reflexio"]["reason"] == "missing_user_id"
    reflexio_mock.search.assert_not_called()


@pytest.mark.asyncio
async def test_async_search_and_publish_resolve_the_same_session(
    async_wrapped_cls, reflexio_mock
):
    """The async path is a separate call site and needs its own guard.

    Written after a mutation proved it: deleting ``session_id=`` from the async
    branch alone left the whole suite green, because every other test here
    drives the sync client. Two call sites, two guards.
    """
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    filters = {"user_id": "u1", "agent_id": "a1"}
    await client.search("q", filters=filters, include_reflexio=True)
    await client.add([{"role": "user", "content": "hello"}], **filters)

    search_session = reflexio_mock.search_async.call_args.kwargs["session_id"]
    publish_session = reflexio_mock.publish_interaction_async.call_args.kwargs[
        "session_id"
    ]
    assert search_session.startswith("mem0-run-v1-")
    assert search_session == publish_session


@pytest.mark.asyncio
async def test_async_run_id_scopes_the_search_session(async_wrapped_cls, reflexio_mock):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    await client.search(
        "q",
        filters={"user_id": "u1", "agent_id": "a1", "run_id": "run-42"},
        include_reflexio=True,
    )
    with_run = reflexio_mock.search_async.call_args.kwargs["session_id"]
    await client.search(
        "q", filters={"user_id": "u1", "agent_id": "a1"}, include_reflexio=True
    )
    assert reflexio_mock.search_async.call_args.kwargs["session_id"] != with_run
