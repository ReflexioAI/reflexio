"""The mirrored mem0 search must not arm the server's session dedup.

Reflexio records a user-playbook exposure for every search that serves one, and
joins those rows to interactions by ``session_id`` (or ``request_id``). A search
sent with neither is refused outright, so it records nothing.

That makes passing a session id look like a free win, and ``_prepare_publish``
already derives exactly such an id from the mem0 identities. It is not free. On
the server a ``session_id`` is not only a correlation key: it also arms the
session-scoped seen-result cache (``services/retrieval/session_dedup.py``),
which skips items already served to that session and keeps an item "suppressed
for the session's lifetime". With a stable per-run id, the second and later
searches of one conversation would quietly return different, lower-ranked
learnings.

mem0's ``search`` contract is "the most relevant memories", not "the most
relevant ones you have not already seen", and this wrapper's whole premise is
that mem0 behaviour is unchanged. So the mirrored search stays **uncorrelated**
until the server can accept a correlation key without arming dedup: retrieval
quality is the product, correlation is instrumentation, and trading the first
for the second silently is the wrong way round.

These tests pin that trade so it cannot be undone by accident -- and so that
whoever reinstates the id does it deliberately, once the server offers a way.
"""

import pytest


def _client(wrapped_cls, reflexio_mock):
    return wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)


@pytest.mark.parametrize(
    "filters",
    [
        pytest.param({"user_id": "u1"}, id="user-only"),
        pytest.param({"user_id": "u1", "agent_id": "a1"}, id="user-and-agent"),
        pytest.param({"user_id": "u1", "app_id": "storefront"}, id="scoped-user"),
        pytest.param(
            {"user_id": "u1", "agent_id": "a1", "run_id": "run-42"},
            id="explicit-run",
        ),
    ],
)
def test_the_mirrored_search_sends_no_session_id(wrapped_cls, reflexio_mock, filters):
    """Including when a ``run_id`` makes one trivially derivable."""
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters=filters, include_reflexio=True)

    assert "session_id" not in reflexio_mock.search.call_args.kwargs


@pytest.mark.asyncio
async def test_the_async_mirrored_search_sends_no_session_id(
    async_wrapped_cls, reflexio_mock
):
    """The async path is a separate call site and needs its own guard.

    Written after a mutation proved it: deleting ``session_id=`` from the async
    branch alone once left the whole suite green, because every other test here
    drives the sync client. Two call sites, two guards -- in both directions.
    """
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    await client.search(
        "q",
        filters={"user_id": "u1", "agent_id": "a1", "run_id": "run-42"},
        include_reflexio=True,
    )

    assert "session_id" not in reflexio_mock.search_async.call_args.kwargs


def test_publish_still_carries_its_session(wrapped_cls, reflexio_mock):
    """The trade is scoped to search. Publish is unaffected and still grouped.

    Without this, dropping the session from BOTH paths would satisfy every
    assertion above while quietly destroying interaction grouping.
    """
    client = _client(wrapped_cls, reflexio_mock)
    client.add([{"role": "user", "content": "hello"}], user_id="u1", agent_id="a1")

    session_id = reflexio_mock.publish_interaction.call_args.kwargs["session_id"]
    assert session_id.startswith("mem0-run-v1-")


def test_a_plain_search_still_makes_no_reflexio_call(wrapped_cls, reflexio_mock):
    """The opt-in must not have become always-on."""
    client = _client(wrapped_cls, reflexio_mock)
    client.search("q", filters={"user_id": "u1"})
    reflexio_mock.search.assert_not_called()


def test_a_skipped_search_still_makes_no_reflexio_call(wrapped_cls, reflexio_mock):
    """No user id means nothing resolvable, so no request at all."""
    client = _client(wrapped_cls, reflexio_mock)
    result = client.search("q", filters={"agent_id": "a1"}, include_reflexio=True)
    assert result["reflexio"]["reason"] == "missing_user_id"
    reflexio_mock.search.assert_not_called()
