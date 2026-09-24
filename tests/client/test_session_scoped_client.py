"""Session-scoped client: one bound session ID, shared by every call.

The load-bearing property here is **not** that a call sends *a* session ID.
It is that two calls in the same turn send the **same** one. An assertion
that merely checks presence passes against an implementation that mints a
fresh ID per call, which is exactly the defect a scoped client exists to
prevent -- so the agreement assertions below compare the two payloads to
each other rather than to a literal.
"""

import inspect
from typing import Any

import pytest

from reflexio import ReflexioClient, SessionScopedClient
from reflexio.models.api_schema.retriever_schema import (
    GetRequestsRequest,
    UnifiedSearchRequest,
)

BOUND = "session-alpha"
OTHER = "session-beta"

INTERACTIONS = [{"role": "user", "content": "how do I get a refund?"}]

# ``for_session`` takes a session_id because it is the binding entry point
# itself, not a call that needs binding. It is the only legitimate exemption.
BINDING_ENTRY_POINT = {"for_session"}

# Every other ReflexioClient method that takes a session-correlation argument.
# Pinned so that adding one upstream fails here until it is given an
# explicit binding on SessionScopedClient rather than silently reaching
# callers uncorrelated.
SESSION_TAKING_CLIENT_METHODS = {
    "delete_session",
    "get_requests",
    "get_retrieved_learning_evaluation_results",
    "get_session_outcomes",
    "grade_on_demand",
    "mark_session_outcome",
    "publish_interaction",
    "publish_interaction_async",
    "search",
    "search_async",
    "search_user_playbooks",
}

_RESPONSES: dict[str, dict[str, Any]] = {
    "/api/search": {
        "success": True,
        "profiles": [],
        "agent_playbooks": [],
        "user_playbooks": [],
    },
    "/api/search_user_playbooks": {"success": True, "user_playbooks": []},
    "/api/publish_interaction": {"success": True},
    "/api/session_outcome": {"success": True, "recorded": True},
    "/api/get_session_outcomes": {"success": True, "outcomes": []},
    "/api/get_requests": {"success": True, "sessions": []},
    "/api/get_retrieved_learning_evaluation_results": {"success": True, "results": []},
    "/api/evaluations/grade_on_demand": {"session_id": BOUND},
    "/api/delete_session": {"success": True},
}


def _make_client() -> ReflexioClient:
    return ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")


def _capture_sync(monkeypatch, client: ReflexioClient) -> dict[str, dict[str, Any]]:
    """Record the JSON body of every sync request, keyed by path."""
    captured: dict[str, dict[str, Any]] = {}

    def fake_make_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        captured[path] = kwargs.get("json", {})
        return _RESPONSES[path]

    monkeypatch.setattr(client, "_make_request", fake_make_request)
    return captured


def _capture_async(monkeypatch, client: ReflexioClient) -> dict[str, dict[str, Any]]:
    """Record the JSON body of every native-async request, keyed by path."""
    captured: dict[str, dict[str, Any]] = {}

    async def fake_make_async_request(
        method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        captured[path] = kwargs.get("json", {})
        return _RESPONSES[path]

    monkeypatch.setattr(client, "_make_async_request", fake_make_async_request)
    return captured


# ---------------------------------------------------------------------------
# The invariant: search and publish resolve to the SAME session
# ---------------------------------------------------------------------------


def test_scoped_search_and_publish_send_one_and_the_same_session(monkeypatch) -> None:
    """The two calls must agree, not merely each carry something.

    This is the assertion that fails if the wrapper ever derives the
    session per call instead of from the single bound value.
    """
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    session = client.for_session(BOUND)
    session.search(query="refunds", user_id="u1")
    session.publish_interaction(user_id="u1", interactions=INTERACTIONS)

    search_session = captured["/api/search"]["session_id"]
    publish_session = captured["/api/publish_interaction"]["session_id"]

    assert search_session == publish_session
    assert search_session == BOUND


@pytest.mark.asyncio
async def test_scoped_async_search_and_publish_send_one_session(monkeypatch) -> None:
    """Same agreement invariant, driven entirely through the async methods."""
    client = _make_client()
    captured = _capture_async(monkeypatch, client)

    session = client.for_session(BOUND)
    await session.search_async(query="refunds", user_id="u1")
    await session.publish_interaction_async(user_id="u1", interactions=INTERACTIONS)

    search_session = captured["/api/search"]["session_id"]
    publish_session = captured["/api/publish_interaction"]["session_id"]

    assert search_session == publish_session
    assert search_session == BOUND


def test_sync_and_async_scoped_calls_agree_with_each_other(monkeypatch) -> None:
    """A turn that searches sync and publishes async still correlates."""
    import asyncio

    client = _make_client()
    sync_captured = _capture_sync(monkeypatch, client)
    async_captured = _capture_async(monkeypatch, client)

    session = client.for_session(BOUND)
    session.search(query="refunds", user_id="u1")
    asyncio.run(
        session.publish_interaction_async(user_id="u1", interactions=INTERACTIONS)
    )

    assert (
        sync_captured["/api/search"]["session_id"]
        == async_captured["/api/publish_interaction"]["session_id"]
    )


def test_whole_session_lifecycle_shares_one_session(monkeypatch) -> None:
    """Every scoped method in a realistic turn resolves to the same ID.

    Compares the full set of emitted session values to a single-element
    set, so any method that derives its own value breaks the test
    regardless of which one it is.
    """
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    session = client.for_session(BOUND)
    session.search(query="refunds", user_id="u1")
    session.search_user_playbooks(query="refunds", user_id="u1")
    session.publish_interaction(user_id="u1", interactions=INTERACTIONS)
    session.mark_session_outcome(outcome="success", occurred_at=1)
    session.get_requests(user_id="u1")
    session.get_retrieved_learning_evaluation_results(user_id="u1")
    session.grade_on_demand(agent_version="v1")

    emitted = {
        payload["session_id"]
        for path, payload in captured.items()
        if "session_id" in payload
    }
    assert emitted == {BOUND}
    # Guard the guard: if nothing were captured, the set comparison above
    # would be vacuous only if it compared to an empty set -- it does not,
    # but pin the breadth anyway so a silently-skipped call is visible.
    assert len(captured) == 7


def test_scoped_reads_bind_the_plural_session_filter(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.for_session(BOUND).get_session_outcomes()

    assert captured["/api/get_session_outcomes"]["session_ids"] == [BOUND]


def test_scoped_delete_session_targets_the_bound_session(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.for_session(BOUND).delete_session(wait_for_response=True)

    assert captured["/api/delete_session"]["session_id"] == BOUND


# ---------------------------------------------------------------------------
# The request-object form must be bound too
# ---------------------------------------------------------------------------


def test_request_object_without_a_session_is_bound(monkeypatch) -> None:
    """``search(request=...)`` bypasses kwargs, so binding must reach inside."""
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    session = client.for_session(BOUND)
    session.search(UnifiedSearchRequest(query="refunds"))
    session.publish_interaction(user_id="u1", interactions=INTERACTIONS)

    assert (
        captured["/api/search"]["session_id"]
        == captured["/api/publish_interaction"]["session_id"]
        == BOUND
    )


def test_request_dict_without_a_session_is_bound(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.for_session(BOUND).get_requests({"user_id": "u1"})

    assert captured["/api/get_requests"]["session_id"] == BOUND


def test_request_object_carrying_the_bound_session_is_accepted(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.for_session(BOUND).search(
        UnifiedSearchRequest(query="refunds", session_id=BOUND)
    )

    assert captured["/api/search"]["session_id"] == BOUND


def test_request_object_carrying_a_different_session_is_refused() -> None:
    session = _make_client().for_session(BOUND)

    with pytest.raises(ValueError, match="cannot belong to two sessions"):
        session.search(UnifiedSearchRequest(query="refunds", session_id=OTHER))


def test_request_dict_carrying_a_different_session_is_refused() -> None:
    session = _make_client().for_session(BOUND)

    with pytest.raises(ValueError, match="cannot belong to two sessions"):
        session.get_requests({"user_id": "u1", "session_id": OTHER})


def test_binding_a_request_object_does_not_mutate_the_caller_s_object(
    monkeypatch,
) -> None:
    client = _make_client()
    _capture_sync(monkeypatch, client)
    request = UnifiedSearchRequest(query="refunds")

    client.for_session(BOUND).search(request)

    assert request.session_id is None


# ---------------------------------------------------------------------------
# Conflicting explicit session IDs are refused, not silently preferred
# ---------------------------------------------------------------------------


CONFLICTING_CALLS = {
    "search": lambda s: s.search(query="q", session_id=OTHER),
    "search_user_playbooks": lambda s: s.search_user_playbooks(
        query="q", session_id=OTHER
    ),
    "publish_interaction": lambda s: s.publish_interaction(
        user_id="u1", interactions=INTERACTIONS, session_id=OTHER
    ),
    "mark_session_outcome": lambda s: s.mark_session_outcome(
        outcome="success", occurred_at=1, session_id=OTHER
    ),
    "delete_session": lambda s: s.delete_session(OTHER),
    "get_requests": lambda s: s.get_requests(session_id=OTHER),
    "get_retrieved_learning_evaluation_results": (
        lambda s: s.get_retrieved_learning_evaluation_results(session_id=OTHER)
    ),
    "grade_on_demand": lambda s: s.grade_on_demand(session_id=OTHER),
}


@pytest.mark.parametrize("method_name", sorted(CONFLICTING_CALLS))
def test_conflicting_session_id_raises(monkeypatch, method_name: str) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)
    session = client.for_session(BOUND)

    with pytest.raises(ValueError, match="cannot belong to two sessions"):
        CONFLICTING_CALLS[method_name](session)

    assert captured == {}, "a refused call must not reach the network"


def test_conflicting_session_ids_filter_raises(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    with pytest.raises(ValueError, match="covers the bound session only"):
        client.for_session(BOUND).get_session_outcomes(session_ids=[OTHER])

    assert captured == {}


@pytest.mark.asyncio
async def test_conflicting_session_id_raises_on_the_async_path(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_async(monkeypatch, client)
    session = client.for_session(BOUND)

    with pytest.raises(ValueError, match="cannot belong to two sessions"):
        await session.search_async(query="q", session_id=OTHER)
    with pytest.raises(ValueError, match="cannot belong to two sessions"):
        await session.publish_interaction_async(
            user_id="u1", interactions=INTERACTIONS, session_id=OTHER
        )

    assert captured == {}


def test_explicit_session_id_equal_to_the_bound_one_is_accepted(monkeypatch) -> None:
    """Existing call sites can be wrapped without being edited first."""
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    session = client.for_session(BOUND)
    session.search(query="q", session_id=BOUND)
    session.publish_interaction(
        user_id="u1", interactions=INTERACTIONS, session_id=BOUND
    )

    assert (
        captured["/api/search"]["session_id"]
        == captured["/api/publish_interaction"]["session_id"]
        == BOUND
    )


# ---------------------------------------------------------------------------
# Binding, sharing, and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["", "   ", None])
def test_for_session_rejects_an_empty_session_id(bad) -> None:
    with pytest.raises(ValueError, match="session_id is required"):
        _make_client().for_session(bad)


def test_scoped_client_shares_the_underlying_client() -> None:
    """A binding of one argument, not a second client."""
    client = _make_client()
    session = client.for_session(BOUND)

    assert isinstance(session, SessionScopedClient)
    assert session.client is client
    assert session.session_id == BOUND
    # Connection pool, auth and cache are the very same objects.
    assert session.client.session is client.session
    assert session.client.api_key == client.api_key
    assert session.client._cache is client._cache


def test_two_scoped_views_share_one_connection_pool() -> None:
    client = _make_client()

    assert (
        client.for_session("a").client.session is client.for_session("b").client.session
    )


def test_repr_names_the_bound_session() -> None:
    assert repr(_make_client().for_session(BOUND)) == (
        f"SessionScopedClient(session_id={BOUND!r})"
    )


# ---------------------------------------------------------------------------
# Forwarding, and the drift guard on it
# ---------------------------------------------------------------------------


def test_session_free_methods_are_forwarded(monkeypatch) -> None:
    client = _make_client()
    calls: list[str] = []
    monkeypatch.setattr(client, "whoami", lambda: calls.append("whoami") or "forwarded")

    assert client.for_session(BOUND).whoami() == "forwarded"
    assert calls == ["whoami"]


def test_forwarding_refuses_an_unbound_session_taking_method() -> None:
    """A new session-taking method must not reach callers via __getattr__."""
    client = _make_client()

    def newly_added(session_id: str | None = None) -> str:
        return "uncorrelated"

    client.newly_added = newly_added  # type: ignore[attr-defined]

    with pytest.raises(AttributeError, match="no binding for 'newly_added'"):
        _ = client.for_session(BOUND).newly_added


def _session_taking_methods(cls: type) -> set[str]:
    found = set()
    for name, member in inspect.getmembers(cls, callable):
        if name.startswith("_"):
            continue
        try:
            parameters = inspect.signature(member).parameters
        except (TypeError, ValueError):
            continue
        if parameters.keys() & {"session_id", "session_ids"}:
            found.add(name)
    return found - BINDING_ENTRY_POINT


def test_every_session_taking_client_method_has_a_scoped_binding() -> None:
    """Drift guard: no session-taking method may lack an explicit wrapper."""
    discovered = _session_taking_methods(ReflexioClient)

    assert discovered, "discovery found nothing -- the scan itself is broken"
    assert discovered == SESSION_TAKING_CLIENT_METHODS, (
        "ReflexioClient's session-taking methods changed. Add an explicit "
        "binding to SessionScopedClient and update "
        "SESSION_TAKING_CLIENT_METHODS."
    )

    unbound = {name for name in discovered if name not in vars(SessionScopedClient)}
    assert not unbound, (
        f"{sorted(unbound)} take a session argument but have no explicit "
        "wrapper on SessionScopedClient, so a scoped call would go out "
        "uncorrelated."
    )


def test_scoped_bindings_accept_every_parameter_of_the_method_they_wrap() -> None:
    """A wrapper that silently drops an upstream parameter is a trap."""
    for name in SESSION_TAKING_CLIENT_METHODS:
        original = inspect.signature(getattr(ReflexioClient, name)).parameters
        wrapper = inspect.signature(getattr(SessionScopedClient, name)).parameters
        missing = original.keys() - wrapper.keys()
        assert not missing, f"{name} wrapper drops {sorted(missing)}"


# ---------------------------------------------------------------------------
# Purely additive: the per-call form keeps working exactly as before
# ---------------------------------------------------------------------------


def test_per_call_session_id_still_works_on_the_unscoped_client(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.search(query="q", session_id="legacy-session")
    client.publish_interaction(
        user_id="u1", interactions=INTERACTIONS, session_id="legacy-session"
    )

    assert captured["/api/search"]["session_id"] == "legacy-session"
    assert captured["/api/publish_interaction"]["session_id"] == "legacy-session"


def test_unscoped_search_may_still_omit_the_session(monkeypatch) -> None:
    """The old permissive behaviour is unchanged -- this change adds, never breaks."""
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.search(query="q")

    assert captured["/api/search"]["session_id"] is None


def test_request_objects_reach_the_unscoped_client_untouched(monkeypatch) -> None:
    client = _make_client()
    captured = _capture_sync(monkeypatch, client)

    client.get_requests(GetRequestsRequest(user_id="u1", session_id="legacy"))

    assert captured["/api/get_requests"]["session_id"] == "legacy"
