"""ReflexioClient against the real FastAPI app over loopback HTTP.

The other client tests either mock ``requests.Session`` or adapt a Starlette
``TestClient`` into it. Neither exercises the asynchronous transport, which
every ``wait_for_response=False`` call uses (``aiohttp``), and neither sends a
real HTTP request through ``requests``. Here the app runs under uvicorn on a
free loopback port with real SQLite storage, so each call goes through the
transport, the route and storage, and its effect is checked in storage.
"""

from __future__ import annotations

import socket
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from unittest.mock import patch

import pytest
import uvicorn

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    InteractionData,
    OperationStatus,
    PlaybookStatus,
    Status,
    UserPlaybook,
)
from reflexio.models.config_schema import Config
from reflexio.server.api import create_app
from reflexio.server.api_endpoints.request_context import (
    RequestContext,
    get_request_context,
)
from reflexio.server.cache.reflexio_cache import get_reflexio
from reflexio.server.rate_limit import limiter
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@dataclass
class LiveServer:
    client: ReflexioClient
    context: RequestContext
    org_id: str

    @property
    def storage(self) -> SQLiteStorage:
        # Routes reach storage either through the request context or through
        # the cached per-org Reflexio; both must resolve the same SQLite file.
        storage = get_reflexio(org_id=self.org_id)._get_storage()
        assert isinstance(storage, SQLiteStorage)
        context_storage = self.context.storage
        assert isinstance(context_storage, SQLiteStorage)
        assert storage.db_path == context_storage.db_path
        return storage


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate: Callable[[], bool], timeout: float = 15.0) -> None:
    """Poll until ``predicate`` holds; fail the test if it never does."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not reached before timeout")


@pytest.fixture()
def live() -> Iterator[LiveServer]:
    org_id = f"client-live-{uuid.uuid4().hex[:12]}"
    # The expensive-operation routes share one in-process rate limiter.
    limiter.reset()
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.1] * 512):
        # LOCAL_STORAGE_PATH is a per-session temp dir (tests/conftest.py); the
        # unique org id gives this test its own SQLite file inside it.
        context = RequestContext(org_id=org_id)
        app = create_app(get_org_id=lambda: org_id)
        app.dependency_overrides[get_request_context] = lambda: context
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=_free_port(),
                lifespan="off",
                log_level="warning",
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        wait_until(lambda: server.started)
        port = server.servers[0].sockets[0].getsockname()[1]
        client = ReflexioClient(
            api_key="live-test-key", url_endpoint=f"http://127.0.0.1:{port}", timeout=30
        )
        try:
            yield LiveServer(client=client, context=context, org_id=org_id)
        finally:
            server.should_exit = True
            thread.join(timeout=10)
            client.session.close()


def _user_playbook(content: str, request_id: str = "req-1") -> UserPlaybook:
    return UserPlaybook(
        agent_version="v1",
        request_id=request_id,
        user_id="alice",
        content=content,
        trigger="When deploying",
        playbook_name="deploy",
    )


def _agent_playbook(content: str) -> dict:
    return {
        "agent_version": "v1",
        "content": content,
        "trigger": "When deploying",
        "playbook_status": "pending",
    }


def _publish(live: LiveServer, user_id: str, request_session: str, text: str) -> None:
    response = live.client.publish_interaction(
        user_id=user_id,
        session_id=request_session,
        interactions=[
            InteractionData(role="user", content=text),
            InteractionData(role="assistant", content=f"ack: {text}"),
        ],
        skip_aggregation=True,
    )
    assert response.success, response.message


def _request_ids(live: LiveServer, user_id: str) -> set[str]:
    return {
        interaction.request_id
        for interaction in live.storage.get_user_interaction(user_id)
    }


def test_user_playbooks_add_update_and_delete_over_both_transports(live):
    added = live.client.add_user_playbook(
        [
            _user_playbook("Ask before deploying."),
            {**_user_playbook("Run tests.").model_dump()},
        ]
    )
    assert added.success and added.added_count == 2
    ids = sorted(r.user_playbook_id for r in live.storage.get_user_playbooks())
    assert len(ids) == 2

    updated = live.client.update_user_playbook(ids[0], content="Always ask first.")
    assert updated.success
    assert {r.content for r in live.storage.get_user_playbooks()} == {
        "Always ask first.",
        "Run tests.",
    }

    # The default delete is fire-and-forget over the asynchronous transport.
    assert live.client.delete_user_playbook(ids[0]) is None
    wait_until(lambda: len(live.storage.get_user_playbooks()) == 1)

    deleted = live.client.delete_user_playbook(ids[1], wait_for_response=True)
    assert deleted is not None and deleted.success
    assert live.storage.get_user_playbooks() == []


def test_user_playbook_bulk_and_delete_all(live):
    live.client.add_user_playbook(
        [_user_playbook(f"rule {n}", request_id=f"r{n}") for n in range(3)]
    )
    ids = sorted(r.user_playbook_id for r in live.storage.get_user_playbooks())

    bulk = live.client.delete_user_playbooks_by_ids(ids[:2])
    assert bulk.success and bulk.deleted_count == 2
    assert [r.user_playbook_id for r in live.storage.get_user_playbooks()] == ids[2:]

    wiped = live.client.delete_all_user_playbooks()
    assert wiped.success
    assert live.storage.get_user_playbooks() == []


def test_agent_playbooks_add_update_status_and_delete(live):
    added = live.client.add_agent_playbooks(
        [
            _agent_playbook("Use blue-green deploys."),
            AgentPlaybook(**_agent_playbook("Tag every release.")),
        ]
    )
    assert added.success and added.added_count == 2
    ids = sorted(p.agent_playbook_id for p in live.storage.get_agent_playbooks())
    assert len(ids) == 2

    updated = live.client.update_agent_playbook(ids[0], content="Use canary deploys.")
    assert updated.success
    approved = live.client.update_agent_playbook_status(
        ids[0], playbook_status=PlaybookStatus.APPROVED
    )
    assert approved.success
    first = next(
        p for p in live.storage.get_agent_playbooks() if p.agent_playbook_id == ids[0]
    )
    assert first.content == "Use canary deploys."
    assert first.playbook_status == PlaybookStatus.APPROVED

    assert live.client.delete_agent_playbook(ids[0]) is None
    wait_until(lambda: len(live.storage.get_agent_playbooks()) == 1)
    deleted = live.client.delete_agent_playbook(ids[1], wait_for_response=True)
    assert deleted is not None and deleted.success
    assert live.storage.get_agent_playbooks() == []


def test_agent_playbook_bulk_and_delete_all(live):
    live.client.add_agent_playbooks([_agent_playbook(f"rule {n}") for n in range(3)])
    ids = sorted(p.agent_playbook_id for p in live.storage.get_agent_playbooks())

    bulk = live.client.delete_agent_playbooks_by_ids(ids[:1])
    assert bulk.success and bulk.deleted_count == 1
    assert (
        sorted(p.agent_playbook_id for p in live.storage.get_agent_playbooks())
        == (ids[1:])
    )

    wiped = live.client.delete_all_agent_playbooks()
    assert wiped.success
    assert live.storage.get_agent_playbooks() == []


def test_delete_all_playbooks_cascades_to_both_stores(live):
    live.client.add_user_playbook([_user_playbook("user rule")])
    live.client.add_agent_playbooks([_agent_playbook("agent rule")])

    wiped = live.client.delete_all_playbooks()
    assert wiped.success
    assert live.storage.get_user_playbooks() == []
    assert live.storage.get_agent_playbooks() == []


def test_profiles_add_read_filter_and_delete(live):
    added = live.client.add_user_profile(
        [
            {"user_id": "alice", "content": "Prefers dark mode."},
            {"user_id": "alice", "content": "Works in UTC.", "source": "import"},
        ]
    )
    assert added.success and added.added_count == 2
    stored = live.storage.get_user_profile("alice")
    assert {p.content for p in stored} == {"Prefers dark mode.", "Works in UTC."}
    # Dict input is completed client-side with the documented defaults.
    dark = next(p for p in stored if p.content == "Prefers dark mode.")
    assert dark.generated_from_request_id == "cli-manual"
    assert dark.source == "cli-manual"

    current = live.client.get_profiles(user_id="alice", status_filter=[None])
    assert {p.content for p in current.user_profiles} == {
        "Prefers dark mode.",
        "Works in UTC.",
    }
    # Status values are accepted as enums or strings.
    none_pending = live.client.get_profiles(
        user_id="alice", status_filter=["pending", Status.ARCHIVED], force_refresh=True
    )
    assert none_pending.user_profiles == []

    everyone = live.client.get_all_profiles(limit=10, status_filter="current")
    assert len(everyone.user_profiles) == 2
    by_source = live.client.get_all_profiles(
        source="import", user_id="alice", date_field="created_at"
    )
    assert [p.content for p in by_source.user_profiles] == ["Works in UTC."]

    deleted = live.client.delete_profile(
        "alice", profile_id=dark.profile_id, wait_for_response=True
    )
    assert deleted is not None and deleted.success
    assert [p.content for p in live.storage.get_user_profile("alice")] == [
        "Works in UTC."
    ]
    utc = live.storage.get_user_profile("alice")[0]
    assert live.client.delete_profile("alice", profile_id=utc.profile_id) is None
    wait_until(lambda: live.storage.get_user_profile("alice") == [])


def test_get_profiles_serves_repeat_reads_from_cache_until_invalidated(live):
    live.client.add_user_profile([{"user_id": "bob", "content": "Likes tea."}])
    first = live.client.get_profiles(user_id="bob")
    assert [p.content for p in first.user_profiles] == ["Likes tea."]

    # A write that bypasses the client is invisible to the cached read...
    live.client.add_user_profile([{"user_id": "bob", "content": "Likes coffee."}])
    assert live.client.get_profiles(user_id="bob") is first
    # ...until the caller forces a refresh.
    refreshed = live.client.get_profiles(user_id="bob", force_refresh=True)
    assert len(refreshed.user_profiles) == 2

    ids = [p.profile_id for p in refreshed.user_profiles]
    bulk = live.client.delete_profiles_by_ids(ids[:1])
    assert bulk.success and bulk.deleted_count == 1
    # The bulk delete invalidated the cache, so the next read is fresh.
    assert len(live.client.get_profiles(user_id="bob").user_profiles) == 1

    wiped = live.client.delete_all_profiles()
    assert wiped.success
    assert live.client.get_profiles(user_id="bob").user_profiles == []


def _keyword_scores(query: str, documents: list[str]) -> list[float]:
    """Deterministic stand-in for the cross-encoder: word overlap with query."""
    words = set(query.lower().split())
    return [
        float(len(words & set(doc.lower().rstrip(".").split()))) for doc in documents
    ]


def test_profile_change_log_rerank_and_storage_stats(live):
    live.client.add_user_profile(
        [
            {"user_id": "carol", "content": "Deploys on Fridays."},
            {"user_id": "carol", "content": "Uses vim."},
        ]
    )
    by_content = {
        p.content: p.profile_id for p in live.storage.get_user_profile("carol")
    }

    assert live.client.get_profile_change_log().success

    stats = live.client.storage_stats("carol")
    assert stats.profile_count == 2

    with patch("reflexio.server.llm.rerank.score_pairs", side_effect=_keyword_scores):
        reranked = live.client.rerank_user_profiles(
            user_id="carol",
            query="uses which editor vim",
            profile_ids=[*by_content.values(), "missing-profile"],
            top_k=1,
        )
    assert reranked.success
    assert [p.profile_id for p in reranked.user_profiles] == [by_content["Uses vim."]]


def test_rerank_outage_surfaces_as_http_503(live):
    from requests import HTTPError

    from reflexio.server.llm.rerank.common import CrossEncoderUnavailableError

    live.client.add_user_profile([{"user_id": "cy", "content": "Uses vim."}])
    ids = [p.profile_id for p in live.storage.get_user_profile("cy")]
    with (
        patch(
            "reflexio.server.llm.rerank.score_pairs",
            side_effect=CrossEncoderUnavailableError("no reranker"),
        ),
        pytest.raises(HTTPError) as error,
    ):
        live.client.rerank_user_profiles(user_id="cy", query="vim", profile_ids=ids)
    assert error.value.response.status_code == 503


def test_deprecated_search_profiles_alias_still_searches(live):
    live.client.add_user_profile([{"user_id": "dave", "content": "Speaks French."}])
    with pytest.warns(DeprecationWarning, match="search_user_profiles"):
        found = live.client.search_profiles(user_id="dave", query="French")
    assert found.success
    assert [p.content for p in found.user_profiles] == ["Speaks French."]


def _request_of(live: LiveServer, user_id: str, text: str) -> str:
    return next(
        i.request_id
        for i in live.storage.get_user_interaction(user_id)
        if i.content == text
    )


def test_interactions_requests_and_sessions_delete(live):
    for name in ("a", "b", "c", "d"):
        _publish(live, "erin", f"s-{name}", f"text {name}")
    interactions = live.storage.get_user_interaction("erin")
    assert len(interactions) == 8

    everyone = live.client.get_all_interactions(limit=50)
    assert len(everyone.interactions) == 8
    assert live.client.search_interactions(user_id="erin", query="text b").success

    victim = next(i for i in interactions if i.content == "text a")
    deleted = live.client.delete_interaction(
        "erin", victim.interaction_id, wait_for_response=True
    )
    assert deleted is not None and deleted.success
    assert "text a" not in {
        i.content for i in live.storage.get_user_interaction("erin")
    }
    ack = next(
        i
        for i in live.storage.get_user_interaction("erin")
        if i.content == "ack: text a"
    )
    assert live.client.delete_interaction("erin", ack.interaction_id) is None
    wait_until(lambda: len(live.storage.get_user_interaction("erin")) == 6)

    dropped = live.client.delete_session("s-b", wait_for_response=True)
    assert dropped is not None and dropped.success
    assert {i.content for i in live.storage.get_user_interaction("erin")} == {
        "text c",
        "ack: text c",
        "text d",
        "ack: text d",
    }

    request_c = _request_of(live, "erin", "text c")
    request_d = _request_of(live, "erin", "text d")
    gone = live.client.delete_request(request_c, wait_for_response=True)
    assert gone is not None and gone.success
    assert _request_ids(live, "erin") == {request_d}
    assert live.client.delete_request(request_d) is None
    wait_until(lambda: _request_ids(live, "erin") == set())


def test_fire_and_forget_session_and_bulk_request_deletes(live):
    _publish(live, "frank", "s-1", "one")
    _publish(live, "frank", "s-2", "two")
    _publish(live, "frank", "s-3", "three")

    assert live.client.delete_session("s-1") is None
    wait_until(lambda: len(_request_ids(live, "frank")) == 2)

    bulk = live.client.delete_requests_by_ids(sorted(_request_ids(live, "frank"))[:1])
    assert bulk.success and bulk.deleted_count == 1
    assert len(_request_ids(live, "frank")) == 1

    wiped = live.client.delete_all_interactions()
    assert wiped.success
    assert live.storage.get_user_interaction("frank") == []


def _progress(live: LiveServer, service_name: str) -> dict | None:
    return OperationStateManager(live.storage, live.org_id, service_name).get_progress()


def test_rerun_generation_waits_for_the_admitted_attempt(live):
    _publish(live, "gina", "s-1", "I always deploy on Mondays.")

    profile_run = live.client.rerun_profile_generation(
        user_id="gina", wait_for_response=True
    )
    assert profile_run is not None
    assert profile_run.operation_id
    assert profile_run.operation_id != "rerun_profile_generation"
    # The client returned only after the server recorded a terminal state for
    # exactly the attempt it admitted.
    state = _progress(live, "profile_generation")
    assert state is not None
    assert state["request_params"]["operation_id"] == profile_run.operation_id
    assert state["status"] in (
        OperationStatus.COMPLETED.value,
        OperationStatus.FAILED.value,
    )

    playbook_run = live.client.rerun_playbook_generation(
        agent_version="v1", wait_for_response=True
    )
    assert playbook_run is not None
    assert playbook_run.operation_id
    assert playbook_run.operation_id != "rerun_playbook_generation"


def test_fire_and_forget_generation_submissions_reach_the_server(live):
    _publish(live, "hank", "s-1", "Never deploy on Fridays.")

    assert live.client.rerun_profile_generation(user_id="hank") is None
    wait_until(lambda: _progress(live, "profile_generation") is not None)
    assert live.client.rerun_playbook_generation(agent_version="v1") is None
    wait_until(lambda: _progress(live, "playbook_generation") is not None)
    assert live.client.manual_profile_generation(user_id="hank") is None
    assert live.client.manual_playbook_generation(agent_version="v1") is None


def test_upgrades_promote_pending_rows(live):
    live.client.add_user_profile(
        [{"user_id": "ivy", "content": "Pending fact.", "status": "pending"}]
    )
    assert live.storage.get_user_profile("ivy", status_filter=[Status.PENDING])
    profiles = live.client.upgrade_profiles(user_id="ivy")
    assert profiles.success and profiles.profiles_promoted == 1
    assert [p.content for p in live.storage.get_user_profile("ivy")] == [
        "Pending fact."
    ]

    pending = _user_playbook("Pending rule.").model_copy(
        update={"status": Status.PENDING}
    )
    live.client.add_user_playbook([pending])
    playbooks = live.client.upgrade_user_playbooks(agent_version="v1")
    assert playbooks.success and playbooks.user_playbooks_promoted == 1
    assert [r.content for r in live.storage.get_user_playbooks()] == ["Pending rule."]


def test_run_playbook_aggregation_over_both_transports(live):
    aggregation = live.client.run_playbook_aggregation(
        agent_version="v1", wait_for_response=True
    )
    assert aggregation is not None
    assert live.client.run_playbook_aggregation(agent_version="v1") is None


def test_whoami_my_config_and_set_config(live):
    who = live.client.whoami()
    assert who.success
    assert who.org_id == live.org_id
    assert who.storage_type == "sqlite"

    mine = live.client.get_my_config()
    assert isinstance(mine.success, bool)

    current = live.client.get_config()
    # A ConfigResponse carries read-only extras; set_config drops them.
    assert live.client.set_config(current)["success"] is True
    plain = Config.model_validate(current.model_dump(mode="python"), extra="ignore")
    assert live.client.set_config(plain.model_dump(mode="json"))["success"] is True
