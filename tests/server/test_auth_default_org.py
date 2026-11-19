"""Tests for the no-auth default org resolver.

``default_get_org_id`` controls the request org for local / no-auth
deployments, which in turn names the ``config_<org>.json`` file and scopes
SQLite data. claude-smart sets ``REFLEXIO_DEFAULT_ORG_ID`` so it stops sharing
``config_self-host-org.json`` with the self-host backend; these tests pin that
contract (and its backward-compatible default).
"""

from __future__ import annotations

import pytest
from fastapi import APIRouter, Depends
from fastapi.testclient import TestClient

from reflexio.server import api
from reflexio.server.auth import DEFAULT_ORG_ID, default_get_org_id

_ENV = "REFLEXIO_DEFAULT_ORG_ID"


def test_defaults_to_self_host_org_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert default_get_org_id() == DEFAULT_ORG_ID == "self-host-org"


def test_honors_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV, "claude-smart")
    assert default_get_org_id() == "claude-smart"


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_value_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, blank: str
) -> None:
    # A blank ``KEY=`` line (python-dotenv exports "") must not strand the org
    # at the empty string — it resolves to the default, like unset.
    monkeypatch.setenv(_ENV, blank)
    assert default_get_org_id() == DEFAULT_ORG_ID


def test_local_apps_retain_independent_construction_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = APIRouter()

    @router.get("/test/local-identity")
    def org(org_id: str = Depends(default_get_org_id)) -> str:
        return org_id

    monkeypatch.setenv(_ENV, "standalone-local")
    standalone = api.create_app(mount_data_plane=False, additional_routers=[router])
    monkeypatch.setenv(_ENV, "claude-smart")
    plugin = api.create_app(mount_data_plane=False, additional_routers=[router])
    monkeypatch.setenv(_ENV, "later-process-value")

    assert (
        TestClient(standalone).get("/test/local-identity").json() == "standalone-local"
    )
    assert TestClient(plugin).get("/test/local-identity").json() == "claude-smart"


def test_local_lifespan_uses_the_same_bound_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from reflexio.server.api_endpoints import request_context
    from reflexio.server.llm import model_defaults
    from reflexio.server.services import durable_learning, search_metering_worker
    from reflexio.server.services.lineage import gc_scheduler, vector_backfill_sweep
    from reflexio.server.services.playbook import aggregation_scheduler

    observed: dict[str, str] = {}

    def record_scheduler(name: str):
        def start(_factory, *, bootstrap_org_id: str, **_kwargs) -> None:
            observed[name] = bootstrap_org_id

        return start

    router = APIRouter()

    @router.get("/test/local-identity")
    def org(org_id: str = Depends(default_get_org_id)) -> str:
        return org_id

    monkeypatch.setattr(api, "freeze_startup_heap", lambda: None)
    monkeypatch.setattr(model_defaults, "validate_llm_availability", lambda: None)
    monkeypatch.setattr(
        vector_backfill_sweep, "install_missing_vector_backfill_sweep", lambda: None
    )
    monkeypatch.setattr(gc_scheduler, "maybe_start_lineage_gc", record_scheduler("gc"))
    monkeypatch.setattr(
        durable_learning, "maybe_start_durable_learning", record_scheduler("durable")
    )
    monkeypatch.setattr(request_context, "RequestContext", lambda *, org_id: org_id)
    monkeypatch.setattr(
        aggregation_scheduler,
        "ensure_local_playbook_aggregation_scheduler",
        lambda org_id: observed.update(aggregation=org_id),
    )
    monkeypatch.setattr(
        search_metering_worker, "start_search_metering_worker", lambda: None
    )
    monkeypatch.setattr(
        search_metering_worker, "stop_search_metering_worker", lambda **_kwargs: 0
    )
    monkeypatch.setattr(api, "_lifespan_org_id_resolver", lambda: "unrelated-app")

    monkeypatch.setenv(_ENV, "local-at-launch")
    app = api.create_app(additional_routers=[router])
    monkeypatch.setenv(_ENV, "later-process-value")
    with TestClient(app) as client:
        assert client.get("/test/local-identity").json() == "local-at-launch"
    assert observed == {
        "gc": "local-at-launch",
        "durable": "local-at-launch",
        "aggregation": "local-at-launch",
    }


class TestLifespanOrgIdResolver:
    """The bootstrap org for lifespan schedulers, which has no request to read.

    ``DEFAULT_ORG_ID`` is a sentinel, not a real organization: in every
    enterprise deployment mode a config read keyed on it raises "Organization
    self-host-org not found". Multi-tenant apps therefore register a resolver
    that returns a real org id.
    """

    @pytest.fixture(autouse=True)
    def _clear_resolver(self):
        api.set_lifespan_org_id_resolver(None)
        yield
        api.set_lifespan_org_id_resolver(None)

    def test_falls_back_to_default_without_a_resolver(self) -> None:
        assert api._resolve_lifespan_org_id(None) == DEFAULT_ORG_ID

    def test_registered_resolver_wins(self) -> None:
        api.set_lifespan_org_id_resolver(lambda: "org_42")
        assert api._resolve_lifespan_org_id(None) == "org_42"

    def test_resolver_wins_over_a_parameterized_dependency(self) -> None:
        """The regression this hook exists for.

        A request-scoped ``get_org_id`` is a FastAPI dependency taking
        parameters, so it can never be called outside a request and the
        lifespan silently fell through to the sentinel.
        """

        def enterprise_get_org_id(credentials=None) -> str:
            raise AssertionError("must not be called outside a request")

        api.set_lifespan_org_id_resolver(lambda: "org_7")
        assert api._resolve_lifespan_org_id(enterprise_get_org_id) == "org_7"

    @pytest.mark.parametrize(
        "resolver",
        [
            pytest.param(
                lambda: (_ for _ in ()).throw(RuntimeError("boom")), id="raises"
            ),
            pytest.param(lambda: "", id="empty"),
            # str(None) is the truthy literal "None" -- a resolver violating the
            # str contract must fall through, not hand back "None" as an org id.
            pytest.param(lambda: None, id="none"),
        ],
    )
    def test_broken_resolver_degrades_to_default(self, resolver) -> None:
        api.set_lifespan_org_id_resolver(resolver)
        assert api._resolve_lifespan_org_id(None) == DEFAULT_ORG_ID
