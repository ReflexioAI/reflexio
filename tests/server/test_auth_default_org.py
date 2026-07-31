"""Tests for the no-auth default org resolver.

``default_get_org_id`` controls the request org for local / no-auth
deployments, which in turn names the ``config_<org>.json`` file and scopes
SQLite data. claude-smart sets ``REFLEXIO_DEFAULT_ORG_ID`` so it stops sharing
``config_self-host-org.json`` with the self-host backend; these tests pin that
contract (and its backward-compatible default).
"""

from __future__ import annotations

import pytest

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
