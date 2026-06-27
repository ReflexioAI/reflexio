"""Tests for the no-auth default org resolver.

``default_get_org_id`` controls the request org for local / no-auth
deployments, which in turn names the ``config_<org>.json`` file and scopes
SQLite data. claude-smart sets ``REFLEXIO_DEFAULT_ORG_ID`` so it stops sharing
``config_self-host-org.json`` with the self-host backend; these tests pin that
contract (and its backward-compatible default).
"""

from __future__ import annotations

import pytest

from reflexio.server._auth import DEFAULT_ORG_ID, default_get_org_id

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
