"""Agent attribution belongs to a caller, never its host environment."""

import pytest

from reflexio import defaults
from reflexio.cli import state
from reflexio.cli.errors import CliError


@pytest.fixture(autouse=True)
def no_ambient_agent_version(monkeypatch):
    monkeypatch.delenv("REFLEXIO_AGENT_VERSION", raising=False)


@pytest.mark.parametrize("value", ["", "private-ambient-agent"])
@pytest.mark.parametrize("explicit", [None, "client-version"])
@pytest.mark.parametrize(
    "resolve",
    [
        defaults.resolve_agent_version,
        state.resolve_agent_version,
        state.require_agent_version,
    ],
)
def test_obsolete_host_identity_cannot_relabel_callers(
    monkeypatch, value, explicit, resolve
):
    monkeypatch.setenv("REFLEXIO_AGENT_VERSION", value)
    with pytest.raises(ValueError, match="REFLEXIO_AGENT_VERSION") as error:
        resolve(explicit)
    assert "private-ambient-agent" not in str(error.value)
    assert "--agent-version" in str(error.value)


def test_two_callers_keep_their_own_identity_and_omission_uses_existing_sentinel():
    assert defaults.resolve_agent_version("agent-one") == "agent-one"
    assert defaults.resolve_agent_version("agent-two") == "agent-two"
    assert state.resolve_agent_version("agent-one") == "agent-one"
    assert (
        defaults.resolve_agent_version() == state.resolve_agent_version() == "agent-v0"
    )


def test_filtering_still_requires_an_explicit_identity():
    with pytest.raises(CliError, match="--agent-version is required"):
        state.require_agent_version(command_hint="regeneration")
    assert state.require_agent_version("selected-version") == "selected-version"


def test_cli_and_server_share_the_existing_default_owner(monkeypatch):
    monkeypatch.setattr(defaults, "DEFAULT_AGENT_VERSION", "future-default")
    assert defaults.resolve_agent_version() == "future-default"
    assert state.resolve_agent_version() == "future-default"
