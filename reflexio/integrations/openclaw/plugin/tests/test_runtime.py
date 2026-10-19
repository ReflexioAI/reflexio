"""The integration retains its existing cross-project learning identity."""

from openclaw_smart import runtime


def test_agent_version_is_openclaw():
    assert runtime.agent_version() == "openclaw"
