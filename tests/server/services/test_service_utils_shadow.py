"""Unit tests for the use_shadow parameter in format_interactions_to_history_string."""

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.server.services.service_utils import format_interactions_to_history_string

_UID = "u1"
_RID = "r1"


def _interaction(**kwargs) -> Interaction:
    """Build an Interaction with required fields pre-filled."""
    defaults = {"user_id": _UID, "request_id": _RID, "user_action": UserActionType.NONE}
    return Interaction(**{**defaults, **kwargs})


def test_use_shadow_renders_shadow_content_for_assistant():
    """When use_shadow=True, assistant turns with shadow_content render shadow text."""
    interactions = [
        _interaction(role="user", content="hi"),
        _interaction(
            role="assistant",
            content="hello (regular)",
            shadow_content="hello (shadow)",
        ),
    ]
    out_regular = format_interactions_to_history_string(interactions, use_shadow=False)
    assert "hello (regular)" in out_regular
    assert "hello (shadow)" not in out_regular

    out_shadow = format_interactions_to_history_string(interactions, use_shadow=True)
    assert "hello (shadow)" in out_shadow
    assert "hello (regular)" not in out_shadow
    # User turn always renders original content regardless of use_shadow
    assert "hi" in out_shadow


def test_use_shadow_falls_back_to_content_when_shadow_empty():
    """When use_shadow=True but shadow_content is empty, fall back to content."""
    interactions = [
        _interaction(role="assistant", content="only regular", shadow_content=""),
    ]
    out_shadow = format_interactions_to_history_string(interactions, use_shadow=True)
    assert "only regular" in out_shadow


def test_default_use_shadow_is_false():
    """Default call (no use_shadow arg) renders regular content."""
    interactions = [
        _interaction(role="assistant", content="regular", shadow_content="shadow"),
    ]
    out = format_interactions_to_history_string(interactions)
    assert "regular" in out
    assert "shadow" not in out


def test_use_shadow_multiple_turns():
    """All assistant turns with shadow_content are substituted when use_shadow=True."""
    interactions = [
        _interaction(role="user", content="q1"),
        _interaction(
            role="assistant", content="a1_regular", shadow_content="a1_shadow"
        ),
        _interaction(role="user", content="q2"),
        _interaction(
            role="assistant", content="a2_regular", shadow_content="a2_shadow"
        ),
    ]
    out = format_interactions_to_history_string(interactions, use_shadow=True)
    assert "a1_shadow" in out
    assert "a2_shadow" in out
    assert "a1_regular" not in out
    assert "a2_regular" not in out
    assert "q1" in out
    assert "q2" in out
