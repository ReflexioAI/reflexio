"""ReflexioClient publish_interaction validation."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.service_schemas import (
    PublishUserInteractionRequest,
)


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_requires_session_id(mock_session_class):
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    client = ReflexioClient(api_key="test_key")

    with pytest.raises(ValueError, match="session_id is required"):
        client.publish_interaction(
            user_id="user",
            interactions=[{"role": "user", "content": "hello"}],
        )

    mock_session.request.assert_not_called()


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_rejects_blank_session_id(mock_session_class):
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    client = ReflexioClient(api_key="test_key")

    with pytest.raises(ValueError, match="session_id is required"):
        client.publish_interaction(
            user_id="user",
            interactions=[{"role": "user", "content": "hello"}],
            session_id=" ",
        )

    mock_session.request.assert_not_called()


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_surfaces_locally_dropped_field_warnings(mock_session_class):
    """SDK callers must see unrecognised-field warnings.

    ``publish_interaction`` builds ``InteractionData`` before
    ``request.model_dump()``, which strips unknown keys — so the payload the
    server receives is already clean and it cannot echo what it never saw.
    Without merging the locally-detected warnings, a mis-keyed field is reported
    over raw HTTP but completely invisible through the client, which is the
    primary integration path.
    """
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value.status_code = 200
    mock_session.request.return_value.json.return_value = {"success": True}
    client = ReflexioClient(api_key="test_key")

    result = client.publish_interaction(
        user_id="user",
        interactions=[{"content": "real turn", "Content": "typo"}],
        session_id="s",
    )

    assert any("Content" in warning for warning in result.warnings), result.warnings
    # Names only — the value is caller payload.
    assert not any("typo" in warning for warning in result.warnings), result.warnings


@patch("reflexio.client.client.requests.Session")
def test_wire_payload_carries_no_unknown_keys(mock_session_class):
    """Pin the premise the client-side merge depends on.

    The merge is only correct because unknown keys never reach the server, so
    the server cannot report them too. If that changed — the strip removed,
    ``model_dump()`` made to include extras — every warning would silently
    double, and the merge has no dedup. Assert the invariant directly.
    """
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value.status_code = 200
    mock_session.request.return_value.json.return_value = {"success": True}
    client = ReflexioClient(api_key="test_key")

    client.publish_interaction(
        user_id="user",
        interactions=[{"content": "real", "Content": "typo"}],
        session_id="s",
    )

    sent = mock_session.request.call_args.kwargs["json"]
    assert "Content" not in str(sent)
    # Re-parse exactly what the route parses: it must find nothing to warn about.
    assert PublishUserInteractionRequest(**sent).payload_warnings() == []
