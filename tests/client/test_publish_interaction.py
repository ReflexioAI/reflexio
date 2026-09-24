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
def test_publish_interaction_surfaces_locally_dropped_field_warnings(
    mock_session_class,
):
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
        interactions=[
            {
                "content": "real",
                "Content": "typo",
                "tools_used": [{"tool_name": "t", "zzz": "typo"}],
            }
        ],
        session_id="s",
    )

    sent = mock_session.request.call_args.kwargs["json"]
    assert "Content" not in str(sent)
    # Nested extras are stripped too, or the nested warnings double as well.
    assert "zzz" not in str(sent)
    # Re-parse exactly what the route parses: it must find nothing to warn about.
    assert PublishUserInteractionRequest(**sent).payload_warnings() == []


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_sends_retrieval_experiment_attribution(
    mock_session_class,
):
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value.status_code = 200
    mock_session.request.return_value.json.return_value = {"success": True}
    client = ReflexioClient(api_key="test_key")

    client.publish_interaction(
        user_id="user",
        interactions=[{"content": "real"}],
        session_id="s",
        retrieval_experiment_id="exp-1",
        retrieval_experiment_arm="treatment",
    )

    sent = mock_session.request.call_args.kwargs["json"]
    assert sent["retrieval_experiment_id"] == "exp-1"
    assert sent["retrieval_experiment_arm"] == "treatment"


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_sends_a_caller_supplied_request_id(mock_session_class):
    """The 202/504 retry contract is unusable through the SDK without this.

    Both responses tell the caller to retry under the SAME request_id, and
    `request_id` is a primary key the server re-checks inside its commit, so
    the retry publishes if and only if the first attempt did not. If the
    public method cannot carry the id, retrying means calling it again, the
    server mints a fresh UUID, and a publish that DID commit is duplicated --
    which is precisely what the advice was meant to prevent.
    """
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value.status_code = 200
    mock_session.request.return_value.json.return_value = {"success": True}
    client = ReflexioClient(api_key="test_key")

    client.publish_interaction(
        user_id="user",
        interactions=[{"content": "real"}],
        session_id="s",
        request_id="req-caller-owned",
    )

    sent = mock_session.request.call_args.kwargs["json"]
    assert sent["request_id"] == "req-caller-owned"


@patch("reflexio.client.client.requests.Session")
def test_publish_interaction_omits_request_id_when_not_supplied(mock_session_class):
    """Omitting it must still let the server mint one.

    The parameter is an override, not a new requirement: sending an explicit
    null is the same as sending nothing, and every existing caller passes
    nothing.
    """
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value.status_code = 200
    mock_session.request.return_value.json.return_value = {"success": True}
    client = ReflexioClient(api_key="test_key")

    client.publish_interaction(
        user_id="user",
        interactions=[{"content": "real"}],
        session_id="s",
    )

    sent = mock_session.request.call_args.kwargs["json"]
    assert sent.get("request_id") is None


@pytest.mark.asyncio
async def test_publish_interaction_async_uses_shared_validation_and_warnings(
    monkeypatch,
):
    client = ReflexioClient(api_key="test_key")
    captured = {}

    async def fake_publish(request, wait_for_response=False):
        captured["request"] = request
        captured["wait_for_response"] = wait_for_response
        from reflexio.models.api_schema.service_schemas import (
            PublishUserInteractionResponse,
        )

        return PublishUserInteractionResponse(success=True)

    monkeypatch.setattr(client, "_publish_interaction_async", fake_publish)
    result = await client.publish_interaction_async(
        user_id="user",
        interactions=[{"content": "real", "Content": "typo"}],
        session_id="s",
    )

    assert captured["request"].session_id == "s"
    assert captured["wait_for_response"] is False
    assert any("Content" in warning for warning in result.warnings)
