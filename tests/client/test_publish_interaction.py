"""ReflexioClient publish_interaction validation."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ReflexioClient


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
