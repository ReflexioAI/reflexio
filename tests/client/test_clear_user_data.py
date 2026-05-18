"""Tests for ReflexioClient.clear_user_data — per-``user_id`` data clearing."""

from unittest.mock import MagicMock, patch

from reflexio.client import ReflexioClient


class TestClearUserData:
    @patch("reflexio.client.client.requests.Session")
    def test_posts_user_id_to_clear_user_data_endpoint(
        self, mock_session_class
    ) -> None:
        """Client must POST {"user_id": ...} to /api/clear_user_data."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "deleted_counts": {
                "interactions": 3,
                "user_playbooks": 1,
                "profiles": 2,
                "requests": 1,
            },
            "message": "Cleared 7 row(s) for user 'userA'",
        }
        mock_session.request.return_value = mock_response

        client = ReflexioClient(api_key="test_key")
        result = client.clear_user_data("userA")

        # Right HTTP method and path.
        call_args = mock_session.request.call_args
        assert call_args.args[0] == "POST"
        assert call_args.args[1].endswith("/api/clear_user_data")
        # Right body shape — single user_id field.
        assert call_args.kwargs["json"] == {"user_id": "userA"}

        # Response is parsed into the typed model with intact counts.
        assert result.success is True
        assert result.deleted_counts == {
            "interactions": 3,
            "user_playbooks": 1,
            "profiles": 2,
            "requests": 1,
        }
        assert result.message and "userA" in result.message

    @patch("reflexio.client.client.requests.Session")
    def test_clear_user_data_invalidates_cache(self, mock_session_class) -> None:
        """clear_user_data must clear the cache so stale per-user reads vanish."""
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_response = MagicMock()
        mock_response.json.side_effect = [
            # get_profiles
            {"success": True, "user_profiles": [], "msg": None},
            # clear_user_data
            {
                "success": True,
                "deleted_counts": {
                    "interactions": 0,
                    "user_playbooks": 0,
                    "profiles": 0,
                    "requests": 0,
                },
                "message": "Cleared 0 row(s) for user 'userA'",
            },
            # get_profiles again (cache miss)
            {"success": True, "user_profiles": [], "msg": None},
        ]
        mock_session.request.return_value = mock_response

        client = ReflexioClient(api_key="test_key")
        client.get_profiles(
            {
                "user_id": "userA",
                "start_time": None,
                "end_time": None,
                "top_k": 30,
            }
        )
        assert mock_session.request.call_count == 1

        client.clear_user_data("userA")
        assert mock_session.request.call_count == 2

        # Cache cleared — next read hits the API again.
        client.get_profiles(
            {
                "user_id": "userA",
                "start_time": None,
                "end_time": None,
                "top_k": 30,
            }
        )
        assert mock_session.request.call_count == 3
