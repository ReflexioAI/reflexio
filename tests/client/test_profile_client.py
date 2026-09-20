"""Profile client behavior tests."""

from unittest.mock import patch
from uuid import UUID

from reflexio.client import ReflexioClient


def test_add_user_profile_defaults_to_uuidv4_and_preserves_supplied_id() -> None:
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    with patch.object(
        client,
        "_make_request",
        return_value={"success": True, "added_count": 2},
    ) as make_request:
        client.add_user_profile(
            [
                {"user_id": "user-1", "content": "Prefers dark mode"},
                {
                    "profile_id": "legacy-custom-id",
                    "user_id": "user-1",
                    "content": "Uses Python",
                },
            ]
        )

    payload = make_request.call_args.kwargs["json"]
    generated_id = payload["user_profiles"][0]["profile_id"]
    assert UUID(generated_id).version == 4
    assert str(UUID(generated_id)) == generated_id
    assert payload["user_profiles"][1]["profile_id"] == "legacy-custom-id"


def test_get_all_profiles_creation_date_filter_serialization():
    from urllib.parse import parse_qs, urlsplit

    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")
    with patch.object(
        client, "_make_request", return_value={"success": True, "user_profiles": []}
    ) as call:
        client.get_all_profiles(start_time=10, end_time=20, date_field="created_at")
        params = parse_qs(urlsplit(call.call_args.args[1]).query)
        assert params["date_field"] == ["created_at"]
        assert params["start_time"] == ["10"]
        assert params["end_time"] == ["20"]
        client.get_all_profiles(start_time=10)
        assert "date_field" not in parse_qs(urlsplit(call.call_args.args[1]).query)
