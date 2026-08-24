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
