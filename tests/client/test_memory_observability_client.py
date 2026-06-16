"""ReflexioClient methods for memory observability and review."""

from unittest.mock import MagicMock, patch

from reflexio.client import ReflexioClient
from reflexio.models.api_schema.service_schemas import Status


def _mock_json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b'{"success": true}'
    response.headers = {"Content-Type": "application/json"}
    return response


@patch("reflexio.client.client.requests.Session")
def test_get_injection_stats_posts_to_endpoint(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _mock_json_response(
        {
            "success": True,
            "stats": [
                {
                    "entity_type": "user_playbook",
                    "entity_id": "42",
                    "surfaced_count": 3,
                    "distinct_session_count": 2,
                    "total_prompt_tokens": 120,
                }
            ],
            "msg": "OK",
        }
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.get_injection_stats(days_back=14)

    call = mock_session.request.call_args
    assert call.args[0] == "POST"
    assert call.args[1].endswith("/api/get_injection_stats")
    assert call.kwargs["json"] == {"days_back": 14}
    assert result.success is True
    assert result.stats[0].entity_id == "42"


@patch("reflexio.client.client.requests.Session")
def test_get_memory_review_defaults_to_user_scope(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _mock_json_response(
        {
            "success": True,
            "candidates": [
                {
                    "entity_type": "user_playbook",
                    "entity_id": "7",
                    "title": "old rule",
                    "signals": ["stale"],
                    "score": 70,
                    "injection_count": 0,
                    "citation_count": 0,
                }
            ],
            "msg": "OK",
        }
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    result = client.get_memory_review(user_id="userA", days_back=60)

    call = mock_session.request.call_args
    assert call.args[0] == "POST"
    assert call.args[1].endswith("/api/get_memory_review")
    assert call.kwargs["json"] == {
        "days_back": 60,
        "signal_filter": None,
        "user_id": "userA",
        "include_all_users": False,
    }
    assert result.candidates[0].entity_id == "7"


@patch("reflexio.client.client.requests.Session")
def test_get_memory_review_allows_explicit_org_wide_review(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _mock_json_response(
        {"success": True, "candidates": [], "msg": "OK"}
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    client.get_memory_review(days_back=30, include_all_users=True)

    assert mock_session.request.call_args.kwargs["json"] == {
        "days_back": 30,
        "signal_filter": None,
        "user_id": None,
        "include_all_users": True,
    }


@patch("reflexio.client.client.requests.Session")
def test_update_user_playbook_sends_review_action_fields(mock_session_class) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _mock_json_response(
        {"success": True, "msg": "updated"}
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    client.update_user_playbook(
        42,
        status=Status.ARCHIVED,
        playbook_metadata='{"superseded_by": 7}',
    )

    assert mock_session.request.call_args.kwargs["json"] == {
        "user_playbook_id": 42,
        "playbook_name": None,
        "content": None,
        "trigger": None,
        "rationale": None,
        "status": "archived",
        "playbook_metadata": '{"superseded_by": 7}',
    }
