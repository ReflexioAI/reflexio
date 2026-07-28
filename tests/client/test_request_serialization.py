"""Every request body the client sends must survive ``json.dumps``.

Request models carry ``datetime`` fields (``start_time`` / ``end_time`` on all
the time-filtered reads). A plain ``model_dump()`` leaves those as ``datetime``
objects, and ``requests`` then raises ``TypeError: Object of type datetime is
not JSON serializable`` *before the request ever leaves the process* — so every
time-filtered read was unusable, on both the kwargs and request-object paths.

A mock-based call-assertion cannot catch this: the mock never serializes. These
tests assert the payload itself is JSON-safe, plus a source-level guard so a new
call site cannot reintroduce the bug.
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from reflexio.client import ReflexioClient
from reflexio.client import client as client_module

_CLIENT_SOURCE = Path(client_module.__file__)

START = datetime(2026, 6, 1, tzinfo=UTC)
END = datetime(2026, 7, 1, tzinfo=UTC)


def _json_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.content = b"{}"
    response.headers = {"Content-Type": "application/json"}
    return response


# Every read that accepts a datetime filter: method, minimal valid response, and
# any required arguments beyond the time filters.
_TIME_FILTERED_READS = [
    (
        "get_agent_success_evaluation_results",
        {"success": True, "agent_success_evaluation_results": []},
        {},
    ),
    ("get_retrieved_learning_evaluation_results", {"success": True, "results": []}, {}),
    ("get_requests", {"success": True, "sessions": []}, {}),
    ("get_interactions", {"success": True, "interactions": []}, {"user_id": "u1"}),
    ("get_profiles", {"success": True, "user_profiles": []}, {"user_id": "u1"}),
    ("get_user_playbooks", {"success": True, "user_playbooks": []}, {}),
    ("get_agent_playbooks", {"success": True, "agent_playbooks": []}, {}),
]


@pytest.mark.parametrize(("method_name", "payload", "extra"), _TIME_FILTERED_READS)
@patch("reflexio.client.client.requests.Session")
def test_datetime_filters_serialize_to_json(
    mock_session_class, method_name: str, payload: dict, extra: dict
) -> None:
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(payload)
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    getattr(client, method_name)(start_time=START, end_time=END, **extra)

    _, kwargs = mock_session.request.call_args
    body = kwargs.get("json")
    assert body is not None, f"{method_name} sent no JSON body"

    # The real failure mode: requests calls json.dumps on this and blows up.
    json.dumps(body)

    # Compare instants, not spelling — pydantic renders UTC as "...Z" while
    # datetime.isoformat() renders "+00:00"; both are valid ISO 8601.
    assert isinstance(body["start_time"], str)
    assert datetime.fromisoformat(body["start_time"]) == START
    assert datetime.fromisoformat(body["end_time"]) == END


@patch("reflexio.client.client.requests.Session")
def test_request_object_path_serializes_too(mock_session_class) -> None:
    """The request-object path must not diverge from the kwargs path.

    Some parameters (``offset`` on ``get_requests``) are only reachable by
    passing a request object, so it is a real call path, not a legacy alias.
    """
    from reflexio.models.api_schema.retriever_schema import GetRequestsRequest

    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.request.return_value = _json_response(
        {"success": True, "sessions": []}
    )
    client = ReflexioClient(api_key="test_key", url_endpoint="http://localhost:8000")

    client.get_requests(
        GetRequestsRequest(source="s", start_time=START, end_time=END, offset=40)
    )

    _, kwargs = mock_session.request.call_args
    json.dumps(kwargs["json"])
    assert kwargs["json"]["offset"] == 40


def _json_kwarg_dumps(tree: ast.AST) -> list[tuple[int, ast.Call]]:
    """Every ``model_dump(...)`` passed as a ``json=`` keyword, with its line."""
    found: list[tuple[int, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            value = keyword.value
            if (
                keyword.arg == "json"
                and isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "model_dump"
            ):
                found.append((value.lineno, value))
    return found


def test_no_request_body_uses_python_mode_model_dump() -> None:
    """Source guard: catches any NEW call site reintroducing the bug.

    Scoped to ``model_dump`` results handed to ``json=`` — a bare
    ``model_dump()`` elsewhere (in-process use) is unaffected and stays legal.
    """
    tree = ast.parse(_CLIENT_SOURCE.read_text(encoding="utf-8"))
    dumps = _json_kwarg_dumps(tree)
    assert dumps, "found no json=<model>.model_dump(...) call sites — guard is blind"

    offenders = [
        lineno
        for lineno, call in dumps
        if not any(
            kw.arg == "mode"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value == "json"
            for kw in call.keywords
        )
    ]
    assert not offenders, (
        f"{_CLIENT_SOURCE.name} lines {offenders} pass model_dump() to json= "
        'without mode="json"; datetime fields will not serialize'
    )
