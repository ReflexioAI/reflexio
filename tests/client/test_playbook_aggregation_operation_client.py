"""The durable aggregation client preserves admission and polling identities."""

import json
from unittest.mock import MagicMock

from reflexio import ReflexioClient


def test_admission_json_and_typed_poll_preserve_operation_identity(monkeypatch):
    operation_id = "operation/segment?tenant=other"
    queued = {
        "operation_id": operation_id,
        "request_id": "stable-request",
        "agent_version": "v1",
        "status": "queued",
        "created_at": 1,
        "updated_at": 1,
    }
    succeeded = {
        **queued,
        "status": "succeeded",
        "updated_at": 2,
        "completed_at": 2,
        "result": {
            "clusters_found": 1,
            "user_playbooks_processed": 2,
            "playbooks_generated": 1,
        },
    }
    responses = []
    for status, payload in ((202, queued), (200, succeeded)):
        response = MagicMock()
        response.status_code = status
        response.content = b"{}"
        response.headers = {"Content-Type": "application/json"}
        response.json.return_value = payload
        responses.append(response)
    session = MagicMock()
    session.request.side_effect = responses
    monkeypatch.setattr("reflexio.client.client.requests.Session", lambda: session)
    client = ReflexioClient(api_key="test-key", url_endpoint="http://localhost:8000")

    admitted = client.submit_playbook_aggregation(
        request_id="stable-request", agent_version="v1"
    )
    terminal = client.get_playbook_aggregation_operation(admitted.operation_id)

    post, poll = session.request.call_args_list
    assert post.args == (
        "POST",
        "http://localhost:8000/api/playbook_aggregation_operations",
    )
    assert json.loads(json.dumps(post.kwargs["json"])) == {
        "request_id": "stable-request",
        "agent_version": "v1",
    }
    assert poll.args == (
        "GET",
        "http://localhost:8000/api/playbook_aggregation_operations/operation%2Fsegment%3Ftenant%3Dother",
    )
    assert admitted.status == "queued" and terminal.status == "succeeded"
    assert terminal.operation_id == admitted.operation_id == operation_id
    assert terminal.request_id == "stable-request"
    assert terminal.result is not None and terminal.result.playbooks_generated == 1
