"""Server-reported publication failure exits nonzero, including mixed batches."""

import json

import pytest

from reflexio.models.api_schema.service_schemas import PublishUserInteractionResponse

pytestmark = pytest.mark.unit


def _payload(user="alice"):
    return {
        "user_id": user,
        "session_id": "s",
        "agent_version": "v1",
        "interactions": [{"role": "user", "content": "Hello"}],
    }


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("input_mode", ["single", "data", "file", "stdin"])
def test_rejected_publish_exits_nonzero(
    runner, app, mock_client, tmp_path, json_mode, input_mode
):
    mock_client.publish_interaction.return_value = PublishUserInteractionResponse(
        success=False, message="database is locked"
    )
    args = ["--json"] if json_mode else []
    args += ["publish", "--wait"]
    stdin = None
    if input_mode == "single":
        args += [
            "--user-id",
            "alice",
            "--session-id",
            "s",
            "--user-message",
            "Hello",
            "--agent-response",
            "Hi",
            "--agent-version",
            "v1",
        ]
    elif input_mode == "data":
        args += ["--data", json.dumps(_payload())]
    elif input_mode == "file":
        path = tmp_path / "publish.json"
        path.write_text(json.dumps(_payload()))
        args += ["--file", str(path)]
    else:
        args += ["--stdin"]
        stdin = json.dumps(_payload())
    result = runner.invoke(app, args, input=stdin)
    assert result.exit_code == 1, result.output
    assert "database is locked" in result.output
    assert mock_client.publish_interaction.call_count == 1
    if json_mode:
        error = json.loads(result.stdout or result.stderr)
        assert error["ok"] is False
        assert error["error"]["type"] == "api"


@pytest.mark.parametrize("json_mode", [False, True])
def test_batch_stops_at_first_rejection_without_replaying_success(
    runner, app, mock_client, json_mode
):
    mock_client.publish_interaction.side_effect = [
        PublishUserInteractionResponse(success=True),
        PublishUserInteractionResponse(success=False, message="Admission rejected"),
        PublishUserInteractionResponse(success=True),
    ]
    args = (["--json"] if json_mode else []) + [
        "publish",
        "--data",
        json.dumps([_payload(user) for user in ("alice", "bob", "carol")]),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1, result.output
    assert [
        call.kwargs["user_id"]
        for call in mock_client.publish_interaction.call_args_list
    ] == ["alice", "bob"]
    assert "Admission rejected" in result.output
    if json_mode:
        assert json.loads(result.stdout)["ok"] is True
        assert json.loads(result.stderr)["ok"] is False


@pytest.mark.parametrize("json_mode", [False, True])
def test_accepted_deferred_publish_is_success(runner, app, mock_client, json_mode):
    mock_client.publish_interaction.return_value = PublishUserInteractionResponse(
        success=True, message="Accepted; extraction deferred", warnings=["Queued"]
    )
    result = runner.invoke(
        app,
        (["--json"] if json_mode else [])
        + ["publish", "--data", json.dumps(_payload())],
    )
    assert result.exit_code == 0, result.output
    if json_mode:
        assert json.loads(result.stdout)["data"]["success"] is True
    else:
        assert "Published (alice)" in result.output
        assert "Queued" in result.output
