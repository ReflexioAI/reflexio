"""CLI aggregation reports admission and follows only its own durable receipt."""

import json
import time
from types import SimpleNamespace

import pytest
import requests

from reflexio.cli.commands import agent_playbooks as command
from reflexio.models.api_schema.aggregation_operations import (
    PlaybookAggregationOperation,
    PlaybookAggregationResult,
)


def receipt(status="queued", **changes):
    values = {
        "operation_id": "operation",
        "request_id": "request",
        "agent_version": "v1",
        "status": status,
        "created_at": 1,
        "updated_at": 1,
    }
    if status == "succeeded":
        values["result"] = PlaybookAggregationResult(
            user_playbooks_processed=2, clusters_found=1, playbooks_generated=1
        )
    values.update(changes)
    return PlaybookAggregationOperation.model_validate(values)


@pytest.fixture(autouse=True)
def deterministic_request(monkeypatch):
    monkeypatch.setattr(command, "uuid4", lambda: "request")
    monkeypatch.setattr(
        command, "time", SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: None)
    )


def test_wait_tracks_queued_running_retrying_to_committed_success(
    runner, app, mock_client
):
    mock_client.submit_playbook_aggregation.return_value = receipt()
    mock_client.get_playbook_aggregation_operation.side_effect = [
        receipt("running"),
        receipt("retrying"),
        receipt("succeeded"),
    ]
    result = runner.invoke(
        app,
        ["--json", "agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"],
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["data"] == {
        "success": True,
        "message": "Processed 2 user playbooks, found 1 clusters, generated 1 agent playbooks",
    }
    assert body["meta"]["status"] == "succeeded"
    assert body["meta"]["operation_id"] == "operation"
    mock_client.submit_playbook_aggregation.assert_called_once_with(
        request_id="request", agent_version="v1"
    )
    assert [
        call.args
        for call in mock_client.get_playbook_aggregation_operation.call_args_list
    ] == [("operation",)] * 3
    mock_client.run_playbook_aggregation.assert_not_called()


@pytest.mark.parametrize("status", ["queued", "running", "retrying"])
def test_without_wait_reports_admission_not_completion(
    runner, app, mock_client, status
):
    mock_client.submit_playbook_aggregation.return_value = receipt(status)
    result = runner.invoke(
        app, ["--json", "agent-playbooks", "aggregate", "--agent-version", "v1"]
    )
    assert result.exit_code == 0, result.output
    body = json.loads(result.stdout)
    assert body["data"]["message"] == f"Aggregation admitted ({status})"
    assert body["meta"]["operation_id"] == "operation"
    assert body["meta"]["request_id"] == "request"
    mock_client.get_playbook_aggregation_operation.assert_not_called()


@pytest.mark.parametrize("after_poll", [False, True])
def test_failed_operation_exits_nonzero(runner, app, mock_client, after_poll):
    failed = receipt("failed", error="input_erased")
    mock_client.submit_playbook_aggregation.return_value = (
        receipt() if after_poll else failed
    )
    mock_client.get_playbook_aggregation_operation.return_value = failed
    result = runner.invoke(
        app,
        ["--json", "agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"],
    )
    assert result.exit_code != 0
    assert "input_erased" in result.output
    assert not result.stdout
    mock_client.submit_playbook_aggregation.assert_called_once()


@pytest.mark.parametrize(
    "changes",
    [
        {"operation_id": ""},
        {"request_id": "other"},
        {"agent_version": "other"},
    ],
)
def test_mismatched_admission_never_succeeds(runner, app, mock_client, changes):
    mock_client.submit_playbook_aggregation.return_value = receipt(**changes)
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code != 0
    assert "mismatched operation identity" in result.output
    mock_client.get_playbook_aggregation_operation.assert_not_called()


@pytest.mark.parametrize(
    "changes",
    [
        {"operation_id": "other"},
        {"request_id": "other"},
        {"agent_version": "other"},
    ],
)
def test_unrelated_success_receipt_is_rejected(runner, app, mock_client, changes):
    mock_client.submit_playbook_aggregation.return_value = receipt()
    mock_client.get_playbook_aggregation_operation.return_value = receipt(
        "succeeded", **changes
    )
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code != 0
    assert "mismatched operation identity" in result.output
    mock_client.submit_playbook_aggregation.assert_called_once()


def test_success_without_result_is_not_reported_as_zero_output(
    runner, app, mock_client
):
    mock_client.submit_playbook_aggregation.return_value = receipt(
        "succeeded", result=None
    )
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code != 0
    assert "without a result" in result.output


def test_timeout_preserves_operation_identity_without_resubmitting(
    runner, app, mock_client, monkeypatch
):
    clock = iter([0.0, 0.0, 601.0])
    monkeypatch.setattr(command.time, "monotonic", lambda: next(clock))
    mock_client.submit_playbook_aggregation.return_value = receipt()
    mock_client.get_playbook_aggregation_operation.return_value = receipt("running")
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code != 0
    assert "Timed out waiting for aggregation operation" in result.output
    assert "may still be running" in result.output
    mock_client.submit_playbook_aggregation.assert_called_once()
    mock_client.get_playbook_aggregation_operation.assert_called_once_with("operation")


@pytest.mark.parametrize("status", [404, 503])
def test_unsupported_api_is_actionable_and_never_uses_legacy_post(
    runner, app, mock_client, status
):
    response = requests.Response()
    response.status_code = status
    mock_client.submit_playbook_aggregation.side_effect = requests.HTTPError(
        response=response
    )
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code != 0
    assert "Check server health and durable aggregation support" in result.output
    assert "Request ID: request" in result.output
    mock_client.run_playbook_aggregation.assert_not_called()


def test_discovery_submits_each_distinct_version_once_and_preserves_name_filter(
    runner, app, mock_client, monkeypatch
):
    request_ids = iter(["request-v1", "request-v2"])
    monkeypatch.setattr(command, "uuid4", lambda: next(request_ids))
    mock_client.get_user_playbooks.return_value = SimpleNamespace(
        user_playbooks=[
            SimpleNamespace(agent_version=v) for v in ("v2", "v1", "v1", "")
        ]
    )
    mock_client.submit_playbook_aggregation.side_effect = lambda **kw: receipt(
        agent_version=kw["agent_version"],
        request_id=kw["request_id"],
        operation_id="operation-" + kw["agent_version"],
    )
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--playbook-name", "legacy-name"]
    )
    assert result.exit_code == 0, result.output
    mock_client.get_user_playbooks.assert_called_once_with(
        playbook_name="legacy-name", limit=10000
    )
    assert [
        call.kwargs["agent_version"]
        for call in mock_client.submit_playbook_aggregation.call_args_list
    ] == ["v1", "v2"]
    assert [
        call.kwargs["request_id"]
        for call in mock_client.submit_playbook_aggregation.call_args_list
    ] == ["request-v1", "request-v2"]


def test_completed_noop_is_a_real_success(runner, app, mock_client):
    mock_client.submit_playbook_aggregation.return_value = receipt(
        "succeeded", result=PlaybookAggregationResult(skipped="no_user_playbooks")
    )
    result = runner.invoke(
        app, ["agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"]
    )
    assert result.exit_code == 0, result.output
    assert "Skipped: no_user_playbooks" in result.output
    mock_client.get_playbook_aggregation_operation.assert_not_called()


@pytest.mark.parametrize("failure", ["rate_limit", "timeout"])
def test_poll_error_keeps_admitted_operation_id_without_resubmission(
    runner, app, mock_client, failure
):
    mock_client.submit_playbook_aggregation.return_value = receipt()
    if failure == "rate_limit":
        response = requests.Response()
        response.status_code = 429
        error = requests.HTTPError(response=response)
    else:
        error = requests.Timeout("sensitive transport detail")
    mock_client.get_playbook_aggregation_operation.side_effect = error
    result = runner.invoke(
        app,
        ["--json", "agent-playbooks", "aggregate", "--agent-version", "v1", "--wait"],
    )
    assert result.exit_code != 0
    assert "Status check failed for aggregation operation" in result.output
    assert "may still be running" in result.output
    assert "sensitive transport detail" not in result.output
    assert not result.stdout
    mock_client.submit_playbook_aggregation.assert_called_once()
    mock_client.get_playbook_aggregation_operation.assert_called_once_with("operation")
