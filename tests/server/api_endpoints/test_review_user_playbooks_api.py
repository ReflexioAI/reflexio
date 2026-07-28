"""HTTP contract tests for persisted user-playbook re-review."""

from reflexio.models.api_schema.domain.entities import ReviewUserPlaybooksResponse


def test_review_user_playbooks_report_mode_returns_decisions_inline(
    client, mock_reflexio, patched_reflexio
) -> None:
    mock_reflexio.review_user_playbooks.return_value = ReviewUserPlaybooksResponse(
        success=True,
        report_only=True,
        run_id="review-run",
        selected_count=2,
        accepted_count=1,
        rejected_count=1,
    )

    response = client.post(
        "/api/review_user_playbooks",
        json={
            "start_time": "2026-07-01T00:00:00Z",
            "end_time": "2026-07-02T00:00:00Z",
            "top_k": 25,
            "report_only": True,
        },
        headers={"User-Agent": "test"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "report_only": True,
        "run_id": "review-run",
        "selected_count": 2,
        "accepted_count": 1,
        "edited_count": 0,
        "rejected_count": 1,
        "skipped_count": 0,
        "results": [],
    }
    request = mock_reflexio.review_user_playbooks.call_args.args[0]
    assert request.top_k == 25
    assert request.report_only is True
    assert request.start_time.isoformat() == "2026-07-01T00:00:00+00:00"
    assert request.end_time.isoformat() == "2026-07-02T00:00:00+00:00"


def test_review_user_playbooks_apply_mode_is_accepted_and_run_in_background(
    client, mock_reflexio, patched_reflexio
) -> None:
    """Apply mode must answer before the run finishes, carrying its run_id."""
    response = client.post(
        "/api/review_user_playbooks",
        json={
            "start_time": "2026-07-01T00:00:00Z",
            "end_time": "2026-07-02T00:00:00Z",
            "top_k": 25,
            "report_only": False,
        },
        headers={"User-Agent": "test"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["report_only"] is False
    assert body["results"] == []
    run_id = body["run_id"]
    assert run_id.startswith("playbook_review_")

    # TestClient drains background tasks after the response, so the run itself
    # still executes — with the run_id the response already handed back.
    request, kwargs = (
        mock_reflexio.review_user_playbooks.call_args.args[0],
        mock_reflexio.review_user_playbooks.call_args.kwargs,
    )
    assert request.report_only is False
    assert kwargs["run_id"] == run_id


def test_review_user_playbooks_rejects_invalid_window_and_limit(client) -> None:
    response = client.post(
        "/api/review_user_playbooks",
        json={
            "start_time": "2026-07-02T00:00:00Z",
            "end_time": "2026-07-01T00:00:00Z",
            "top_k": 101,
        },
        headers={"User-Agent": "test"},
    )

    assert response.status_code == 422
