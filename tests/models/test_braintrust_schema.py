"""Round-trip tests for the Braintrust connector Pydantic models."""

from reflexio.models.api_schema.braintrust_schema import (
    BraintrustConnection,
    BraintrustProjectSummary,
    BraintrustStatusResponse,
    BraintrustWorkspaceSummary,
    ConnectBraintrustRequest,
    ConnectBraintrustResponse,
    ImportedScore,
    SelectProjectsRequest,
    SelectProjectsResponse,
    SyncBraintrustResponse,
)


def test_braintrust_connection_defaults() -> None:
    """A minimal BraintrustConnection round-trips and defaults sensibly."""
    c = BraintrustConnection(
        org_id="org_test",
        api_key_enc="enc_value",
        workspace_id="ws_42",
    )
    assert c.workspace_name == ""
    assert c.project_ids == []
    assert c.last_sync_ts is None
    assert c.last_error is None


def test_imported_score_defaults_source_to_braintrust() -> None:
    s = ImportedScore(
        org_id="org_test",
        source_run_id="span_1",
        scorer_name="hallucination",
        value=0.25,
        ts=1700000000,
    )
    assert s.source == "braintrust"
    assert s.session_id is None


def test_connect_request_rejects_empty_key() -> None:
    """Empty API key fails validation (min_length=1)."""
    try:
        ConnectBraintrustRequest(api_key="")
    except ValueError:
        return
    raise AssertionError("Expected ValidationError for empty api_key")


def test_workspace_summary_roundtrips() -> None:
    w = BraintrustWorkspaceSummary(
        workspace_id="ws_1",
        workspace_name="My Workspace",
        projects=[
            BraintrustProjectSummary(project_id="p_1", project_name="Production"),
            BraintrustProjectSummary(project_id="p_2", project_name="Staging"),
        ],
    )
    dumped = w.model_dump()
    restored = BraintrustWorkspaceSummary(**dumped)
    assert restored.workspace_name == "My Workspace"
    assert len(restored.projects) == 2
    assert restored.projects[0].project_id == "p_1"


def test_status_response_defaults_to_disconnected() -> None:
    s = BraintrustStatusResponse(connected=False)
    assert s.workspace_id == ""
    assert s.project_count == 0
    assert s.last_sync_ts is None


def test_select_projects_response_round_trip() -> None:
    r = SelectProjectsResponse(success=True, msg="ok")
    assert r.success is True
    assert r.msg == "ok"
    # Also exercise SelectProjectsRequest validation
    req = SelectProjectsRequest(
        api_key="sk_test",
        workspace_id="ws_1",
        workspace_name="WS",
        project_ids=["p_1", "p_2"],
    )
    assert req.project_ids == ["p_1", "p_2"]


def test_connect_response_default_lists() -> None:
    r = ConnectBraintrustResponse(success=False, msg="invalid key")
    assert r.workspaces == []


def test_sync_response_defaults() -> None:
    r = SyncBraintrustResponse(success=True, scored_count=42)
    assert r.msg == ""
    assert r.scored_count == 42
