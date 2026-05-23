"""Pydantic models for the Braintrust connector (Plan C-backend).

The connector imports per-scorer outputs from a customer's Braintrust
workspace and surfaces them next to Reflexio's own evaluation results.
This module owns the schemas; the orchestration lives in
`reflexio.server.services.braintrust.service`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ===============================
# Stored entities
# ===============================


class BraintrustConnection(BaseModel):
    """A persisted Braintrust workspace connection for one Reflexio org.

    The `api_key_enc` field carries the Fernet-encrypted API key. Storage
    layers MUST not log or echo this field.

    Args:
        org_id (str): Reflexio org owning the connection.
        api_key_enc (str): Fernet-encrypted Braintrust API key.
        workspace_id (str): Braintrust workspace (organization) id.
        workspace_name (str): Human-readable workspace name (cached).
        project_ids (list[str]): Selected Braintrust project ids to sync.
        last_sync_ts (int | None): Unix epoch of last successful sync.
        last_error (str | None): Last sync error message, if any.
    """

    org_id: str
    api_key_enc: str
    workspace_id: str
    workspace_name: str = ""
    project_ids: list[str] = Field(default_factory=list)
    last_sync_ts: int | None = None
    last_error: str | None = None


class ImportedScore(BaseModel):
    """One scorer output imported from an external eval tool.

    Source is generic to enable future LangSmith / Patronus imports
    against the same table.

    Args:
        org_id (str): Reflexio org owning the score.
        source (Literal["braintrust"]): Provider — extensible.
        source_run_id (str): Provider-side span/run id (unique per source).
        session_id (str | None): Reflexio session this score attaches to
            when matched via `span.metadata.reflexio_session_id`; None
            until the customer instruments their Braintrust spans.
        scorer_name (str): Provider-side scorer name.
        value (float): Scorer output value; range is provider-specific.
        ts (int): Unix epoch of the span's creation in the provider.
    """

    org_id: str
    source: Literal["braintrust"] = "braintrust"
    source_run_id: str
    session_id: str | None = None
    scorer_name: str
    value: float
    ts: int


# ===============================
# Requests / responses
# ===============================


class BraintrustWorkspaceSummary(BaseModel):
    """One workspace + its projects as returned by Braintrust."""

    workspace_id: str
    workspace_name: str
    projects: list[BraintrustProjectSummary] = Field(default_factory=list)


class BraintrustProjectSummary(BaseModel):
    project_id: str
    project_name: str


class ConnectBraintrustRequest(BaseModel):
    """Step 1 of the connect flow: validate the API key + list workspaces.

    The key is NOT persisted here; only after `select_projects` does it
    land in storage (encrypted).
    """

    api_key: str = Field(min_length=1)


class ConnectBraintrustResponse(BaseModel):
    success: bool
    workspaces: list[BraintrustWorkspaceSummary] = Field(default_factory=list)
    msg: str = ""


class SelectProjectsRequest(BaseModel):
    """Step 2 of the connect flow: commit the connection."""

    api_key: str = Field(min_length=1)
    workspace_id: str = Field(min_length=1)
    workspace_name: str = ""
    project_ids: list[str] = Field(default_factory=list)


class SelectProjectsResponse(BaseModel):
    success: bool
    msg: str = ""


class BraintrustStatusResponse(BaseModel):
    """Returned by GET /api/braintrust/status.

    Connected reflects whether a row exists in storage; sensitive fields
    (the API key) are never serialized.
    """

    connected: bool
    workspace_id: str = ""
    workspace_name: str = ""
    project_count: int = 0
    last_sync_ts: int | None = None
    last_error: str | None = None


class SyncBraintrustResponse(BaseModel):
    success: bool
    scored_count: int = 0
    msg: str = ""


BraintrustWorkspaceSummary.model_rebuild()
