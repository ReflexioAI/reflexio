"""Orchestrates the Braintrust connector: connect, select-projects, sync, status.

The service is stateless across requests; it loads the connection from
storage on each call. API keys are encrypted in storage and decrypted
only when needed for a Braintrust HTTP call.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

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
from reflexio.server.services.braintrust._encryption import decrypt, encrypt
from reflexio.server.services.braintrust.client import (
    DEFAULT_BASE_URL,
    BraintrustAuthError,
    BraintrustClient,
    BraintrustHTTPError,
)


def _default_client_factory(api_key: str) -> BraintrustClient:
    """Default factory that honors `BRAINTRUST_BASE_URL` env override.

    Set the env var to point at a staging or mock Braintrust instance
    (useful for dev / cron testing without hitting the real API).
    """
    import os

    base_url = os.environ.get("BRAINTRUST_BASE_URL", "").strip() or DEFAULT_BASE_URL
    return BraintrustClient(api_key, base_url=base_url)


logger = logging.getLogger(__name__)


_METADATA_SESSION_KEY = "reflexio_session_id"


@dataclass
class BraintrustConnectorService:
    """Top-level orchestrator for the Braintrust connector.

    Args:
        storage: BaseStorage-like; uses save/get/delete_braintrust_connection
            and save_imported_scores. Default no-op storage methods keep
            this workable until per-backend implementations land.
        org_id (str): The Reflexio org for which the connector operates.
        client_factory (Callable[[str], BraintrustClient]): Factory that
            takes an API key and returns a BraintrustClient. Injectable so
            tests can pass mocks.
    """

    storage: object
    org_id: str
    client_factory: Callable[[str], BraintrustClient] = field(
        default=_default_client_factory
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, request: ConnectBraintrustRequest) -> ConnectBraintrustResponse:
        """Step 1: validate the API key and return the workspace/project tree.

        Persists nothing — the caller chooses which projects to sync and
        then calls `select_projects` with the same key.
        """
        client = self.client_factory(request.api_key)
        try:
            if not client.validate_key():
                return ConnectBraintrustResponse(
                    success=False, msg="Braintrust rejected the API key."
                )
            workspaces = self._fetch_workspace_tree(client)
        except BraintrustAuthError:
            return ConnectBraintrustResponse(
                success=False, msg="Braintrust rejected the API key."
            )
        except BraintrustHTTPError as e:
            return ConnectBraintrustResponse(
                success=False, msg=f"Braintrust HTTP error: {e}"
            )
        finally:
            client.close()

        return ConnectBraintrustResponse(success=True, workspaces=workspaces, msg="")

    def select_projects(self, request: SelectProjectsRequest) -> SelectProjectsResponse:
        """Step 2: persist the connection with the selected projects.

        Stores the API key encrypted. Overwrites any existing connection
        for the org.
        """
        connection = BraintrustConnection(
            org_id=self.org_id,
            api_key_enc=encrypt(request.api_key),
            workspace_id=request.workspace_id,
            workspace_name=request.workspace_name,
            project_ids=request.project_ids,
            last_sync_ts=None,
            last_error=None,
        )
        self._save_connection(connection)
        return SelectProjectsResponse(success=True, msg="Connected.")

    def status(self) -> BraintrustStatusResponse:
        """Return whether the org is connected + sync state. Never echoes the key."""
        connection = self._load_connection()
        if connection is None:
            return BraintrustStatusResponse(connected=False)
        return BraintrustStatusResponse(
            connected=True,
            workspace_id=connection.workspace_id,
            workspace_name=connection.workspace_name,
            project_count=len(connection.project_ids),
            last_sync_ts=connection.last_sync_ts,
            last_error=connection.last_error,
        )

    def disconnect(self) -> None:
        """Delete the persisted connection."""
        self.storage.delete_braintrust_connection(self.org_id)  # type: ignore[attr-defined]

    def sync_once(self, backfill_days: int = 90) -> SyncBraintrustResponse:
        """One-shot sync: walk projects → experiments → spans → write scores.

        Updates `last_sync_ts` and `last_error` on the persisted connection.
        Idempotent at the storage layer (per-backend overrides should
        upsert by (source, source_run_id, scorer_name)).
        """
        connection = self._load_connection()
        if connection is None:
            return SyncBraintrustResponse(
                success=False, msg="Not connected to Braintrust."
            )

        try:
            api_key = decrypt(connection.api_key_enc)
        except Exception as e:  # noqa: BLE001
            return SyncBraintrustResponse(
                success=False, msg=f"Failed to decrypt API key: {e}"
            )

        since_ts = connection.last_sync_ts
        if since_ts is None:
            since_ts = max(0, int(time.time()) - backfill_days * 24 * 60 * 60)

        all_scores: list[ImportedScore] = []
        client = self.client_factory(api_key)
        try:
            for project_id in connection.project_ids:
                experiments = client.list_experiments(project_id, since_ts=since_ts)
                for exp in experiments:
                    spans = client.list_spans(exp["id"])
                    all_scores.extend(_scores_from_spans(spans, self.org_id))
        except BraintrustAuthError:
            self._persist_sync_outcome(connection, error="API key invalid.")
            return SyncBraintrustResponse(
                success=False, msg="API key invalid; halting sync."
            )
        except BraintrustHTTPError as e:
            self._persist_sync_outcome(connection, error=str(e))
            return SyncBraintrustResponse(success=False, msg=str(e))
        finally:
            client.close()

        self.storage.save_imported_scores(all_scores)  # type: ignore[attr-defined]
        self._persist_sync_outcome(connection, error=None)
        return SyncBraintrustResponse(
            success=True, scored_count=len(all_scores), msg=""
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fetch_workspace_tree(
        self, client: BraintrustClient
    ) -> list[BraintrustWorkspaceSummary]:
        """Pull workspaces + their projects in one shot for the connect UX."""
        result: list[BraintrustWorkspaceSummary] = []
        for ws in client.list_organizations():
            workspace_id = str(ws.get("id", ""))
            workspace_name = str(ws.get("name", ""))
            if not workspace_id:
                continue
            projects_raw = client.list_projects(workspace_id)
            projects = [
                BraintrustProjectSummary(
                    project_id=str(p.get("id", "")),
                    project_name=str(p.get("name", "")),
                )
                for p in projects_raw
                if p.get("id")
            ]
            result.append(
                BraintrustWorkspaceSummary(
                    workspace_id=workspace_id,
                    workspace_name=workspace_name,
                    projects=projects,
                )
            )
        return result

    def _load_connection(self) -> BraintrustConnection | None:
        return self.storage.get_braintrust_connection(self.org_id)  # type: ignore[attr-defined]

    def _save_connection(self, connection: BraintrustConnection) -> None:
        self.storage.save_braintrust_connection(connection)  # type: ignore[attr-defined]

    def _persist_sync_outcome(
        self, connection: BraintrustConnection, *, error: str | None
    ) -> None:
        update: dict[str, int | str | None] = {"last_error": error}
        if error is None:
            update["last_sync_ts"] = int(time.time())
        updated = connection.model_copy(update=update)
        self._save_connection(updated)


def _scores_from_spans(
    spans: list[dict],
    org_id: str,
) -> list[ImportedScore]:
    """Extract ImportedScore rows from a list of Braintrust spans.

    A single span may carry multiple scorers — one row per scorer.
    `session_id` is pulled from `span.metadata.reflexio_session_id` when
    present (the progressive-matching upgrade path).
    """
    out: list[ImportedScore] = []
    for span in spans:
        span_id = str(span.get("id", ""))
        if not span_id:
            continue
        ts_raw = span.get("created_at", 0)
        try:
            ts = int(ts_raw)
        except (TypeError, ValueError):
            ts = 0
        metadata = span.get("metadata") or {}
        session_id: str | None = None
        if isinstance(metadata, dict):
            v = metadata.get(_METADATA_SESSION_KEY)
            if isinstance(v, str) and v:
                session_id = v
        scores = span.get("scores") or {}
        if not isinstance(scores, dict):
            continue
        for scorer_name, value in scores.items():
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                continue
            out.append(
                ImportedScore(
                    org_id=org_id,
                    source_run_id=span_id,
                    session_id=session_id,
                    scorer_name=str(scorer_name),
                    value=fvalue,
                    ts=ts,
                )
            )
    return out
