"""HTTP client for the Braintrust REST API.

Pure HTTP wrapper — no storage, no encryption, no orchestration. The
service layer composes this with storage to build the connector.

The exact endpoint paths (`/v1/api_key`, `/v1/organization`, etc.) follow
the spec §5.1; if Braintrust evolves the API, only this file needs to
change — call sites use the typed methods.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api.braintrust.dev"
DEFAULT_TIMEOUT_SECONDS = 30.0


class BraintrustAuthError(RuntimeError):
    """Raised when the Braintrust API rejects the key (401 / 403)."""


class BraintrustHTTPError(RuntimeError):
    """Raised on non-2xx, non-auth-error responses from Braintrust."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"Braintrust HTTP {status_code}: {body[:200]}")
        self.status_code = status_code
        self.body = body


class BraintrustClient:
    """Thin wrapper around the Braintrust REST API.

    All methods are synchronous; they use a per-client `httpx.Client`
    with a fixed timeout so tests can monkey-patch the client.

    Args:
        api_key (str): Customer-supplied Braintrust API key.
        base_url (str): Override for testing.
        timeout (float): Per-request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
                "User-Agent": "reflexio-braintrust-connector/1.0",
            },
        )

    def close(self) -> None:
        self._client.close()

    # ------------------------------------------------------------------
    # API methods
    # ------------------------------------------------------------------

    def validate_key(self) -> bool:
        """Return True iff the API key is valid.

        Returns:
            bool: True on 2xx, False on 401/403. Other errors raise.
        """
        try:
            self._get("/v1/api_key")
        except BraintrustAuthError:
            return False
        return True

    def list_organizations(self) -> list[dict[str, Any]]:
        """List workspaces (Braintrust calls these "organizations").

        Returns:
            list[dict]: Each item has at least `id` and `name`.
        """
        return self._unwrap(self._get("/v1/organization"))

    def list_projects(self, workspace_id: str) -> list[dict[str, Any]]:
        """List projects in a workspace.

        Args:
            workspace_id (str): Braintrust organization id.

        Returns:
            list[dict]: Each item has at least `id` and `name`.
        """
        return self._unwrap(self._get("/v1/project", params={"org_id": workspace_id}))

    def list_experiments(
        self, project_id: str, since_ts: int | None = None
    ) -> list[dict[str, Any]]:
        """List experiments in a project.

        Args:
            project_id (str): Braintrust project id.
            since_ts (int | None): Optional unix-epoch lower bound on
                `experiment.created_at`.

        Returns:
            list[dict]: Each item has at least `id`, `name`, `created_at`.
        """
        params: dict[str, Any] = {"project_id": project_id}
        if since_ts is not None:
            params["since"] = since_ts
        return self._unwrap(self._get("/v1/experiment", params=params))

    def list_spans(self, experiment_id: str) -> list[dict[str, Any]]:
        """List spans for an experiment.

        Returns:
            list[dict]: Each item has at least `id`, `metadata`, `scores`,
                `created_at`. Other fields are ignored by the caller.
        """
        return self._unwrap(
            self._get("/v1/span", params={"experiment_id": experiment_id})
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _get(
        self, path: str, *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            response = self._client.get(url, params=params)
        except httpx.RequestError as e:
            raise BraintrustHTTPError(503, str(e)) from e
        if response.status_code in (401, 403):
            raise BraintrustAuthError(
                f"Braintrust rejected the API key (HTTP {response.status_code})"
            )
        if not (200 <= response.status_code < 300):
            raise BraintrustHTTPError(response.status_code, response.text)
        return response

    @staticmethod
    def _unwrap(response: httpx.Response) -> list[dict[str, Any]]:
        """Pull the list payload from Braintrust's standard envelope.

        Braintrust wraps list responses in `{"objects": [...]}`. We
        accept either the wrapped form or a bare list (some endpoints).
        """
        data = response.json()
        if isinstance(data, dict) and "objects" in data:
            return data["objects"]
        if isinstance(data, list):
            return data
        return []
