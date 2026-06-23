"""Read-only storage reader for the B3 pre-cutover parity check.

``RestStorageReader`` exposes only the read methods that
``reconstruct_profile_change_log`` + ``run_parity_check`` call (the
``ParityReadStorage`` protocol), sourced from Supabase **PostgREST GETs**. It is
deliberately NOT a ``BaseStorage`` subclass and performs **no writes** — so it
can be pointed at a production data ref to run the parity check without the side
effects of constructing the real ``SupabaseStorage`` (whose ``__init__`` can
mutate PostgREST schema config).

The HTTP layer is injectable (``fetch``) so the reconstruction/classification
pipeline can be unit-tested with canned rows and no network.
"""

from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse

from reflexio.models.api_schema.domain.entities import (
    LineageEvent,
    ProfileChangeLog,
    UserProfile,
)

# Page size for unbounded reads. A read returning a full page may be truncated,
# which would silently skew the parity verdict — so reaching it flips
# ``truncated`` and run_parity_check surfaces the run as INCONCLUSIVE.
_FETCH_CAP = 10_000

_LINEAGE_EVENT_FIELDS = set(LineageEvent.model_fields)
_USER_PROFILE_FIELDS = set(UserProfile.model_fields)
_CHANGE_LOG_FIELDS = set(ProfileChangeLog.model_fields)

# Explicit column list for profiles — avoids pulling the embedding/FTS columns.
_PROFILE_COLUMNS = (
    "profile_id,user_id,content,last_modified_timestamp,"
    "generated_from_request_id,status,superseded_by,merged_into,source"
)

Fetch = Callable[[str, dict], list[dict]]


def _model[M](cls: type[M], fields: set[str], row: dict) -> M:
    """Build a Pydantic model from a row, ignoring columns it does not declare."""
    return cls(**{k: v for k, v in row.items() if k in fields})


class RestStorageReader:
    """Read-only PostgREST reader implementing the parity-check read surface."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        org_id: str,
        schema: str = "public",
        fetch: Fetch | None = None,
    ) -> None:
        # The api_key is a service_role credential (RLS-bypassing). Refuse to
        # attach it to anything but an https Supabase host, so a typo'd or
        # hostile --supabase-url cannot exfiltrate it.
        host = urlparse(base_url).hostname or ""
        if urlparse(base_url).scheme != "https" or not host.endswith(".supabase.co"):
            raise ValueError(
                f"base_url must be an https://<ref>.supabase.co URL; got {base_url!r}"
            )
        self.org_id = org_id
        self._base = base_url.rstrip("/")
        self._key = api_key
        self._schema = schema
        self._fetch: Fetch = fetch or self._http_fetch
        # Set when any unbounded read returns a full _FETCH_CAP page: the
        # reconstruction inputs may be truncated, so the verdict is untrustworthy.
        self.truncated = False

    def _http_fetch(self, table: str, params: dict) -> list[dict]:
        import requests  # noqa: PLC0415 — only needed on the live path

        headers = {"apikey": self._key, "Authorization": f"Bearer {self._key}"}
        # Non-public schemas (the shared org_<id> cohort) are selected per-request
        # via Accept-Profile; dedicated refs keep data in public.
        if self._schema and self._schema != "public":
            headers["Accept-Profile"] = self._schema
        resp = requests.get(
            f"{self._base}/rest/v1/{table}",
            headers=headers,
            params=params,
            timeout=30,
            allow_redirects=False,  # never forward the service_role key on a 3xx
        )
        if resp.is_redirect:
            raise RuntimeError(
                f"PostgREST returned a redirect for {table!r} — refusing to follow it "
                "and forward credentials. Check --supabase-url."
            )
        if resp.status_code in (401, 403):
            raise RuntimeError(
                f"PostgREST {resp.status_code} for {table!r}: the service_role key is "
                "not valid for this data ref (keys are per-ref / per-server)."
            )
        if resp.status_code in (404, 406):
            raise RuntimeError(
                f"PostgREST {resp.status_code} for {table!r}: table/schema not exposed "
                f"— check --schema (got {self._schema!r}; use org_<id> for the shared cohort)."
            )
        resp.raise_for_status()
        return resp.json()

    def _note_page(self, rows: list[dict]) -> list[dict]:
        """Flag a possibly-truncated read (a full page may have more behind it)."""
        if len(rows) >= _FETCH_CAP:
            self.truncated = True
        return rows

    def get_profile_change_logs(self, limit: int = 100) -> list[ProfileChangeLog]:
        rows = self._fetch(
            "profile_change_logs",
            # Mirror the real backend's ordering (created_at DESC).
            {"select": "*", "order": "created_at.desc", "limit": limit},
        )
        out: list[ProfileChangeLog] = []
        for raw in rows:
            row = dict(raw)
            for key in ("added_profiles", "removed_profiles", "mentioned_profiles"):
                row[key] = [
                    _model(UserProfile, _USER_PROFILE_FIELDS, p)
                    for p in (row.get(key) or [])
                ]
            out.append(_model(ProfileChangeLog, _CHANGE_LOG_FIELDS, row))
        return out

    def get_lineage_events(
        self,
        *,
        entity_type: str | None = None,
        entity_id: str | None = None,
        org_id: str | None = None,
        request_id: str | None = None,
    ) -> list[LineageEvent]:
        params: dict = {"select": "*", "order": "event_id.asc", "limit": _FETCH_CAP}
        if entity_type:
            params["entity_type"] = f"eq.{entity_type}"
        if entity_id:
            params["entity_id"] = f"eq.{entity_id}"
        if request_id:
            params["request_id"] = f"eq.{request_id}"
        if org_id:
            params["org_id"] = f"eq.{org_id}"
        rows = self._note_page(self._fetch("lineage_event", params))
        return [_model(LineageEvent, _LINEAGE_EVENT_FIELDS, r) for r in rows]

    def get_distinct_generated_from_request_ids(self) -> list[str]:
        rows = self._note_page(
            self._fetch(
                "profiles",
                {"select": "generated_from_request_id", "limit": _FETCH_CAP},
            )
        )
        return sorted(
            {
                r["generated_from_request_id"]
                for r in rows
                if r.get("generated_from_request_id")
            }
        )

    def get_profiles_by_generated_from_request_id(
        self, request_id: str
    ) -> list[UserProfile]:
        rows = self._note_page(
            self._fetch(
                "profiles",
                {
                    "select": _PROFILE_COLUMNS,
                    "generated_from_request_id": f"eq.{request_id}",
                    "limit": _FETCH_CAP,
                },
            )
        )
        return [_model(UserProfile, _USER_PROFILE_FIELDS, r) for r in rows]

    def get_all_generated_profiles(self) -> list[UserProfile]:
        # Fetch all profiles in one page and filter non-empty gfr client-side
        # (mirrors get_distinct; avoids PostgREST empty-string operator ambiguity).
        rows = self._note_page(
            self._fetch("profiles", {"select": _PROFILE_COLUMNS, "limit": _FETCH_CAP})
        )
        return [
            _model(UserProfile, _USER_PROFILE_FIELDS, r)
            for r in rows
            if r.get("generated_from_request_id")
        ]

    def get_profile_by_id(
        self,
        profile_id: str,
        *,
        include_tombstones: bool = False,  # noqa: ARG002 — id lookup always returns tombstones
    ) -> UserProfile | None:
        rows = self._fetch(
            "profiles",
            {"select": _PROFILE_COLUMNS, "profile_id": f"eq.{profile_id}", "limit": 1},
        )
        return _model(UserProfile, _USER_PROFILE_FIELDS, rows[0]) if rows else None
