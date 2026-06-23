"""Unit tests: read-only RestStorageReader for change-log reconstruction.

Two layers are covered:
- the reconstruction pipeline given canned rows (injected fetcher, no network);
- the reader's own job — translating method args into PostgREST query params /
  headers, truncation detection, and URL validation — which is the error-prone
  part a silently-wrong read would hide in.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from reflexio.lib._lineage_parity_readers import _FETCH_CAP, RestStorageReader
from reflexio.lib._profiles import reconstruct_profile_change_log

_URL = "https://example.supabase.co"


def _now() -> int:
    return int(datetime.now(UTC).timestamp())


def _profile_row(profile_id: str, content: str, gfr: str, status: str | None) -> dict:
    return {
        "profile_id": profile_id,
        "user_id": "u1",
        "content": content,
        "last_modified_timestamp": _now(),
        "generated_from_request_id": gfr,
        "status": status,
        "superseded_by": None,
        "merged_into": None,
        "source": None,
    }


def _fake_dedup_dataset() -> dict[str, list[dict]]:
    """One clean dedup run R: p-new added (gfr=R), p-old superseded under R.

    p-old carries gfr="" so it is not itself a separate add-only run.
    """
    p_old = _profile_row("p-old", "old facts", gfr="", status="superseded")
    p_new = _profile_row("p-new", "new facts", gfr="R", status=None)
    event = {
        "event_id": 1,
        "org_id": "o1",
        "entity_type": "profile",
        "entity_id": "p-old",
        "op": "status_change",
        "prov_relation": "wasInvalidatedBy",
        "source_ids": [],
        "actor": "dedup",
        "request_id": "R",
        "reason": "None->superseded",
        "created_at": _now(),
        "from_status": None,
        "to_status": "superseded",
        "status_namespace": "lifecycle_status",
    }
    return {
        "profiles": [p_old, p_new],
        "lineage_event": [event],
    }


def _make_reader(dataset: dict[str, list[dict]]) -> RestStorageReader:
    def fake_fetch(table: str, params: dict) -> list[dict]:
        if table == "lineage_event":
            return list(dataset["lineage_event"])
        if table == "profiles":
            rows = dataset["profiles"]
            if params.get("select") == "generated_from_request_id":
                return [
                    {"generated_from_request_id": r["generated_from_request_id"]}
                    for r in rows
                ]
            for key in ("generated_from_request_id", "profile_id"):
                if key in params:
                    want = params[key].removeprefix("eq.")
                    return [r for r in rows if r[key] == want]
            return list(rows)
        raise AssertionError(f"unexpected table {table!r}")

    return RestStorageReader(_URL, "svc-key", org_id="o1", fetch=fake_fetch)


def test_rest_reader_reconstructs_dedup_run():
    """The reader drives reconstruction: one run R with p-new added, p-old removed."""
    reader = _make_reader(_fake_dedup_dataset())
    response = reconstruct_profile_change_log(reader)

    assert response.success is True
    by_req = {row.request_id: row for row in response.profile_change_logs}
    assert "R" in by_req
    row = by_req["R"]
    assert {p.profile_id for p in row.added_profiles} == {"p-new"}
    assert {p.profile_id for p in row.removed_profiles} == {"p-old"}


def test_rest_reader_no_signal_yields_no_rows():
    """No reconstructible signal (no events, empty gfr) -> no reconstructed rows."""
    dataset = _fake_dedup_dataset()
    dataset["lineage_event"] = []
    for r in dataset["profiles"]:
        r["generated_from_request_id"] = ""
    reader = _make_reader(dataset)

    response = reconstruct_profile_change_log(reader)
    assert response.profile_change_logs == []


def test_reader_builds_expected_postgrest_params():
    """The reader translates method args into the right PostgREST query params.

    This is the load-bearing, error-prone part: a wrong filter/column would make
    PostgREST return the wrong rows and reconstruction would silently lie.
    """
    calls: list[tuple[str, dict]] = []

    def recording_fetch(table: str, params: dict) -> list[dict]:
        calls.append((table, dict(params)))
        return []

    reader = RestStorageReader(
        _URL, "k", org_id="o1", schema="org_5", fetch=recording_fetch
    )

    reader.get_lineage_events(entity_type="profile", org_id="o1")
    assert calls[-1] == (
        "lineage_event",
        {
            "select": "*",
            "order": "event_id.asc",
            "limit": _FETCH_CAP,
            "entity_type": "eq.profile",
            "org_id": "eq.o1",
        },
    )

    reader.get_profiles_by_generated_from_request_id("R")
    table, params = calls[-1]
    assert table == "profiles"
    assert params["generated_from_request_id"] == "eq.R"

    reader.get_profile_by_id("p-1", include_tombstones=True)
    table, params = calls[-1]
    assert table == "profiles"
    assert params["profile_id"] == "eq.p-1"
    assert params["limit"] == 1


def test_reader_truncation_marks_run_truncated(monkeypatch):
    """A full-page intermediate read flips ``truncated``.

    Cap is patched low so the 2-profile dataset's distinct-gfr read fills a page.
    """
    import reflexio.lib._lineage_parity_readers as readers

    monkeypatch.setattr(readers, "_FETCH_CAP", 2)
    reader = _make_reader(_fake_dedup_dataset())  # 2 profiles -> distinct read hits cap

    reader.get_distinct_generated_from_request_ids()
    assert reader.truncated is True


def test_reader_rejects_non_supabase_url():
    """The service_role key is only ever attached to an https Supabase host."""
    for bad in (
        "http://example.supabase.co",
        "https://evil.com",
        "https://supabase.co.evil.com",
    ):
        with pytest.raises(ValueError, match="supabase.co"):
            RestStorageReader(bad, "k", org_id="o1")


def test_http_fetch_sets_schema_header_and_disables_redirects(monkeypatch):
    """_http_fetch attaches Accept-Profile for a cohort schema and never follows 3xx."""
    captured: dict = {}

    class _Resp:
        status_code = 200
        is_redirect = False

        def raise_for_status(self):
            return None

        def json(self):
            return []

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return _Resp()

    import requests

    monkeypatch.setattr(requests, "get", fake_get)

    reader = RestStorageReader(_URL, "svc-key", org_id="7", schema="org_7")
    reader.get_distinct_generated_from_request_ids()

    assert captured["url"] == f"{_URL}/rest/v1/profiles"
    assert captured["allow_redirects"] is False
    assert captured["headers"]["Accept-Profile"] == "org_7"
    assert captured["headers"]["apikey"] == "svc-key"
