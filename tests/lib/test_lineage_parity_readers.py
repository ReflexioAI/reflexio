"""Unit tests: read-only RestStorageReader for the B3 pre-cutover parity check.

The reader exposes only the methods reconstruct_profile_change_log +
run_parity_check need, sourced from Supabase PostgREST GETs. These tests inject a
fake fetcher (no network) and drive the REAL reconstruct + classify pipeline, so
they verify row->model mapping AND parity classification end to end.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from reflexio.lib._lineage_parity import ParityClass, run_parity_check
from reflexio.lib._lineage_parity_readers import RestStorageReader
from reflexio.server.services.storage.storage_base import BaseStorage


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
    legacy = {
        "id": 1,
        "user_id": "u1",
        "request_id": "R",
        "created_at": _now(),
        "added_profiles": [p_new],
        "removed_profiles": [p_old],
        "mentioned_profiles": [],
    }
    return {
        "profiles": [p_old, p_new],
        "lineage_event": [event],
        "profile_change_logs": [legacy],
    }


def _make_reader(dataset: dict[str, list[dict]]) -> RestStorageReader:
    def fake_fetch(table: str, params: dict) -> list[dict]:
        if table == "lineage_event":
            return list(dataset["lineage_event"])
        if table == "profile_change_logs":
            return list(dataset["profile_change_logs"])
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

    return RestStorageReader(
        "https://example.supabase.co", "svc-key", org_id="o1", fetch=fake_fetch
    )


def test_rest_reader_parity_match_for_dedup_run():
    reader = _make_reader(_fake_dedup_dataset())
    results = run_parity_check(cast(BaseStorage, reader))

    by_req = {r.request_id: r.classification for r in results}
    assert by_req.get("R") == ParityClass.MATCH, by_req
    assert not [r for r in results if r.classification == ParityClass.RECON_MISSING]


def test_rest_reader_recon_missing_when_no_lineage():
    """Legacy row with no reconstructible signal -> RECON_MISSING or LEGACY_MISSING.

    Drop the lineage event and the survivor's gfr so reconstruction produces
    nothing for R while legacy still has the row.
    """
    dataset = _fake_dedup_dataset()
    dataset["lineage_event"] = []
    for r in dataset["profiles"]:
        r["generated_from_request_id"] = ""
    reader = _make_reader(dataset)

    results = run_parity_check(cast(BaseStorage, reader))
    by_req = {r.request_id: r.classification for r in results}
    # No reconstructible signal for R -> the legacy-only row is classified
    # LEGACY_MISSING (tolerated), never silently MATCH.
    assert by_req.get("R") in {ParityClass.LEGACY_MISSING, ParityClass.RECON_MISSING}
