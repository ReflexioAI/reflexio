"""HTTP publishes persist before returning; only embedded publishes sweep caps."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.server import publish_timing
from reflexio.server.api_endpoints import publisher_api
from reflexio.server.routes import interactions
from reflexio.server.services.lineage.gc_scheduler import LineageGCScheduler
from reflexio.server.services.storage import retention_sweep
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

from .test_generation_service_publish_timing import _publish_request, _reflexio


@pytest.fixture
def reflexio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Reflexio]:
    monkeypatch.setenv(publish_timing.ENV_ENABLED, "true")
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )
    monkeypatch.setattr(retention_sweep, "_library_last_sweep", {})
    monkeypatch.setattr(
        retention_sweep, "get_row_retention_limits", lambda: {"requests": 2}
    )
    instance = _reflexio(str(tmp_path))
    config = instance.request_context.configurator.get_config()
    config.lineage_gc.enabled = False
    config.expiry_reclamation.enabled = False
    yield instance


def _request(number: int):
    return _publish_request().model_copy(
        update={"request_id": f"retention-{number}", "evaluation_only": True}
    )


def test_http_persists_without_sweeping_then_scheduler_enforces_caps(
    reflexio: Reflexio,
) -> None:
    app = FastAPI()
    app.include_router(interactions.router)
    # Resolve the real route's two auth dependencies without booting enterprise.
    route = next(
        r
        for r in interactions.router.routes
        if isinstance(r, APIRoute) and r.path == "/api/publish_interaction"
    )
    for dependency in route.dependant.dependencies:
        assert dependency.call is not None
        if dependency.name == "org_id":
            app.dependency_overrides[dependency.call] = lambda: (
                reflexio.request_context.org_id
            )
        elif dependency.name == "_gate":
            app.dependency_overrides[dependency.call] = lambda: None
    storage = reflexio.get_storage()
    assert isinstance(storage, SQLiteStorage)
    with (
        patch.object(publisher_api, "get_reflexio", return_value=reflexio),
        patch.object(
            storage,
            "count_retention_target_rows",
            wraps=storage.count_retention_target_rows,
        ) as count,
        TestClient(app) as client,
    ):
        for number in range(3):
            # No throttle entry can hide a mistakenly invoked sweep.
            retention_sweep._library_last_sweep.clear()
            response = client.post(
                "/api/publish_interaction",
                json=_request(number).model_dump(mode="json"),
            )
            assert response.status_code == 200
            assert response.json()["success"] is True
            # A separate connection cannot see an uncommitted acknowledgement.
            with sqlite3.connect(storage.db_path) as reader:
                assert (
                    reader.execute(
                        "SELECT count(*) FROM requests WHERE request_id = ?",
                        (f"retention-{number}",),
                    ).fetchone()[0]
                    == 1
                )
                assert (
                    reader.execute(
                        "SELECT count(*) FROM interactions WHERE request_id = ?",
                        (f"retention-{number}",),
                    ).fetchone()[0]
                    == 1
                )
        with patch.object(
            storage,
            "add_user_interactions_bulk",
            side_effect=RuntimeError("write failed"),
        ):
            failed = client.post(
                "/api/publish_interaction", json=_request(3).model_dump(mode="json")
            )
        assert failed.json()["success"] is False
        assert storage.get_request("retention-3") is None
        count.assert_not_called()
    assert storage.count_retention_target_rows("requests") == 3
    assert retention_sweep._library_last_sweep == {}
    scheduler = LineageGCScheduler(
        request_context_factory=lambda _: reflexio.request_context,
        bootstrap_org_id=reflexio.request_context.org_id,
    )
    # Real discovery, project loop and deletion, not a stubbed sweep callback.
    interval = scheduler._run_once()
    assert (
        interval
        == reflexio.request_context.configurator.get_config().lineage_gc.poll_interval_seconds
    )
    assert storage.count_retention_target_rows("requests") == 2


def test_embedded_deferred_publish_still_sweeps_and_throttles(
    reflexio: Reflexio,
) -> None:
    storage = reflexio.get_storage()
    assert isinstance(storage, SQLiteStorage)
    with (
        patch.object(
            storage,
            "count_retention_target_rows",
            wraps=storage.count_retention_target_rows,
        ) as count,
        publish_timing.collect(),
    ):
        first = reflexio.publish_interaction(_request(0), defer_learning=True)
        assert first.success, first.message
        assert count.call_count == 1
        phases = publish_timing.snapshot()
        assert phases is not None and "retention_ms" in phases
        second = reflexio.publish_interaction(_request(1), defer_learning=True)
        assert second.success, second.message
        assert count.call_count == 1
        retention_sweep._library_last_sweep.clear()
        third = reflexio.publish_interaction(_request(2), defer_learning=True)
        assert third.success, third.message
        assert count.call_count == 2
    assert storage.count_retention_target_rows("requests") == 2


def test_server_scope_resets_after_failure_and_does_not_suppress_other_threads(
    reflexio: Reflexio,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    storage = reflexio.get_storage()
    assert isinstance(storage, SQLiteStorage)
    org = reflexio.request_context.org_id
    with (
        pytest.raises(RuntimeError, match="publish failed"),
        retention_sweep.scheduler_managed_retention(),
    ):
        assert (
            retention_sweep.maybe_sweep_retention_caps_for_library(org, storage) is None
        )
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert (
                executor.submit(
                    retention_sweep.maybe_sweep_retention_caps_for_library, org, storage
                ).result()
                is not None
            )
        raise RuntimeError("publish failed")
    retention_sweep._library_last_sweep.clear()
    assert (
        retention_sweep.maybe_sweep_retention_caps_for_library(org, storage) is not None
    )
