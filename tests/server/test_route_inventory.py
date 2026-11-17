"""Tests for the shared route enumerator and its no-raw-``.routes`` guard."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI

import reflexio
from reflexio.server.api import core_router, create_app
from reflexio.server.route_inventory import api_route_contexts, find_raw_route_reads

_REPO_ROOT = Path(reflexio.__file__).resolve().parents[1]


def _gate() -> None:
    return None


def test_included_routes_carry_prefix_and_include_dependencies() -> None:
    inner = APIRouter()

    @inner.get("/items/{item_id}")
    def read_item(item_id: str) -> str:
        return item_id

    app = FastAPI()
    app.include_router(inner, prefix="/a", dependencies=[Depends(_gate)])
    app.include_router(inner, prefix="/b")

    contexts = {ctx.path: ctx for ctx in api_route_contexts(app)}

    # Only path operations: FastAPI's docs/OpenAPI routes are not APIRoutes.
    assert set(contexts) == {"/a/items/{item_id}", "/b/items/{item_id}"}
    assert contexts["/a/items/{item_id}"].endpoint is read_item
    gated = [d.call for d in contexts["/a/items/{item_id}"].dependant.dependencies]
    ungated = [d.call for d in contexts["/b/items/{item_id}"].dependant.dependencies]
    assert gated == [_gate]
    assert ungated == []


def test_core_router_enumeration_reaches_every_domain_router() -> None:
    paths = {ctx.path for ctx in api_route_contexts(core_router)}
    # One handler from the first and the last included domain router.
    assert "/health" in paths
    assert "/api/search_profiles" in paths
    app_paths = {ctx.path for ctx in api_route_contexts(create_app())}
    assert paths <= app_paths


def test_no_raw_route_enumeration_outside_the_helper() -> None:
    """``app.routes`` / ``router.routes`` is FastAPI's internal tree since
    0.137 and silently omits included handlers; enumerate through
    ``api_route_contexts`` instead."""
    sources = [
        path
        for root in ("reflexio", "tests", "scripts")
        for path in (_REPO_ROOT / root).rglob("*.py")
        if ".venv" not in path.parts
    ]
    assert len(sources) > 100, "scan found too few files to mean anything"
    assert find_raw_route_reads(sources) == []


def test_raw_route_scan_detects_each_receiver_shape(tmp_path: Path) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text(
        "for r in app.routes: pass\n"
        "x = core_router.routes\n"
        "y = client.app.routes\n"
        "z = create_app().routes\n"
        "ok = provider.routes\n",
        encoding="utf-8",
    )
    assert find_raw_route_reads([sample]) == [
        f"{sample}:1",
        f"{sample}:2",
        f"{sample}:3",
        f"{sample}:4",
    ]
