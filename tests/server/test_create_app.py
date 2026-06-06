from fastapi.testclient import TestClient

from reflexio.server.api import create_app


def _paths(app) -> set[str]:
    return {r.path for r in app.routes}


def test_mount_data_plane_true_includes_core_routes():
    app = create_app(mount_data_plane=True)
    paths = _paths(app)
    assert "/api/search_profiles" in paths  # a core_router data-plane route
    assert "/meta/version" in paths


def test_mount_data_plane_false_excludes_core_routes_keeps_scaffolding():
    app = create_app(mount_data_plane=False)
    paths = _paths(app)
    assert "/api/search_profiles" not in paths  # data-plane routes gone
    assert "/meta/version" in paths  # scaffolding stays


def test_mount_data_plane_false_still_mounts_additional_routers():
    from fastapi import APIRouter

    r = APIRouter()

    @r.get("/cp/ping")
    def _ping() -> dict:
        return {"ok": True}

    app = create_app(mount_data_plane=False, additional_routers=[r])
    with TestClient(app) as c:
        assert c.get("/cp/ping").json() == {"ok": True}
