"""Host readiness is optional and cannot leak between FastAPI app instances."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.server.readiness import ReadinessCheck, ReadinessResponse
from reflexio.server.routes.system import router


def test_health_is_app_scoped_and_standalone_shape_is_preserved():
    standalone, enterprise = FastAPI(), FastAPI()
    for app in (standalone, enterprise):
        app.include_router(router)
    enterprise.state.readiness_provider = lambda: ReadinessResponse(
        status="unhealthy",
        checks={
            "database": ReadinessCheck(status="failed", reason="dependency_unavailable")
        },
    )
    with TestClient(standalone) as oss, TestClient(enterprise) as hosted:
        assert oss.get("/health").json() == {"status": "healthy"}
        response = hosted.get("/health")
        assert response.status_code == 503
        assert response.headers["Cache-Control"] == "no-store"
        assert (
            response.json()["checks"]["database"]["reason"] == "dependency_unavailable"
        )
        assert oss.get("/health").status_code == 200
