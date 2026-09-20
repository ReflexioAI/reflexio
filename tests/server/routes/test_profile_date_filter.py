"""Creation-time drill-down is opt-in; ordinary date filters keep their meaning."""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from reflexio.models.api_schema.retriever_schema import GetUserProfilesResponse
from reflexio.server.routes import profiles


def test_profile_date_field_transport_and_validation(monkeypatch):
    service = MagicMock()
    service.get_all_profiles.return_value = GetUserProfilesResponse(
        success=True, user_profiles=[]
    )
    monkeypatch.setattr(profiles.reflexio_cache, "get_reflexio", lambda **_: service)
    app = FastAPI()
    app.include_router(profiles.router)
    app.dependency_overrides[profiles.default_get_org_id] = lambda: "org-test"
    client = TestClient(app)
    for query, expected in [
        ("", "last_modified_timestamp"),
        ("&date_field=created_at", "created_at"),
    ]:
        response = client.get(
            "/api/get_all_profiles?limit=1&user_id=u&start_time=10&end_time=20" + query
        )
        assert response.status_code == 200
        kwargs = service.get_all_profiles.call_args.kwargs
        assert (
            kwargs["date_field"],
            kwargs["start_time"],
            kwargs["end_time"],
            kwargs["limit"],
            kwargs["user_id"],
        ) == (expected, 10, 20, 1, "u")
    service.reset_mock()
    assert client.get("/api/get_all_profiles?date_field=content").status_code == 422
    service.get_all_profiles.assert_not_called()
