from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from reflexio.server.api import create_app
from reflexio.server.llm.rerank.common import CrossEncoderUnavailableError


def test_explicit_profile_rerank_returns_503_when_reranker_unavailable() -> None:
    fake_reflexio = MagicMock()
    fake_reflexio.rerank_user_profiles.side_effect = CrossEncoderUnavailableError(
        "internal service failed at http://private-inference:8089"
    )
    app = create_app(get_org_id=lambda: "org-1")

    with patch(
        "reflexio.server.routes.search.reflexio_cache.get_reflexio",
        return_value=fake_reflexio,
    ):
        response = TestClient(app).post(
            "/api/rerank_user_profiles",
            json={
                "user_id": "user-1",
                "query": "Italian food",
                "profile_ids": ["profile-1"],
            },
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Reranking is currently unavailable"}
    assert "private-inference" not in response.text
    fake_reflexio.rerank_user_profiles.assert_called_once()
