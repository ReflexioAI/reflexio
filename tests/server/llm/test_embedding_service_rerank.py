from unittest.mock import patch

from fastapi.testclient import TestClient

from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.rerank.cross_encoder_reranker import (
    RERANK_MODEL,
    CrossEncoderUnavailableError,
)


def _client() -> TestClient:
    return TestClient(create_embedding_app())


def test_rerank_endpoint_returns_scores_in_input_order() -> None:
    with patch(
        "reflexio.server.llm.embedding_service._score_pairs_local",
        return_value=[0.25, -1.5],
    ) as mock_score:
        response = _client().post(
            "/v1/rerank",
            json={
                "model": RERANK_MODEL,
                "query": "italian food",
                "documents": ["pasta", "weather"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "model": RERANK_MODEL,
        "data": [{"index": 0, "score": 0.25}, {"index": 1, "score": -1.5}],
    }
    mock_score.assert_called_once_with("italian food", ["pasta", "weather"])


def test_rerank_endpoint_accepts_empty_documents() -> None:
    with patch(
        "reflexio.server.llm.embedding_service._score_pairs_local",
        return_value=[],
    ) as mock_score:
        response = _client().post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": "q", "documents": []},
        )

    assert response.status_code == 200
    assert response.json()["data"] == []
    mock_score.assert_called_once_with("q", [])


def test_rerank_endpoint_rejects_unsupported_model() -> None:
    response = _client().post(
        "/v1/rerank",
        json={"model": "other/model", "query": "q", "documents": ["doc"]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model: other/model"


def test_rerank_endpoint_reports_unavailable_cross_encoder() -> None:
    with patch(
        "reflexio.server.llm.embedding_service._score_pairs_local",
        side_effect=CrossEncoderUnavailableError("no model"),
    ):
        response = _client().post(
            "/v1/rerank",
            json={"model": RERANK_MODEL, "query": "q", "documents": ["doc"]},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "no model"
