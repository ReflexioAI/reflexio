from __future__ import annotations

from fastapi.testclient import TestClient

from reflexio.server.llm import embedding_service
from reflexio.server.llm.embedding_service import create_embedding_app


def test_health_exposes_configured_model(monkeypatch) -> None:
    configured_model = "local/multilingual-e5-small"
    monkeypatch.setattr(
        embedding_service,
        "_embed_texts",
        lambda _model, texts: [[1.0, 0.0] for _ in texts],
    )
    app = create_embedding_app(
        default_model=configured_model,
        allowed_models={configured_model},
    )

    with TestClient(app) as client:
        assert client.get("/health").json()["configured_model"] == configured_model
        response = client.post(
            "/v1/embeddings",
            json={"model": configured_model, "input": ["中文", "English"]},
        )

    assert response.status_code == 200
    assert response.json()["model"] == configured_model
    assert len(response.json()["data"]) == 2


def test_dedicated_service_rejects_model_other_than_baked_model(monkeypatch) -> None:
    configured_model = "local/multilingual-e5-small"
    monkeypatch.setattr(
        embedding_service,
        "_embed_texts",
        lambda _model, texts: [[1.0, 0.0] for _ in texts],
    )
    app = create_embedding_app(
        default_model=configured_model,
        allowed_models={configured_model},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "local/nomic-embed-text-v1.5", "input": "hello"},
        )

    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]


def test_multilingual_e5_rejects_non_storage_dimensions(monkeypatch) -> None:
    configured_model = "local/multilingual-e5-small"
    monkeypatch.setattr(
        embedding_service,
        "_embed_texts",
        lambda _model, texts: [[1.0] * 512 for _ in texts],
    )
    app = create_embedding_app(
        default_model=configured_model,
        allowed_models={configured_model},
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": configured_model, "input": "中文", "dimensions": 384},
        )

    assert response.status_code == 400
    assert "fixed 512-dimension" in response.json()["detail"]


def test_default_local_service_rejects_multilingual_e5(monkeypatch) -> None:
    monkeypatch.setattr(
        embedding_service,
        "_embed_texts",
        lambda _model, texts: [[1.0, 0.0] for _ in texts],
    )

    with TestClient(create_embedding_app()) as client:
        response = client.post(
            "/v1/embeddings",
            json={"model": "local/multilingual-e5-small", "input": "中文"},
        )

    assert response.status_code == 400
