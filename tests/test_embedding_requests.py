"""The shared HTTP budget retains custom values without server imports."""

import pytest

from reflexio.embedding_requests import embedding_request_max_texts


@pytest.mark.parametrize(
    ("saved", "expected"),
    [
        (None, 32),
        ("", 32),
        ("  ", 32),
        ("2", 2),
        ("64", 64),
        (" 17 ", 17),
        ("0", 32),
        ("-1", 32),
        ("nan", 32),
        ("1.5", 32),
    ],
)
def test_embedding_request_capacity(monkeypatch, saved, expected):
    if saved is None:
        monkeypatch.delenv(
            "REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", raising=False
        )
    else:
        monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", saved)
    assert embedding_request_max_texts() == expected


def test_sdk_http_caller_uses_the_shared_capacity(monkeypatch):
    import json

    import httpx

    from reflexio.server.llm.providers import embedding_service_provider as service

    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "internal_service")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_URL", "http://offline")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_SERVICE_MAX_TEXTS_PER_REQUEST", "2")
    monkeypatch.delenv("REFLEXIO_QUERY_EMBEDDING_SERVICE_URL", raising=False)
    batches = []

    def respond(request):
        texts = json.loads(request.content)["input"]
        batches.append(texts)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(value)]}
                    for i, value in enumerate(texts)
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(service, "_http_client", lambda: client)
        result = service.get_service_embeddings(
            ["0", "1", "2", "3", "4"], model="local/synthetic"
        )
    assert result == [[float(i)] for i in range(5)]
    assert [len(batch) for batch in batches] == [2, 2, 1]
