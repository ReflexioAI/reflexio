"""C45 complete-pair planning and shared inference ownership contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.rerank import activation_client
from reflexio.server.llm.rerank.activation import activation_plan, query_view
from reflexio.server.llm.rerank.common import (
    MULTILINGUAL_RERANK_MODEL,
    MULTILINGUAL_RERANK_REVISION,
    CrossEncoderUnavailableError,
)
from reflexio.server.llm.rerank.cross_encoder_model import CrossEncoderRunner

pytestmark = pytest.mark.unit


class CharacterTokenizer:
    def encode(self, query, pair=None, *, add_special_tokens):
        return SimpleNamespace(
            ids=list(range(len(query) + (int(pair) if pair is not None else 0))),
            offsets=[(i, i + 1) for i in range(len(query))],
        )


def test_query_view_boundary_preserves_original_unicode_characters():
    tokenizer = CharacterTokenizer()
    query = "東" * 128 + "♞" + "é" * 128
    assert query_view(query[:256], tokenizer) == query[:256]
    assert query_view(query, tokenizer) == "東" * 128 + "\n[…]\n" + "é" * 128


def test_invalid_offsets_refuse_instead_of_decoding_tokens():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = SimpleNamespace(
        ids=[1] * 257, offsets=[(0, 300)] * 257
    )
    with pytest.raises(ValueError, match="offsets"):
        query_view("q" * 257, tokenizer)


def test_complete_pair_boundary_512_scores_513_preserves_unscored():
    view, indices = activation_plan("q", ["511", "512"], CharacterTokenizer())
    assert view == "q" and indices == [0]


def test_zero_token_pair_refuses():
    with pytest.raises(ValueError, match="contain tokens"):
        activation_plan("", ["0"], CharacterTokenizer())


@pytest.mark.parametrize("raw", [[2.87], [float("nan"), 3.0]])
def test_owner_refuses_partial_or_nonfinite_logits(monkeypatch, raw):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(WordLevel({"[UNK]": 0, "q": 1, "d": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(backend_tokenizer=backend),
        predict=MagicMock(return_value=raw),
    )
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    monkeypatch.setattr(runner, "_get_model", lambda: model)
    with pytest.raises(CrossEncoderUnavailableError):
        runner.score_user_playbook_activation("q", ["d", "d"])


def test_owner_uses_untruncated_clone_raw_activation_and_preserves_overlength(
    monkeypatch,
):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from torch import nn

    backend = Tokenizer(WordLevel({"[UNK]": 0, "q": 1, "d": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A", pair="$A $B")
    backend.enable_truncation(max_length=5)
    model = SimpleNamespace(
        tokenizer=SimpleNamespace(backend_tokenizer=backend),
        predict=MagicMock(return_value=[2.87]),
    )
    runner = CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
    monkeypatch.setattr(runner, "_get_model", lambda: model)
    assert runner.score_user_playbook_activation("q", ["d " * 511, "d " * 512]) == [
        2.87,
        None,
    ]
    pairs = model.predict.call_args.args[0]
    assert pairs == [("q", "d " * 511)]
    assert isinstance(model.predict.call_args.kwargs["activation_fn"], nn.Identity)
    assert backend.truncation["max_length"] == 5


def test_endpoint_uses_existing_owner_search_queue_and_returns_revision():
    runner = SimpleNamespace(
        score_user_playbook_activation=MagicMock(return_value=[2.87, None])
    )
    submitted = []

    async def submit(work, docs, priority, timeout_ms, chunk):
        submitted.append((priority, timeout_ms, chunk))
        return work(docs)

    client = TestClient(
        create_embedding_app(
            reranker_model=MULTILINGUAL_RERANK_MODEL,
            reranker_runner=runner,
            inference_submit=submit,
        )
    )
    response = client.post(
        "/v1/user-playbook-activation",
        json={
            "model": MULTILINGUAL_RERANK_MODEL,
            "query": "q",
            "documents": ["a", "b"],
            "timeout_ms": 1234,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "model": MULTILINGUAL_RERANK_MODEL,
        "revision": MULTILINGUAL_RERANK_REVISION,
        "scores": [2.87, None],
    }
    assert submitted == [("search", 1234, 4)]
    runner.score_user_playbook_activation.assert_called_once_with("q", ["a", "b"])


def test_query_worker_refuses_activation_without_dispatch():
    runner = MagicMock()
    client = TestClient(
        create_embedding_app(
            reranker_model=MULTILINGUAL_RERANK_MODEL,
            reranker_runner=runner,
            query_only=True,
        )
    )
    response = client.post(
        "/v1/user-playbook-activation",
        json={
            "model": MULTILINGUAL_RERANK_MODEL,
            "query": "q",
            "documents": ["a"],
        },
    )
    assert response.status_code == 503
    runner.score_user_playbook_activation.assert_not_called()


@pytest.mark.parametrize(
    "body",
    [
        {"revision": "wrong", "scores": [1.0]},
        {"revision": MULTILINGUAL_RERANK_REVISION, "scores": []},
        {"revision": MULTILINGUAL_RERANK_REVISION, "scores": [True]},
    ],
)
def test_client_refuses_wrong_identity_or_invalid_scores(monkeypatch, body):
    monkeypatch.setattr(activation_client, "search_reranker_service_url", lambda: None)
    monkeypatch.setattr(
        activation_client, "inference_service_url", lambda: "http://inference"
    )
    response = httpx.Response(
        200,
        json={"model": MULTILINGUAL_RERANK_MODEL, **body},
        request=httpx.Request("POST", "http://inference"),
    )
    client = MagicMock()
    client.post.return_value = response
    monkeypatch.setattr(activation_client, "inference_http_client", lambda: client)
    with pytest.raises(CrossEncoderUnavailableError):
        activation_client.score_user_playbook_activation("q", ["d"])


def test_client_keeps_search_route_and_no_local_model(monkeypatch):
    monkeypatch.setattr(
        activation_client, "search_reranker_service_url", lambda: "http://search"
    )
    monkeypatch.setattr(
        activation_client, "validate_search_reranker_service", lambda: "http://search"
    )
    response = httpx.Response(
        200,
        json={
            "model": MULTILINGUAL_RERANK_MODEL,
            "revision": MULTILINGUAL_RERANK_REVISION,
            "scores": [None, 2.87],
        },
        request=httpx.Request("POST", "http://search"),
    )
    client = MagicMock()
    client.post.return_value = response
    monkeypatch.setattr(activation_client, "inference_http_client", lambda: client)
    assert activation_client.score_user_playbook_activation("q", ["a", "b"]) == [
        None,
        2.87,
    ]
    assert client.post.call_args.args[0] == "http://search/v1/user-playbook-activation"


def test_client_wraps_unavailable_search_route(monkeypatch):
    monkeypatch.setattr(
        activation_client, "search_reranker_service_url", lambda: "http://search"
    )
    monkeypatch.setattr(
        activation_client,
        "validate_search_reranker_service",
        MagicMock(
            side_effect=activation_client.EmbeddingUnavailableError("wrong identity")
        ),
    )
    with pytest.raises(CrossEncoderUnavailableError, match="route is unavailable"):
        activation_client.score_user_playbook_activation("q", ["d"])
