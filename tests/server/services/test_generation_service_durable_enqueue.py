"""Tests for the durable atomic enqueue path (REFLEXIO_DURABLE_LEARNING_QUEUE=true).

Uses SQLite-backed GenerationService — no real LLM/embedding calls needed.
"""

from __future__ import annotations

import datetime
import tempfile
from datetime import UTC
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.generation_service import GenerationService


def _make_svc(tmp_dir: str) -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(org_id="test_org", storage_base_dir=tmp_dir),
    )


def _publish_request(*, user_id: str, request_id: str) -> PublishUserInteractionRequest:
    return PublishUserInteractionRequest(
        request_id=request_id,
        user_id=user_id,
        interaction_data_list=[
            InteractionData(
                content="test interaction",
                created_at=int(datetime.datetime.now(UTC).timestamp()),
            )
        ],
        session_id="test-session",
    )


def test_durable_enqueue_persists_job_atomically(monkeypatch):
    """When REFLEXIO_DURABLE_LEARNING_QUEUE=true and defer_learning=True, a
    pending learning job is persisted in the same atomic scope as the request
    and its interactions."""
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")

    with tempfile.TemporaryDirectory() as tmp_dir:
        svc = _make_svc(tmp_dir)
        req = _publish_request(user_id="u1", request_id="r1")

        with (
            patch(
                "reflexio.server.services.generation_service.ProfileGenerationService",
                side_effect=AssertionError("profile extraction should be deferred"),
            ),
            patch(
                "reflexio.server.services.generation_service.PlaybookGenerationService",
                side_effect=AssertionError("playbook extraction should be deferred"),
            ),
        ):
            result = svc.run(req, defer_learning=True)

        assert result.request_id == "r1"
        assert svc.storage is not None
        assert svc.storage.get_request("r1") is not None

        jobs = svc.storage.claim_learning_jobs(
            claimed_by="test-worker", limit=10, lease_seconds=300
        )
        assert any(j.user_id == "u1" for j in jobs), "expected one pending job for u1"


def test_enqueue_failure_rolls_back_interactions(monkeypatch):
    """Zero-loss proof: if enqueue_learning_job raises inside commit_scope, the
    entire transaction rolls back — neither the request nor its interactions
    persist."""
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")

    with tempfile.TemporaryDirectory() as tmp_dir:
        svc = _make_svc(tmp_dir)
        req = _publish_request(user_id="u2", request_id="r2")

        monkeypatch.setattr(
            type(svc.storage),
            "enqueue_learning_job",
            Mock(side_effect=RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            svc.run(req, defer_learning=True)

        # Both the request and its interactions must be absent (rolled back).
        assert svc.storage is not None
        assert svc.storage.get_request("r2") is None, (
            "request must not persist when enqueue_learning_job fails inside commit_scope"
        )
        assert svc.storage.get_user_interaction("u2") == [], (
            "interactions must not persist when enqueue_learning_job fails inside commit_scope"
        )


def test_no_get_embeddings_inside_scope_when_prepare_degraded(monkeypatch):
    """Fix 3: when prepare_interaction_embeddings degrades (EmbeddingUnavailableError
    sets embedding=[]), add_user_interactions_bulk with embeddings_prepared=True must
    NOT call get_embeddings inside the commit_scope (which holds the SQLite RLock)."""
    from reflexio.server.llm.providers.embedding_service_provider import (
        EmbeddingUnavailableError,
    )

    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")

    with tempfile.TemporaryDirectory() as tmp_dir:
        svc = _make_svc(tmp_dir)
        req = _publish_request(user_id="u3", request_id="r3")

        # Simulate embedding service being unavailable during prepare.
        monkeypatch.setattr(
            svc.storage.llm_client,
            "get_embeddings",
            Mock(side_effect=EmbeddingUnavailableError("unavailable")),
        )

        # Run the durable path — prepare degrades to [], then embeddings_prepared=True
        # prevents a second get_embeddings call inside the scope.
        result = svc.run(req, defer_learning=True)

        assert result.request_id == "r3"
        # Data landed despite degraded embeddings.
        assert svc.storage is not None
        assert svc.storage.get_request("r3") is not None
        interactions = svc.storage.get_user_interaction("u3")
        assert len(interactions) == 1
        assert interactions[0].embedding == [], (
            "degraded embedding should be empty list"
        )

        # get_embeddings was called once (prepare step) but not again inside the scope.
        # After the prepare call raises, the mock records exactly 1 call.
        assert svc.storage.llm_client.get_embeddings.call_count == 1, (
            "get_embeddings must be called at most once (prepare phase), "
            "never inside commit_scope"
        )
