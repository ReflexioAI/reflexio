"""Tests for the durable atomic enqueue path (REFLEXIO_DURABLE_LEARNING_QUEUE=true).

Uses SQLite-backed GenerationService — no real LLM/embedding calls needed.
"""

from __future__ import annotations

import datetime
import tempfile
from datetime import UTC
from unittest.mock import Mock, patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import (
    InteractionData,
    PublishUserInteractionRequest,
)
from reflexio.models.config_schema import (
    Config,
    PlaybookConfig,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.cache.reflexio_cache import clear_reflexio_cache
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.configurator.configurator import DefaultConfigurator
from reflexio.server.services.durable_learning.worker import DurableLearningWorker
from reflexio.server.services.generation_service import (
    GenerationService,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture(autouse=True)
def stop_background_scheduler(monkeypatch):
    monkeypatch.setattr(
        "reflexio.server.services.durable_learning.local.ensure_local_extraction",
        lambda _: None,
    )


def _make_svc(tmp_dir: str, org_id: str = "test_org") -> GenerationService:
    return GenerationService(
        llm_client=LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini")),
        request_context=RequestContext(org_id=org_id, storage_base_dir=tmp_dir),
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
                "reflexio.server.services.playbook.service.PlaybookGenerationService",
                side_effect=AssertionError("playbook extraction should be deferred"),
            ),
        ):
            result = svc.run(req, defer_learning=True)

        assert result.request_id == "r1"
        assert svc.storage is not None
        assert svc.storage.get_request("r1") is not None

        claim = svc.storage.claim_extraction("test-worker", 300)
        assert claim is not None and claim[0] == "u1"


def test_public_publish_to_durable_worker_persists_profile_and_playbook(monkeypatch):
    """The public deferred-publish path survives without the retired service.

    This exercises the complete in-process boundary: public facade, atomic queue
    enqueue, real durable worker, extraction, and persisted profile/playbook rows.
    The suite-level LiteLLM double is the only mocked external boundary.
    """
    monkeypatch.setenv("REFLEXIO_DURABLE_LEARNING_QUEUE", "true")
    monkeypatch.setenv("REFLEXIO_EMBEDDING_PROVIDER", "off")

    with tempfile.TemporaryDirectory() as tmp_dir:
        org_id = "durable_public_boundary"
        db_path = f"{tmp_dir}/durable-boundary.db"
        configurator = DefaultConfigurator(org_id=org_id, base_dir=tmp_dir)
        configurator.set_config(
            Config(
                storage_config=StorageConfigSQLite(db_path=db_path),
                window_size=1,
                stride_size=1,
                agent_context_prompt="A customer-support assistant.",
                profile_extractor_config=ProfileExtractorConfig(
                    extraction_definition_prompt=(
                        "Extract stable user preferences and account context."
                    ),
                ),
                user_playbook_extractor_config=PlaybookConfig(
                    extraction_definition_prompt=(
                        "Extract instructions that would improve the next response."
                    ),
                ),
            )
        )
        reflexio = Reflexio(
            org_id=org_id,
            storage_base_dir=tmp_dir,
            configurator=configurator,
        )

        response = reflexio.publish_interaction(
            {
                "request_id": "durable-public-request",
                "user_id": "durable-public-user",
                "session_id": "durable-public-session",
                "agent_version": "v1",
                "source": "durable-boundary-test",
                "interaction_data_list": [
                    {
                        "content": (
                            "I prefer concise answers. Next time, confirm the account "
                            "number before suggesting a billing change."
                        ),
                        "created_at": int(datetime.datetime.now(UTC).timestamp()),
                    }
                ],
            },
            defer_learning=True,
        )

        assert response.success is True
        storage = reflexio.get_storage()
        assert storage.count_all_profiles() == 0
        assert storage.count_user_playbooks() == 0

        worker = DurableLearningWorker(
            lambda _requested_org_id: reflexio.request_context,
            instance_id="durable-boundary-worker",
        )
        assert worker.drain_org(org_id, batch_size=4, lease_seconds=300) == 4
        assert storage.count_all_profiles() > 0
        assert storage.count_user_playbooks() > 0

    clear_reflexio_cache()


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
            "admit_extraction",
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
        assert isinstance(svc.storage, SQLiteStorage)

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


# --- Org-allowlist gate on durable enqueue ---------------------------------
#
# The allowlist (REFLEXIO_DURABLE_LEARNING_QUEUE_ORG_ALLOWLIST) only *narrows*
# the durable path: empty/unset = current global behavior (all orgs durable
# when the flag is on); a false flag always wins.
