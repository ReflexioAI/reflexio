"""Shared retrieval and the single audited consolidation dispatch contract."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.services.playbook.components.consolidator import (
    PlaybookConsolidator,
)


def _make_user_playbook(
    idx: int,
    playbook_name: str = "test_fb",
    content: str | None = None,
    trigger: str | None = None,
    source_interaction_ids: list[int] | None = None,
    user_playbook_id: int = 0,
    source_span: str | None = None,
    notes: str | None = None,
    reader_angle: str | None = None,
) -> UserPlaybook:
    """Helper to create a UserPlaybook object for tests."""
    return UserPlaybook(
        user_playbook_id=user_playbook_id,
        agent_version="v1",
        request_id=f"req_{idx}",
        playbook_name=playbook_name,
        content=content or f"content_{idx}",
        trigger=trigger or f"condition_{idx}",
        source="test",
        source_interaction_ids=source_interaction_ids or [],
        source_span=source_span,
        notes=notes,
        reader_angle=reader_angle,
    )


@pytest.fixture
def mock_consolidator():
    """Create a PlaybookConsolidator with mocked dependencies."""
    mock_request_context = MagicMock()
    mock_request_context.storage = MagicMock()
    mock_request_context.prompt_manager = MagicMock()
    mock_request_context.prompt_manager.render_prompt.return_value = "mock prompt"

    mock_llm_client = MagicMock()

    def _generate_with_provenance(*args, **kwargs):
        value = mock_llm_client.generate_chat_response(*args, **kwargs)
        if isinstance(value, CompletionResult):
            return value
        return CompletionResult(value=value, provenance=ModelProvenance())

    mock_llm_client.generate_chat_response_with_provenance.side_effect = (
        _generate_with_provenance
    )

    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as mock_svm:
        mock_svm.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-test"
        }
        return PlaybookConsolidator(
            request_context=mock_request_context, llm_client=mock_llm_client
        )


class TestRetrieveExistingPlaybooks:
    """Tests for retrieve_existing_playbooks."""

    def test_copied_instructions_are_not_model_consolidation_inputs(
        self, mock_consolidator
    ):
        import json

        from reflexio.server.services.playbook.source_quoted_overflow import POLICY

        candidate = _make_user_playbook(0)
        copied = _make_user_playbook(
            1, user_playbook_id=10, notes=json.dumps({"policy": POLICY})
        )
        ordinary = _make_user_playbook(2, user_playbook_id=20)
        storage = mock_consolidator.request_context.storage
        storage._get_embedding.return_value = [0.1, 0.2, 0.3]
        storage.search_user_playbooks.return_value = [copied, ordinary]
        assert mock_consolidator.retrieve_existing_playbooks([candidate]) == [ordinary]
        assert storage.search_user_playbooks.call_args.args[1].exclude_source_quoted

    def test_audited_prefetched_inputs_also_exclude_copied_instructions(
        self, mock_consolidator, monkeypatch
    ):
        import json

        from reflexio.models.config_schema import DeduplicationConfig
        from reflexio.server.services.playbook.source_quoted_overflow import POLICY

        monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
        mock_consolidator._dedup_config = DeduplicationConfig()
        copied = _make_user_playbook(
            1, user_playbook_id=10, notes=json.dumps({"policy": POLICY})
        )
        with patch(
            "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
            return_value=([], [], []),
        ) as reconcile:
            mock_consolidator.deduplicate(
                [[_make_user_playbook(0)]],
                request_id="later",
                user_id="u",
                agent_version="v1",
                existing_playbooks=[copied],
            )
        assert reconcile.call_args.args[2] == []

    def test_with_embeddings(self, mock_consolidator):
        """Test retrieval embeds queries via the storage's own embedding path."""
        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        existing_fb = _make_user_playbook(
            1, user_playbook_id=100, trigger="billing inquiry"
        )

        storage = mock_consolidator.request_context.storage
        storage._get_embedding.return_value = [0.1, 0.2, 0.3]
        storage.search_user_playbooks.return_value = [existing_fb]

        result = mock_consolidator.retrieve_existing_playbooks([new_fb])

        assert len(result) == 1
        assert result[0].user_playbook_id == 100
        storage._get_embedding.assert_called_once_with(
            "user asks about billing", purpose="query"
        )
        # The dedup path must never use the client's default embedding model:
        # it resolves the OSS default, which does not match the indexed vectors
        # (and the enterprise embedding daemon rejects it with 409).
        mock_consolidator.client.get_embeddings.assert_not_called()

    def test_fallback_embeddings_use_storage_model_and_query_prefix(
        self, mock_consolidator
    ):
        """Without a storage embedding path, pin the client to the storage's model."""
        storage = mock_consolidator.request_context.storage
        storage._get_embedding = None
        storage.embedding_model_name = "local/test-embedding-model"
        storage.embedding_dimensions = 768
        storage.search_user_playbooks.return_value = []
        mock_consolidator.client.get_embeddings.return_value = [[0.1] * 768]

        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        mock_consolidator.retrieve_existing_playbooks([new_fb])

        mock_consolidator.client.get_embeddings.assert_called_once_with(
            ["user asks about billing"],
            model="local/test-embedding-model",
            dimensions=768,
        )

    def test_fallback_to_text_search(self, mock_consolidator):
        """Test fallback to text-only search when embedding generation fails."""
        new_fb = _make_user_playbook(0)
        existing_fb = _make_user_playbook(1, user_playbook_id=200)

        storage = mock_consolidator.request_context.storage
        storage._get_embedding.side_effect = Exception("embed error")
        storage.search_user_playbooks.return_value = [existing_fb]

        result = mock_consolidator.retrieve_existing_playbooks([new_fb])

        assert len(result) == 1

    def test_empty_query_texts(self, mock_consolidator):
        """Test that empty when_condition playbooks return no results."""
        fb = UserPlaybook(
            agent_version="v1",
            request_id="req1",
            playbook_name="test",
            content="",
            trigger="",
        )

        result = mock_consolidator.retrieve_existing_playbooks([fb])

        assert result == []

    def test_deduplicates_by_id(self, mock_consolidator):
        """Test that duplicate existing playbooks from multiple queries are deduplicated."""
        fb1 = _make_user_playbook(0, trigger="query1")
        fb2 = _make_user_playbook(1, trigger="query2")

        shared_existing = _make_user_playbook(99, user_playbook_id=500)

        mock_consolidator.client.get_embeddings.return_value = [
            [0.1],
            [0.2],
        ]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            shared_existing
        ]

        result = mock_consolidator.retrieve_existing_playbooks([fb1, fb2])

        # Should only appear once despite being returned for both queries
        assert len(result) == 1


class TestRetrieveExistingPlaybooksWithUserId:
    """Tests for retrieve_existing_playbooks with user_id filter."""

    def test_user_id_passed_to_search(self, mock_consolidator):
        """Test that user_id is passed through to the search request."""
        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        existing_fb = _make_user_playbook(1, user_playbook_id=100)

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing_fb
        ]

        mock_consolidator.retrieve_existing_playbooks([new_fb], user_id="user_abc")

        # Verify search was called with user_id in the SearchUserPlaybookRequest
        call_args = (
            mock_consolidator.request_context.storage.search_user_playbooks.call_args
        )
        search_request = call_args[0][0]
        assert search_request.user_id == "user_abc"

    def test_none_user_id_passed_to_search(self, mock_consolidator):
        """Test that None user_id is passed through correctly."""
        new_fb = _make_user_playbook(0, trigger="some condition")

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []

        mock_consolidator.retrieve_existing_playbooks([new_fb], user_id=None)

        call_args = (
            mock_consolidator.request_context.storage.search_user_playbooks.call_args
        )
        search_request = call_args[0][0]
        assert search_request.user_id is None


class TestRetrievalResilience:
    """A failed hybrid search degrades recall; it must not fail the batch.

    Losing a search result can only cause consolidation to miss an existing row
    (a duplicate is stored). Raising instead would discard every candidate in the
    batch, including the LLM work already paid for.
    """

    def test_search_failure_is_skipped_not_raised(self, mock_consolidator):
        first = _make_user_playbook(0, trigger="first condition")
        second = _make_user_playbook(1, trigger="second condition")
        survivor = _make_user_playbook(9, user_playbook_id=77)

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.side_effect = [
            RuntimeError("vector backend unavailable"),
            [survivor],
        ]

        result = mock_consolidator.retrieve_existing_playbooks([first, second])

        # The surviving query's rows are still returned.
        assert [playbook.user_playbook_id for playbook in result] == [77]

    def test_all_searches_failing_returns_empty_not_raises(self, mock_consolidator):
        playbook = _make_user_playbook(0, trigger="only condition")
        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.side_effect = (
            RuntimeError("vector backend unavailable")
        )

        assert mock_consolidator.retrieve_existing_playbooks([playbook]) == []

    def test_not_ready_yet_is_skipped_at_warning_not_error(
        self, mock_consolidator, caplog
    ):
        """A deploy's warm-up ("embedding not ready yet") degrades recall like
        any failed search, but logs WARNING: it is expected and clears itself."""
        import logging

        from reflexio.server.services.storage.error import ReadinessUnavailableError

        first = _make_user_playbook(0, trigger="first condition")
        second = _make_user_playbook(1, trigger="second condition")
        survivor = _make_user_playbook(9, user_playbook_id=77)
        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.side_effect = [
            ReadinessUnavailableError("embedding_readiness_unavailable"),
            [survivor],
        ]

        with caplog.at_level(logging.WARNING):
            result = mock_consolidator.retrieve_existing_playbooks([first, second])

        assert [playbook.user_playbook_id for playbook in result] == [77]
        levels = {
            r.levelno for r in caplog.records if "playbook_consolidation" in r.message
        }
        assert levels == {logging.WARNING}


@pytest.mark.parametrize("with_existing", [False, True])
def test_every_nonempty_batch_uses_audited_method(
    mock_consolidator, monkeypatch, with_existing
):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    candidate = _make_user_playbook(0)
    existing = [_make_user_playbook(1, user_playbook_id=12)] if with_existing else []
    with patch(
        "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
        return_value=([candidate], [], []),
    ) as reconcile:
        assert mock_consolidator.deduplicate(
            [[candidate]],
            generation_request_id="generation",
            user_id="owner",
            agent_version="v1",
            existing_playbooks=existing,
        ) == ([candidate], [], [])
    reconcile.assert_called_once_with(
        mock_consolidator,
        [candidate],
        existing,
        user_id="owner",
        agent_version="v1",
        generation_request_id="generation",
    )


def test_empty_work_never_calls_a_provider(mock_consolidator, monkeypatch):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", "false")
    with patch(
        "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
        side_effect=AssertionError("empty work has no method to audit"),
    ):
        assert mock_consolidator.deduplicate(
            [],
            generation_request_id="generation",
            agent_version="v1",
        ) == ([], [], [])


@pytest.mark.parametrize("mock_value", ["true", "True"])
def test_explicit_mock_mode_does_not_run_a_paid_method(
    mock_consolidator, monkeypatch, mock_value
):
    monkeypatch.setenv("MOCK_LLM_RESPONSE", mock_value)
    candidate = _make_user_playbook(0)
    with patch(
        "reflexio.server.services.playbook.components.reconciliation.reconcile_playbooks",
        side_effect=AssertionError("explicit mock mode"),
    ):
        assert mock_consolidator.deduplicate(
            [[candidate]],
            generation_request_id="generation",
            agent_version="v1",
        ) == ([candidate], [], [])
