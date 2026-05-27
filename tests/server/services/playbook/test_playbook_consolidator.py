"""Tests for playbook consolidation service.

Exercises the dispatch-and-apply path of the new ``PlaybookConsolidationOutput``
discriminated union: ``DuplicateDecision`` is the canonical merge case; other
kinds are covered by ``test_playbook_consolidator_integration.py`` end-to-end.
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.playbook.playbook_consolidator import (
    DuplicateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
)

# ===============================
# Fixtures
# ===============================


def _make_user_playbook(
    idx: int,
    playbook_name: str = "test_fb",
    content: str | None = None,
    trigger: str | None = None,
    source_interaction_ids: list[int] | None = None,
    user_playbook_id: int = 0,
    polarity: str = "positive",
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
        polarity=polarity,  # type: ignore[arg-type]
    )


@pytest.fixture
def mock_consolidator():
    """Create a PlaybookConsolidator with mocked dependencies."""
    mock_request_context = MagicMock()
    mock_request_context.storage = MagicMock()
    mock_request_context.prompt_manager = MagicMock()
    mock_request_context.prompt_manager.render_prompt.return_value = "mock prompt"

    mock_llm_client = MagicMock()

    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as mock_svm:
        mock_svm.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-test"
        }
        return PlaybookConsolidator(
            request_context=mock_request_context, llm_client=mock_llm_client
        )


def _duplicate(
    item_ids: list[str],
    *,
    merged_content: str = "merged content",
    merged_trigger: str = "merged trigger",
    merged_rationale: str = "merged rationale",
    merged_polarity: str = "positive",
) -> DuplicateDecision:
    """Build a ``DuplicateDecision`` with sane defaults for merge-shape tests."""
    return DuplicateDecision(
        item_ids=item_ids,
        merged_content=merged_content,
        merged_trigger=merged_trigger,
        merged_rationale=merged_rationale,
        merged_polarity=merged_polarity,  # type: ignore[arg-type]
    )


# ===============================
# Tests for _format_playbooks_with_prefix
# ===============================


class TestFormatPlaybooksWithPrefix:
    """Tests for _format_playbooks_with_prefix."""

    def test_single_playbook(self, mock_consolidator):
        """Test formatting a single playbook."""
        fb = _make_user_playbook(0, content="do X when Y")
        result = mock_consolidator._format_playbooks_with_prefix([fb], "NEW")
        assert '[NEW-0] Content: "do X when Y"' in result
        assert "Name: test_fb" in result
        assert "Source: test" in result

    def test_multiple_playbooks(self, mock_consolidator):
        """Test formatting multiple playbooks with incrementing indices."""
        playbooks = [_make_user_playbook(i) for i in range(3)]
        result = mock_consolidator._format_playbooks_with_prefix(playbooks, "EXISTING")
        assert "[EXISTING-0]" in result
        assert "[EXISTING-1]" in result
        assert "[EXISTING-2]" in result

    def test_empty_list(self, mock_consolidator):
        """Test formatting empty list returns '(None)'."""
        result = mock_consolidator._format_playbooks_with_prefix([], "NEW")
        assert result == "(None)"


# ===============================
# Tests for _format_new_and_existing_for_prompt
# ===============================


class TestFormatNewAndExistingForPrompt:
    """Tests for _format_new_and_existing_for_prompt."""

    def test_formats_both_lists(self, mock_consolidator):
        """Test that new and existing playbooks are formatted with correct prefixes."""
        new_fbs = [_make_user_playbook(0)]
        existing_fbs = [_make_user_playbook(1)]

        new_text, existing_text = mock_consolidator._format_new_and_existing_for_prompt(
            new_fbs, existing_fbs
        )

        assert "[NEW-0]" in new_text
        assert "[EXISTING-0]" in existing_text

    def test_empty_existing(self, mock_consolidator):
        """Test formatting with empty existing playbooks."""
        new_fbs = [_make_user_playbook(0)]

        new_text, existing_text = mock_consolidator._format_new_and_existing_for_prompt(
            new_fbs, []
        )

        assert "[NEW-0]" in new_text
        assert existing_text == "(None)"


# ===============================
# Tests for _retrieve_existing_playbooks
# ===============================


class TestRetrieveExistingPlaybooks:
    """Tests for _retrieve_existing_playbooks."""

    def test_with_embeddings(self, mock_consolidator):
        """Test retrieval using embeddings for vector search."""
        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        existing_fb = _make_user_playbook(
            1, user_playbook_id=100, trigger="billing inquiry"
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1, 0.2, 0.3]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing_fb
        ]

        result = mock_consolidator._retrieve_existing_playbooks([new_fb])

        assert len(result) == 1
        assert result[0].user_playbook_id == 100
        mock_consolidator.client.get_embeddings.assert_called_once()

    def test_fallback_to_text_search(self, mock_consolidator):
        """Test fallback to text-only search when embedding generation fails."""
        new_fb = _make_user_playbook(0)
        existing_fb = _make_user_playbook(1, user_playbook_id=200)

        mock_consolidator.client.get_embeddings.side_effect = Exception("embed error")
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing_fb
        ]

        result = mock_consolidator._retrieve_existing_playbooks([new_fb])

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

        result = mock_consolidator._retrieve_existing_playbooks([fb])

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

        result = mock_consolidator._retrieve_existing_playbooks([fb1, fb2])

        # Should only appear once despite being returned for both queries
        assert len(result) == 1


# ===============================
# Tests for deduplicate
# ===============================


class TestDeduplicate:
    """Tests for the main deduplicate method."""

    def test_mock_mode_skips_deduplication(self, mock_consolidator):
        """Test that MOCK_LLM_RESPONSE=true skips deduplication."""
        fb1 = _make_user_playbook(0)
        fb2 = _make_user_playbook(1)

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "true"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb1], [fb2]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 2
        assert delete_ids == []

    def test_empty_results(self, mock_consolidator):
        """Test deduplication with no playbooks."""
        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[]], request_id="req1", agent_version="v1"
            )

        assert result == []
        assert delete_ids == []

    def test_error_fallback_returns_all(self, mock_consolidator):
        """Test that LLM call error falls back to returning all playbooks."""
        fb = _make_user_playbook(0)

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.side_effect = Exception(
            "LLM error"
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1
        assert delete_ids == []


# ===============================
# Tests for _build_deduplicated_results
# ===============================


class TestBuildDeduplicatedResults:
    """Tests for ``_build_deduplicated_results`` decision dispatch."""

    def test_merge_group_combines_source_interaction_ids(self, mock_consolidator):
        """A duplicate decision merges source_interaction_ids from all NEW members."""
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[1, 2]),
            _make_user_playbook(1, source_interaction_ids=[3, 4]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[_duplicate(["NEW-0", "NEW-1"])],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert set(result[0].source_interaction_ids) == {1, 2, 3, 4}
        assert delete_ids == []

    def test_independent_decisions_passed_through(self, mock_consolidator):
        """``IndependentDecision`` rows insert the candidate unchanged."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                IndependentDecision(new_id="NEW-0"),
                IndependentDecision(new_id="NEW-1"),
            ],
        )

        result, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 2

    def test_existing_playbooks_to_delete(self, mock_consolidator):
        """A duplicate decision archives EXISTING members for deletion."""
        new_playbooks = [_make_user_playbook(0)]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[_duplicate(["NEW-0", "EXISTING-0"])],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert 999 in delete_ids

    def test_safety_fallback_unhandled_playbooks(self, mock_consolidator):
        """NEW playbooks not referenced by any decision are added via safety fallback."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
            _make_user_playbook(2),
        ]

        # LLM only mentions index 0
        dedup_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-0")],
        )

        result, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # Index 0 via independent decision + index 1 and 2 via safety fallback
        assert len(result) == 3


# ===============================
# Tests for deduplicate happy path and advanced scenarios
# ===============================


class TestDeduplicateHappyPath:
    """Tests for the full deduplicate() flow with LLM mocks returning PlaybookConsolidationOutput."""

    def test_happy_path_with_duplicates(self, mock_consolidator):
        """Full happy path: LLM returns a duplicate decision and an independent."""
        fb0 = _make_user_playbook(0, content="do X when Y", source_interaction_ids=[10])
        fb1 = _make_user_playbook(
            1, content="do X when Y again", source_interaction_ids=[20]
        )
        fb2 = _make_user_playbook(2, content="do Z when W", source_interaction_ids=[30])

        # No existing playbooks found via search
        mock_consolidator.client.get_embeddings.return_value = [
            [0.1],
            [0.2],
            [0.3],
        ]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []

        # LLM merges fb0 and fb1, keeps fb2 as independent
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _duplicate(
                        ["NEW-0", "NEW-1"],
                        merged_content="do X",
                        merged_trigger="when Y",
                    ),
                    IndependentDecision(new_id="NEW-2"),
                ],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb0, fb1], [fb2]], request_id="req_test", agent_version="v1"
            )

        # 1 merged + 1 independent = 2 playbooks
        assert len(result) == 2
        assert delete_ids == []

        # Merged playbook should have combined source_interaction_ids
        merged = result[0]
        assert set(merged.source_interaction_ids) == {10, 20}

        # Independent playbook should be fb2
        assert result[1].content == "do Z when W"

    def test_multiple_extractor_results_nested_lists(self, mock_consolidator):
        """Multiple extractor results (nested list of lists) are flattened correctly."""
        fb0 = _make_user_playbook(0, content="playbook from extractor 1")
        fb1 = _make_user_playbook(1, content="playbook from extractor 2")
        fb2 = _make_user_playbook(2, content="playbook from extractor 3")

        mock_consolidator.client.get_embeddings.return_value = [
            [0.1],
            [0.2],
            [0.3],
        ]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []

        # LLM says all are independent
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    IndependentDecision(new_id="NEW-0"),
                    IndependentDecision(new_id="NEW-1"),
                    IndependentDecision(new_id="NEW-2"),
                ],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb0], [fb1], [fb2]], request_id="req_test", agent_version="v1"
            )

        assert len(result) == 3
        assert delete_ids == []

    def test_all_playbooks_are_duplicates_of_existing(self, mock_consolidator):
        """A NEW playbook merging with an EXISTING entry archives the existing row."""
        fb0 = _make_user_playbook(0, content="do X when Y", source_interaction_ids=[10])
        existing_fb = _make_user_playbook(
            99,
            user_playbook_id=500,
            content="do X when Y (existing)",
            source_interaction_ids=[5],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing_fb
        ]

        # LLM merges NEW-0 with EXISTING-0
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _duplicate(
                        ["NEW-0", "EXISTING-0"],
                        merged_content="do X",
                        merged_trigger="when Y",
                    ),
                ],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb0]], request_id="req_test", agent_version="v1"
            )

        # 1 merged playbook replaces both
        assert len(result) == 1
        # Existing playbook should be marked for deletion
        assert 500 in delete_ids
        # Merged playbook should combine source_interaction_ids from both
        assert set(result[0].source_interaction_ids) == {5, 10}


# ===============================
# Edge cases for _build_deduplicated_results
# ===============================


class TestBuildDeduplicatedResultsEdgeCases:
    """Extended tests for _build_deduplicated_results edge cases."""

    def test_template_fallback_to_existing_playbook(self, mock_consolidator):
        """A duplicate decision with only EXISTING members uses an EXISTING row as template."""
        existing_playbooks = [
            _make_user_playbook(
                0,
                user_playbook_id=100,
                playbook_name="existing_fb",
                source_interaction_ids=[5],
            ),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _duplicate(
                    ["EXISTING-0"],
                    merged_content="merged do",
                    merged_trigger="merged when",
                ),
            ],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=[],
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        # Template should come from existing playbook
        assert result[0].playbook_name == "existing_fb"
        assert 100 in delete_ids

    def test_template_fallback_skips_out_of_range_existing(self, mock_consolidator):
        """A duplicate decision pointing at no resolvable members yields no row."""
        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _duplicate(
                    ["EXISTING-99"],
                    merged_content="merged do",
                    merged_trigger="merged when",
                ),
            ],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=[],
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # Decision should be skipped entirely since no valid template was found
        assert len(result) == 0
        assert delete_ids == []

    def test_source_interaction_ids_combined_from_new_and_existing(
        self, mock_consolidator
    ):
        """Source interaction ids combine across NEW + EXISTING duplicate members."""
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[1, 2]),
        ]
        existing_playbooks = [
            _make_user_playbook(1, user_playbook_id=100, source_interaction_ids=[3, 4]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _duplicate(
                    ["NEW-0", "EXISTING-0"],
                    merged_content="merged",
                    merged_trigger="merged condition",
                ),
            ],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert set(result[0].source_interaction_ids) == {1, 2, 3, 4}
        assert 100 in delete_ids

    def test_source_interaction_ids_deduplication(self, mock_consolidator):
        """Duplicate source interaction ids across members are not repeated."""
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[1, 2]),
            _make_user_playbook(1, source_interaction_ids=[2, 3]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _duplicate(
                    ["NEW-0", "NEW-1"],
                    merged_content="merged",
                    merged_trigger="merged cond",
                ),
            ],
        )

        result, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        # ID 2 should appear only once
        assert result[0].source_interaction_ids == [1, 2, 3]

    def test_unhandled_playbooks_safety_net(self, mock_consolidator):
        """Playbooks not referenced by any decision are added via safety fallback."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
            _make_user_playbook(2),
        ]

        # LLM only mentions index 1 as independent, leaves 0 and 2 unmentioned
        dedup_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-1")],
        )

        result, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 3
        # Index 1 is from independent decision, indices 0 and 2 from safety fallback
        contents = {fb.content for fb in result}
        assert "content_0" in contents
        assert "content_1" in contents
        assert "content_2" in contents

    def test_invalid_item_ids_skipped_in_duplicate(self, mock_consolidator):
        """Unparseable item ids in a duplicate decision's ``item_ids`` are skipped."""
        new_playbooks = [_make_user_playbook(0)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _duplicate(
                    ["BADFORMAT", "NEW-0"],
                    merged_content="merged",
                    merged_trigger="merged trigger",
                ),
            ],
        )

        result, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # BADFORMAT skipped; NEW-0 absorbed into the merged row
        assert len(result) == 1

    def test_independent_for_unknown_new_id_fails_apply(self, mock_consolidator):
        """``IndependentDecision`` referencing an unknown NEW id is counted as failed.

        The unknown reference raises inside ``_apply_one``; the per-decision
        ``try/except`` isolates the failure so the rest of the batch (and the
        safety fallback) still runs.
        """
        new_playbooks = [_make_user_playbook(0)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-99")],
        )

        result, delete_ids = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # Decision failed, but safety fallback adds NEW-0 as-is
        assert len(result) == 1
        assert delete_ids == []


class TestFormatItemsForPrompt:
    """Tests for _format_items_for_prompt (delegates to _format_playbooks_with_prefix)."""

    def test_delegates_with_new_prefix(self, mock_consolidator):
        """Test that _format_items_for_prompt uses 'NEW' prefix."""
        playbooks = [_make_user_playbook(0)]
        result = mock_consolidator._format_items_for_prompt(playbooks)
        assert "[NEW-0]" in result

    def test_empty_list(self, mock_consolidator):
        """Test that empty list returns '(None)'."""
        result = mock_consolidator._format_items_for_prompt([])
        assert result == "(None)"


class TestFormatPlaybooksEdgeCases:
    """Edge cases for _format_playbooks_with_prefix."""

    def test_empty_playbook_name_shows_unknown(self, mock_consolidator):
        """Test that empty playbook_name displays as 'unknown'."""
        fb = UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="req1",
            playbook_name="",
            content="content",
        )
        result = mock_consolidator._format_playbooks_with_prefix([fb], "NEW")
        assert "Name: unknown" in result

    def test_none_source_shows_unknown(self, mock_consolidator):
        """Test that None source displays as 'unknown'."""
        fb = UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="req1",
            playbook_name="fb",
            content="content",
            source=None,
        )
        result = mock_consolidator._format_playbooks_with_prefix([fb], "NEW")
        assert "Source: unknown" in result


class TestMockModeCheck:
    """Tests for mock mode check in deduplicate."""

    def test_mock_mode_handles_non_list_results(self, mock_consolidator):
        """Test that mock mode isinstance check filters non-list items."""
        fb = _make_user_playbook(0)

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "true"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1
        assert delete_ids == []

    def test_mock_mode_case_insensitive(self, mock_consolidator):
        """Test that mock mode check is case insensitive."""
        fb = _make_user_playbook(0)

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "True"}):
            result, delete_ids = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1
        assert delete_ids == []

    def test_mock_mode_false_proceeds_normally(self, mock_consolidator):
        """Test that mock mode disabled runs full dedup path."""
        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[IndependentDecision(new_id="NEW-0")],
            )
        )

        fb = _make_user_playbook(0)
        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _ = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1


class TestRetrieveExistingPlaybooksWithUserId:
    """Tests for _retrieve_existing_playbooks with user_id filter."""

    def test_user_id_passed_to_search(self, mock_consolidator):
        """Test that user_id is passed through to the search request."""
        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        existing_fb = _make_user_playbook(1, user_playbook_id=100)

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing_fb
        ]

        mock_consolidator._retrieve_existing_playbooks([new_fb], user_id="user_abc")

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

        mock_consolidator._retrieve_existing_playbooks([new_fb], user_id=None)

        call_args = (
            mock_consolidator.request_context.storage.search_user_playbooks.call_args
        )
        search_request = call_args[0][0]
        assert search_request.user_id is None
