"""Tests for playbook consolidation service.

Exercises the dispatch-and-apply path of the ``PlaybookConsolidationOutput``
discriminated union (4 kinds: ``unify``, ``reject_new``, ``differentiate``,
``independent``). Storage-end-to-end coverage of all kinds lives in
``test_playbook_consolidator_integration.py``.
"""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.llm.litellm_client import (
    LiteLLMClientError,
    StructuredOutputRepairError,
)
from reflexio.server.services.playbook.components.consolidator import (
    DifferentiateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidationResult,
    PlaybookConsolidator,
    RejectNewDecision,
    UnifyDecision,
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


def _unify(
    new_id: str | list[str],
    archive_existing_ids: list[int] | None = None,
    *,
    content: str = "unified content",
    trigger: str = "unified trigger",
    rationale: str = "unified rationale",
) -> UnifyDecision:
    """Build a ``UnifyDecision`` with sane defaults for the apply tests.

    Polarity is not a decision field; the unified row's orientation is derived
    from ``content`` / ``rationale`` wording at apply time. The defaults derive
    positive (no avoidance framing).
    """
    return UnifyDecision(
        new_id=new_id,
        archive_existing_ids=archive_existing_ids or [],
        content=content,
        trigger=trigger,
        rationale=rationale,
    )


def _shared_repair_side_effect(*outputs: PlaybookConsolidationOutput):
    """Simulate the shared client validator/repair contract for service tests."""

    provenance = ModelProvenance(
        model_name="served-model",
        provider="provider",
    )

    def _side_effect(*, response_format, structured_output_validator, model, **_kwargs):
        assert response_format is PlaybookConsolidationOutput
        first_output = outputs[0]
        first_errors = structured_output_validator(first_output)
        if not first_errors:
            return first_output
        if len(outputs) > 1:
            repaired_output = outputs[1]
            repaired_errors = structured_output_validator(repaired_output)
            if not repaired_errors:
                return repaired_output
            raise StructuredOutputRepairError(
                "repair exhausted",
                failure_kind="semantic",
                model=model,
                parsed_output=repaired_output,
                validation_errors=tuple(repaired_errors),
                first_parsed_provenance=provenance,
            )
        raise StructuredOutputRepairError(
            "repair exhausted",
            failure_kind="semantic",
            model=model,
            parsed_output=first_output,
            validation_errors=tuple(first_errors),
            first_parsed_provenance=provenance,
        )

    return _side_effect


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

    def test_row_exposes_trigger_and_rationale(self, mock_consolidator):
        """Rendered rows MUST expose `Trigger` and `Rationale` alongside `Content`.

        Several decision kinds compare existing-vs-new triggers (``differentiate``,
        same-situation contradictions, trigger refinements). Without the trigger
        field in the prompt payload the model is guessing about the field it is
        supposed to refine. This regression test pins the row shape.
        """
        # Under Option B there is no derived ``Polarity`` field in the row: the
        # LLM reads orientation directly from the content/rationale wording.
        fb = UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="req1",
            playbook_name="fb",
            content="Do not start unbilled work when Y",
            trigger="user asks about billing",
            rationale="prevents unbilled work after user pushback",
            source="extractor",
        )
        result = mock_consolidator._format_playbooks_with_prefix([fb], "EXISTING")
        assert 'Trigger: "user asks about billing"' in result, result
        assert 'Rationale: "prevents unbilled work after user pushback"' in result, (
            result
        )
        # Content / name / source must still render alongside. The derived
        # ``Polarity`` field was removed under Option B.
        assert 'Content: "Do not start unbilled work when Y"' in result
        assert "Polarity:" not in result
        assert "Name: fb" in result
        assert "Source: extractor" in result

    def test_row_handles_missing_trigger_and_rationale(self, mock_consolidator):
        """``trigger`` and ``rationale`` are nullable on UserPlaybook; the
        formatter must render an empty string in their slot rather than
        ``None`` so the prompt stays well-formed."""
        fb = UserPlaybook(
            user_playbook_id=0,
            agent_version="v1",
            request_id="req1",
            playbook_name="fb",
            content="content",
            trigger=None,
            rationale=None,
        )
        result = mock_consolidator._format_playbooks_with_prefix([fb], "NEW")
        assert 'Trigger: ""' in result, result
        assert 'Rationale: ""' in result, result
        assert "None" not in result, "literal None must not leak into the prompt"


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
        """Test retrieval embeds queries via the storage's own embedding path."""
        new_fb = _make_user_playbook(0, trigger="user asks about billing")
        existing_fb = _make_user_playbook(
            1, user_playbook_id=100, trigger="billing inquiry"
        )

        storage = mock_consolidator.request_context.storage
        storage._get_embedding.return_value = [0.1, 0.2, 0.3]
        storage.search_user_playbooks.return_value = [existing_fb]

        result = mock_consolidator._retrieve_existing_playbooks([new_fb])

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
        mock_consolidator._retrieve_existing_playbooks([new_fb])

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
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb1], [fb2]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 2
        assert delete_ids == []

    def test_empty_results(self, mock_consolidator):
        """Test deduplication with no playbooks."""
        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
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
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1
        assert delete_ids == []


# ===============================
# Tests for _build_deduplicated_results
# ===============================


class TestBuildDeduplicatedResults:
    """Tests for ``_build_deduplicated_results`` decision dispatch."""

    def test_unify_pair_replacement(self, mock_consolidator):
        """Pair replacement (was ``prefer_new``): one NEW + one EXISTING archived.

        Verifies that ``unify`` with a single archived EXISTING produces one
        inserted row carrying the LLM-supplied content and the existing id is
        added to the archive list.
        """
        new_playbooks = [
            _make_user_playbook(0, content="new content"),
        ]
        existing_playbooks = [
            _make_user_playbook(1, user_playbook_id=500, content="old content"),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    archive_existing_ids=[0],
                    content="final content",
                )
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert result[0].content == "final content"
        assert delete_ids == [500]

    def test_unify_n_way_merge(self, mock_consolidator):
        """N-way merge (was ``duplicate``): one NEW + multiple EXISTING archived.

        Verifies that ``unify`` collapses one candidate plus several existing
        rows into a single inserted row, archiving every referenced EXISTING id.
        """
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[10]),
        ]
        existing_playbooks = [
            _make_user_playbook(
                1,
                user_playbook_id=501,
                source_interaction_ids=[1],
            ),
            _make_user_playbook(
                2,
                user_playbook_id=502,
                source_interaction_ids=[2],
            ),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    archive_existing_ids=[0, 1],
                    content="merged content",
                )
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert result[0].content == "merged content"
        # Both existing rows archived.
        assert set(delete_ids) == {501, 502}
        # Source interaction ids combine NEW + every EXISTING member.
        assert set(result[0].source_interaction_ids) == {10, 1, 2}

    def test_unify_insert_without_archive(self, mock_consolidator):
        """Insert-without-archive: one NEW + empty archive list.

        The storage layer allows ``unify`` with no archived rows (it produces
        a single insert and zero archives). The prompt steers the LLM away
        from this shape — ``independent`` is the right kind here — but the
        apply path must support it without raising.
        """
        new_playbooks = [
            _make_user_playbook(0, content="solo new"),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    archive_existing_ids=[],
                    content="solo final",
                )
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 1
        assert result[0].content == "solo final"
        assert delete_ids == []

    def test_unify_over_budget_logs_warning(self, mock_consolidator, caplog):
        """An over-budget unify logs a backstop warning but still applies.

        The complexity budget is a soft signal: the merge proceeds (the row is
        inserted), and ``event=consolidation_over_budget`` is emitted with the
        offending length and the configured budget.
        """
        mock_consolidator._dedup_config.max_unified_content_chars = 10
        over_budget_content = "x" * 50

        new_playbooks = [_make_user_playbook(0, content="new content")]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify("NEW-0", archive_existing_ids=[], content=over_budget_content)
            ],
        )

        with caplog.at_level("WARNING"):
            result, _, _ = mock_consolidator._build_deduplicated_results(
                new_playbooks=new_playbooks,
                existing_playbooks=[],
                dedup_output=dedup_output,
                request_id="req1",
                agent_version="v1",
            )

        # Merge still applies (soft backstop, not a blocker).
        assert len(result) == 1
        assert result[0].content == over_budget_content
        assert any(
            "event=consolidation_over_budget" in rec.message
            and "len=50" in rec.message
            and "budget=10" in rec.message
            for rec in caplog.records
        )

    def test_unify_within_budget_no_warning(self, mock_consolidator, caplog):
        """A within-budget unify emits no over-budget warning."""
        mock_consolidator._dedup_config.max_unified_content_chars = 1000

        new_playbooks = [_make_user_playbook(0, content="new content")]
        dedup_output = PlaybookConsolidationOutput(
            decisions=[_unify("NEW-0", archive_existing_ids=[], content="short")],
        )

        with caplog.at_level("WARNING"):
            mock_consolidator._build_deduplicated_results(
                new_playbooks=new_playbooks,
                existing_playbooks=[],
                dedup_output=dedup_output,
                request_id="req1",
                agent_version="v1",
            )

        assert not any(
            "event=consolidation_over_budget" in rec.message for rec in caplog.records
        )

    def test_unify_counter_bumps_once_per_decision(self, mock_consolidator):
        """``unify_count`` increments by exactly one per applied ``UnifyDecision``,
        regardless of archive cardinality (0, 1, or N).
        """
        result_counters = PlaybookConsolidationResult()
        for kind in ["unify", "unify", "unify"]:
            mock_consolidator._bump_counter(result_counters, kind)
        assert result_counters.unify_count == 3

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

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 2

    def test_build_deduplicated_results_stamps_generation_request_id(
        self, mock_consolidator
    ):
        """Newly synthesized rows should carry the generation provenance id."""
        new_playbooks = [
            _make_user_playbook(0, content="Use the deployment checklist."),
        ]
        dedup_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(kind="independent", new_id="NEW-0")]
        )

        result, delete_ids, merge_groups = (
            mock_consolidator._build_deduplicated_results(
                new_playbooks=new_playbooks,
                existing_playbooks=[],
                dedup_output=dedup_output,
                generation_request_id="rerun_playbook_ab12cd34",
                agent_version="v1",
            )
        )

        assert delete_ids == []
        assert merge_groups == []
        assert [row.request_id for row in result] == ["rerun_playbook_ab12cd34"]

    def test_build_deduplicated_results_accepts_legacy_request_id_keyword(
        self, mock_consolidator
    ):
        """The helper should keep the legacy ``request_id=`` alias working."""
        new_playbooks = [
            _make_user_playbook(0, content="Use the deployment checklist."),
        ]
        dedup_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(kind="independent", new_id="NEW-0")]
        )

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="legacy_req_1",
            agent_version="v1",
        )

        assert [row.request_id for row in result] == ["legacy_req_1"]

    def test_independent_decision_accepts_multiple_new_ids(self, mock_consolidator):
        """A list-valued ``new_id`` inserts each named NEW candidate once."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
            _make_user_playbook(2),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                IndependentDecision(new_id=["[NEW-0]", "NEW-2"]),  # type: ignore[arg-type]
            ],
        )

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        contents = {fb.content for fb in result}
        assert contents == {"content_0", "content_2", "content_1"}

    def test_unify_accepts_bracketed_new_id_from_llm(self, mock_consolidator):
        """Regression for Sentry: ``[NEW-0]`` normalizes before apply lookup."""
        new_playbooks = [_make_user_playbook(0, source_interaction_ids=[10])]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify("[NEW-0]", content="merged", trigger="merged trigger"),
            ],
        )

        result, delete_ids, merge_groups = (
            mock_consolidator._build_deduplicated_results(
                new_playbooks=new_playbooks,
                existing_playbooks=[],
                dedup_output=dedup_output,
                request_id="req1",
                agent_version="v1",
            )
        )

        assert len(result) == 1
        assert result[0].content == "merged"
        assert result[0].trigger == "merged trigger"
        assert result[0].source_interaction_ids == [10]
        assert delete_ids == []
        assert merge_groups == []

    def test_unify_stamps_generation_request_id(self, mock_consolidator):
        """``unify`` should stamp synthesized rows with the provenance id."""
        new_playbooks = [_make_user_playbook(0, source_interaction_ids=[10])]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify("NEW-0", content="merged", trigger="merged trigger"),
            ],
        )

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            generation_request_id="rerun_playbook_unify_1",
            agent_version="v1",
        )

        assert [row.request_id for row in result] == ["rerun_playbook_unify_1"]

    def test_unify_accepts_multiple_new_ids(self, mock_consolidator):
        """Multi-NEW ``unify`` emits one synthesized row and consumes all NEW ids."""
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[10]),
            _make_user_playbook(1, source_interaction_ids=[11]),
            _make_user_playbook(2, source_interaction_ids=[12]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    ["[NEW-0]", "NEW-1"],
                    content="merged",
                    trigger="merged trigger",
                ),
            ],
        )

        result, delete_ids, merge_groups = (
            mock_consolidator._build_deduplicated_results(
                new_playbooks=new_playbooks,
                existing_playbooks=[],
                dedup_output=dedup_output,
                request_id="req1",
                agent_version="v1",
            )
        )

        assert len(result) == 2
        assert result[0].content == "merged"
        assert result[0].source_interaction_ids == [10, 11]
        assert result[1].content == "content_2"
        assert delete_ids == []
        assert merge_groups == []

    def test_reject_new_no_storage_changes(self, mock_consolidator):
        """``RejectNewDecision`` produces no inserts and no archives."""
        new_playbooks = [_make_user_playbook(0)]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=999),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # No insert from reject_new; safety fallback does not re-insert because
        # the candidate was consumed by the decision.
        assert result == []
        assert delete_ids == []

    def test_reject_new_resolves_existing_position(self, mock_consolidator):
        """``reject_new`` ids may refer to the rendered ``EXISTING-N`` position.

        The consolidation prompt shows rows as ``[EXISTING-0]`` etc. If the LLM
        emits ``0`` for that first row, the apply path must consume the NEW
        candidate rather than treating the id as invalid and triggering the
        safety fallback.
        """
        new_playbooks = [_make_user_playbook(0)]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                RejectNewDecision(new_id="NEW-0", superseded_by_existing_id=0),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert result == []
        assert delete_ids == []

    def test_reject_new_accepts_multiple_new_ids_and_existing_refs(
        self, mock_consolidator
    ):
        """Multi-NEW ``reject_new`` consumes all named NEWs when refs resolve."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
            _make_user_playbook(2),
        ]
        existing_playbooks = [
            _make_user_playbook(3, user_playbook_id=998),
            _make_user_playbook(4, user_playbook_id=999),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                RejectNewDecision(
                    new_id=["[NEW-0]", "NEW-2"],  # type: ignore[arg-type]
                    superseded_by_existing_id=["[EXISTING-0]", "EXISTING-1"],  # type: ignore[arg-type]
                ),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert [fb.content for fb in result] == ["content_1"]
        assert delete_ids == []

    def test_reject_new_partial_existing_refs_reinserts_all(self, mock_consolidator):
        """If ANY superseding ref is unknown, the decision is malformed: no NEW
        id is consumed, so the safety fallback re-inserts every candidate."""
        new_playbooks = [
            _make_user_playbook(0),
            _make_user_playbook(1),
            _make_user_playbook(2),
        ]
        existing_playbooks = [_make_user_playbook(3, user_playbook_id=998)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                RejectNewDecision(
                    new_id=["NEW-0", "NEW-2"],  # type: ignore[arg-type]
                    # EXISTING-0 resolves; EXISTING-5 has no row/id -> unknown.
                    superseded_by_existing_id=["EXISTING-0", "EXISTING-5"],  # type: ignore[arg-type]
                ),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert {fb.content for fb in result} == {"content_0", "content_1", "content_2"}
        assert delete_ids == []

    def test_differentiate_resolves_existing_position(self, mock_consolidator):
        """``differentiate`` ids may refer to the rendered ``EXISTING-N`` position."""
        new_playbooks = [_make_user_playbook(0, trigger="broad trigger")]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                DifferentiateDecision(
                    new_id="NEW-0",
                    existing_id=0,
                    refined_new_trigger="narrow new trigger",
                    refined_existing_trigger="narrow existing trigger",
                ),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 2
        assert result[0].trigger == "narrow new trigger"
        assert result[1].trigger == "narrow existing trigger"
        assert delete_ids == [999]

    def test_differentiate_falls_back_to_db_id(self, mock_consolidator):
        """When the id misses every ``EXISTING-N`` position, the apply path
        falls back to a DB ``user_playbook_id`` for older prompt outputs.

        Here the single existing row sits at position 0 but carries DB id 999.
        Emitting ``999`` misses ``EXISTING-999`` and must resolve via the DB-id
        fallback, archiving the right row.
        """
        new_playbooks = [_make_user_playbook(0, trigger="broad trigger")]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                DifferentiateDecision(
                    new_id="NEW-0",
                    existing_id=999,
                    refined_new_trigger="narrow new trigger",
                    refined_existing_trigger="narrow existing trigger",
                ),
            ],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        assert len(result) == 2
        assert delete_ids == [999]

    def test_differentiate_stamps_generation_request_id(self, mock_consolidator):
        """``differentiate`` should stamp both synthesized rows with provenance."""
        new_playbooks = [_make_user_playbook(0, trigger="broad trigger")]
        existing_playbooks = [_make_user_playbook(1, user_playbook_id=999)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[
                DifferentiateDecision(
                    new_id="NEW-0",
                    existing_id=0,
                    refined_new_trigger="narrow new trigger",
                    refined_existing_trigger="narrow existing trigger",
                ),
            ],
        )

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
            dedup_output=dedup_output,
            generation_request_id="rerun_playbook_diff_1",
            agent_version="v1",
        )

        assert [row.request_id for row in result] == [
            "rerun_playbook_diff_1",
            "rerun_playbook_diff_1",
        ]

    def test_existing_reference_ignores_out_of_range_position_key(
        self, mock_consolidator
    ):
        """Out-of-range position-like keys must not shadow legacy DB ids."""
        db_row = _make_user_playbook(1, user_playbook_id=999)
        stray_position_row = _make_user_playbook(2, user_playbook_id=123)

        resolved = mock_consolidator._resolve_existing_reference(
            999,
            existing_by_position={
                "EXISTING-0": db_row,
                "EXISTING-999": stray_position_row,
            },
            existing_by_id={999: db_row},
        )

        assert resolved is db_row

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

        result, _, _ = mock_consolidator._build_deduplicated_results(
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

    def test_deduplicate_stamps_generation_request_id(self, mock_consolidator):
        """The main entry point should propagate ``generation_request_id``."""
        fb0 = _make_user_playbook(0, content="playbook from extractor 1")

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[IndependentDecision(new_id="NEW-0")],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb0]],
                generation_request_id="rerun_playbook_dedup_1",
                agent_version="v1",
            )

        assert [row.request_id for row in result] == ["rerun_playbook_dedup_1"]
        assert delete_ids == []

    def test_deduplicate_accepts_legacy_request_id_keyword(self, mock_consolidator):
        """The main entry point should keep the legacy ``request_id=`` alias."""
        fb0 = _make_user_playbook(0, content="playbook from extractor 1")

        mock_consolidator.client.get_embeddings.return_value = [[0.1]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[IndependentDecision(new_id="NEW-0")],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb0]],
                request_id="legacy_req_dedup_1",
                agent_version="v1",
            )

        assert [row.request_id for row in result] == ["legacy_req_dedup_1"]
        assert delete_ids == []

    def test_happy_path_with_unify(self, mock_consolidator):
        """Full happy path: LLM returns a ``unify`` decision and an ``independent``.

        ``unify`` collapses one NEW into one merged row; the other NEW flows
        through as ``independent``. Combined the batch produces two rows.
        """
        fb0 = _make_user_playbook(0, content="do X when Y", source_interaction_ids=[10])
        fb1 = _make_user_playbook(2, content="do Z when W", source_interaction_ids=[30])

        # No existing playbooks found via search
        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []

        # LLM unifies fb0 (no archives), keeps fb1 as independent
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        archive_existing_ids=[],
                        content="do X",
                        trigger="when Y",
                    ),
                    IndependentDecision(new_id="NEW-1"),
                ],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb0], [fb1]], request_id="req_test", agent_version="v1"
            )

        # 1 unified + 1 independent = 2 playbooks
        assert len(result) == 2
        assert delete_ids == []
        # Independent playbook should be fb1
        assert any(r.content == "do Z when W" for r in result)
        # Unified playbook should carry the LLM-supplied final content
        assert any(r.content == "do X" for r in result)

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
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb0], [fb1], [fb2]], request_id="req_test", agent_version="v1"
            )

        assert len(result) == 3
        assert delete_ids == []

    def test_unify_against_existing_archives_the_existing(self, mock_consolidator):
        """A NEW unified with an EXISTING entry archives the existing row.

        Carries forward the legacy "all playbooks are duplicates of existing"
        scenario under the new ``unify`` shape.
        """
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

        # LLM unifies NEW-0 with EXISTING-0
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        archive_existing_ids=[0],
                        content="do X",
                        trigger="when Y",
                    ),
                ],
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb0]], request_id="req_test", agent_version="v1"
            )

        # 1 unified playbook replaces both
        assert len(result) == 1
        # Existing playbook should be marked for deletion
        assert 500 in delete_ids
        # Unified playbook should combine source_interaction_ids from both
        assert set(result[0].source_interaction_ids) == {5, 10}


class TestConsolidationRepair:
    """Tests for pre-apply validation and the single repair pass."""

    def test_repair_fallback_uses_first_parsed_output_provenance(
        self, mock_consolidator
    ):
        first_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-0")]
        )
        first_parsed_provenance = ModelProvenance(model_name="served-first-parsed")

        def repair_exhausted(*, structured_output_validator, **_kwargs):
            structured_output_validator(first_output)
            raise StructuredOutputRepairError(
                "repair exhausted",
                failure_kind="semantic",
                model="configured-model",
                first_parsed_provenance=first_parsed_provenance,
            )

        mock_consolidator.client.generate_chat_response.side_effect = repair_exhausted

        result = mock_consolidator._consolidation_decisions(
            [_make_user_playbook(0), _make_user_playbook(1)], []
        )

        assert result is first_output
        assert mock_consolidator.model_provenance == first_parsed_provenance

    def test_repair_fallback_returns_first_parsed_output_without_provenance(
        self, mock_consolidator
    ):
        first_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-0")]
        )

        def repair_exhausted(*, structured_output_validator, **_kwargs):
            structured_output_validator(first_output)
            raise StructuredOutputRepairError(
                "repair exhausted",
                failure_kind="semantic",
                model="configured-model",
            )

        mock_consolidator.client.generate_chat_response.side_effect = repair_exhausted

        result = mock_consolidator._consolidation_decisions(
            [_make_user_playbook(0), _make_user_playbook(1)], []
        )

        assert result is first_output
        assert mock_consolidator.model_provenance is None

    def test_transport_failure_preserves_first_parsed_output(self, mock_consolidator):
        first_output = PlaybookConsolidationOutput(
            decisions=[IndependentDecision(new_id="NEW-0")]
        )
        first_parsed_provenance = ModelProvenance(model_name="served-first-parsed")

        def transport_failure(*, structured_output_validator, **_kwargs):
            structured_output_validator(first_output)
            raise LiteLLMClientError(
                "repair transport failed",
                first_parsed_provenance=first_parsed_provenance,
            )

        mock_consolidator.client.generate_chat_response.side_effect = transport_failure

        result = mock_consolidator._consolidation_decisions(
            [_make_user_playbook(0), _make_user_playbook(1)], []
        )

        assert result is first_output
        assert mock_consolidator.model_provenance == first_parsed_provenance

    def test_under_consumed_output_repairs_to_multi_new_unify(self, mock_consolidator):
        new_0 = _make_user_playbook(
            0, content="alpha beta", source_interaction_ids=[10]
        )
        new_1 = _make_user_playbook(
            1, content="gamma delta", source_interaction_ids=[11]
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        bad_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    content="alpha beta gamma delta",
                    trigger="when combined",
                )
            ]
        )
        mock_consolidator.client.generate_chat_response.side_effect = (
            _shared_repair_side_effect(
                bad_output,
                PlaybookConsolidationOutput(
                    decisions=[
                        _unify(
                            ["NEW-0", "NEW-1"],
                            content="alpha beta gamma delta",
                            trigger="when combined",
                        )
                    ]
                ),
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, delete_ids, merge_groups = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_repair",
                agent_version="v1",
            )

        assert len(result) == 1
        assert result[0].content == "alpha beta gamma delta"
        assert result[0].source_interaction_ids == [10, 11]
        assert delete_ids == []
        assert merge_groups == []
        assert mock_consolidator.client.generate_chat_response.call_count == 1
        validator = mock_consolidator.client.generate_chat_response.call_args.kwargs[
            "structured_output_validator"
        ]
        assert any("missing NEW ids: NEW-1" in error for error in validator(bad_output))

    def test_repair_failure_falls_back_to_original_output(
        self, mock_consolidator, caplog
    ):
        new_0 = _make_user_playbook(
            0, content="alpha beta", source_interaction_ids=[10]
        )
        new_1 = _make_user_playbook(
            1, content="gamma delta", source_interaction_ids=[11]
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        original_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    content="alpha beta gamma delta",
                    trigger="when combined",
                )
            ]
        )
        mock_consolidator.client.generate_chat_response.side_effect = (
            _shared_repair_side_effect(
                original_output,
                PlaybookConsolidationOutput(
                    decisions=[IndependentDecision(new_id="NEW-0")]
                ),
            )
        )

        with (
            patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
            caplog.at_level("WARNING"),
        ):
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_repair_fail",
                agent_version="v1",
            )

        assert [row.content for row in result] == [
            "alpha beta gamma delta",
            "gamma delta",
        ]
        assert delete_ids == []
        assert mock_consolidator.client.generate_chat_response.call_count == 1
        assert mock_consolidator.model_provenance is not None
        assert mock_consolidator.model_provenance.model_name == "served-model"

    def test_suspicious_same_source_split_triggers_repair(self, mock_consolidator):
        new_0 = _make_user_playbook(
            0,
            content="check target group security group health check",
            source_interaction_ids=[15, 16],
        )
        new_1 = _make_user_playbook(
            1,
            content="check target group security group health check",
            source_interaction_ids=[15, 16],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        bad_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    content="check target group security group health check",
                    trigger="when deploying",
                ),
                IndependentDecision(new_id="NEW-1"),
            ]
        )
        mock_consolidator.client.generate_chat_response.side_effect = (
            _shared_repair_side_effect(
                bad_output,
                PlaybookConsolidationOutput(
                    decisions=[
                        _unify(
                            ["NEW-0", "NEW-1"],
                            content="check target group security group health check",
                            trigger="when deploying",
                        )
                    ]
                ),
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_suspicious",
                agent_version="v1",
            )

        assert len(result) == 1
        assert result[0].source_interaction_ids == [15, 16]
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_low_overlap_same_source_split_does_not_trigger_repair(
        self, mock_consolidator
    ):
        new_0 = _make_user_playbook(
            0,
            content="check target group health checks",
            source_interaction_ids=[15],
        )
        new_1 = _make_user_playbook(
            1,
            content="avoid friday afternoon deploys",
            source_interaction_ids=[15],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        content="check target group health checks",
                        trigger="when deploying",
                    ),
                    IndependentDecision(new_id="NEW-1"),
                ]
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_low_overlap",
                agent_version="v1",
            )

        assert len(result) == 2
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_reject_new_consumed_overlap_does_not_trigger_repair(
        self, mock_consolidator
    ):
        """A high-overlap same-source row consumed by ``reject_new`` is dropped,
        so it cannot duplicate the unify survivor — no repair call is spent."""
        new_0 = _make_user_playbook(
            0,
            content="check target group security group health check",
            source_interaction_ids=[15, 16],
        )
        new_1 = _make_user_playbook(
            1,
            content="check target group security group health check",
            source_interaction_ids=[15, 16],
        )
        existing = _make_user_playbook(
            50,
            content="existing deploy checklist",
            source_interaction_ids=[1],
            user_playbook_id=50,
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = [
            existing
        ]
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        content="check target group security group health check",
                        trigger="when deploying",
                    ),
                    RejectNewDecision(new_id="NEW-1", superseded_by_existing_id=0),
                ]
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_reject_new_overlap",
                agent_version="v1",
            )

        assert len(result) == 1
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_sibling_unify_same_source_overlap_triggers_repair(self, mock_consolidator):
        """Two same-source unify decisions with overlapping FINAL contents each
        persist a survivor row — the same duplicate-storage class as an
        under-consumed unify — so the pair must trigger the repair pass."""
        new_0 = _make_user_playbook(
            0,
            content="check target group security group health check",
            source_interaction_ids=[15, 16],
        )
        new_1 = _make_user_playbook(
            1,
            content="verify security group and health check wiring",
            source_interaction_ids=[15, 16],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.side_effect = _shared_repair_side_effect(
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        content="check target group security group health check wiring",
                        trigger="when deploying",
                    ),
                    _unify(
                        "NEW-1",
                        content="check target group security group health check wiring",
                        trigger="when deploying",
                    ),
                ]
            ),
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        ["NEW-0", "NEW-1"],
                        content="check target group security group health check wiring",
                        trigger="when deploying",
                    )
                ]
            ),
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_sibling_unify",
                agent_version="v1",
            )

        assert len(result) == 1
        assert result[0].source_interaction_ids == [15, 16]
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_distinct_sibling_unifies_do_not_trigger_repair(self, mock_consolidator):
        """Same-source unify pair with genuinely different final contents is a
        legitimate split — no repair call is spent."""
        new_0 = _make_user_playbook(
            0,
            content="check target group health checks before deploying",
            source_interaction_ids=[15],
        )
        new_1 = _make_user_playbook(
            1,
            content="announce the rollout in the deploy channel first",
            source_interaction_ids=[15],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.return_value = (
            PlaybookConsolidationOutput(
                decisions=[
                    _unify(
                        "NEW-0",
                        content="check target group health checks before deploying",
                        trigger="when deploying",
                    ),
                    _unify(
                        "NEW-1",
                        content="announce the rollout in the deploy channel first",
                        trigger="when announcing a rollout",
                    ),
                ]
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_distinct_unifies",
                agent_version="v1",
            )

        assert len(result) == 2
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_suspicious_repair_returning_same_decisions_is_accepted(
        self, mock_consolidator
    ):
        new_0 = _make_user_playbook(
            0,
            content="sync env var secret port references",
            source_interaction_ids=[20],
        )
        new_1 = _make_user_playbook(
            1,
            content="sync env var secret port references",
            source_interaction_ids=[20],
        )
        same_output = PlaybookConsolidationOutput(
            decisions=[
                _unify(
                    "NEW-0",
                    content="sync env var secret port references",
                    trigger="when editing docs",
                ),
                IndependentDecision(new_id="NEW-1"),
            ]
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.side_effect = (
            _shared_repair_side_effect(same_output, same_output)
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[new_0, new_1]],
                request_id="req_same_repair",
                agent_version="v1",
            )

        assert len(result) == 2
        assert mock_consolidator.client.generate_chat_response.call_count == 1

    def test_same_batch_duplicate_regression_drops_raw_source_after_repair(
        self, mock_consolidator
    ):
        merged_source = _make_user_playbook(
            0,
            content="Always update target groups and security groups.",
            source_interaction_ids=[15, 16, 19, 20],
        )
        raw_source = _make_user_playbook(
            1,
            content="Always update security groups.",
            source_interaction_ids=[15, 16, 19, 20],
        )

        mock_consolidator.client.get_embeddings.return_value = [[0.1], [0.2]]
        mock_consolidator.request_context.storage.search_user_playbooks.return_value = []
        mock_consolidator.client.generate_chat_response.side_effect = (
            _shared_repair_side_effect(
                PlaybookConsolidationOutput(
                    decisions=[
                        _unify(
                            "NEW-0",
                            content="Always update target groups and security groups.",
                            trigger="when changing AWS deploy wiring",
                        )
                    ]
                ),
                PlaybookConsolidationOutput(
                    decisions=[
                        _unify(
                            ["NEW-0", "NEW-1"],
                            content="Always update target groups and security groups.",
                            trigger="when changing AWS deploy wiring",
                        )
                    ]
                ),
            )
        )

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}):
            result, _, _ = mock_consolidator.deduplicate(
                results=[[merged_source, raw_source]],
                request_id="req_duplicate_regression",
                agent_version="v1",
            )

        assert [row.content for row in result] == [
            "Always update target groups and security groups."
        ]


# ===============================
# Edge cases for _build_deduplicated_results
# ===============================


class TestBuildDeduplicatedResultsEdgeCases:
    """Extended tests for _build_deduplicated_results edge cases."""

    def test_unify_with_unknown_existing_position_fails_apply(self, mock_consolidator):
        """``unify`` referencing an EXISTING-{idx} that doesn't exist is a soft failure.

        Per-decision try/except isolates the failure; safety fallback still
        re-inserts the NEW candidate as-is so the data is not silently lost.
        """
        new_playbooks = [_make_user_playbook(0)]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[_unify("NEW-0", archive_existing_ids=[99])],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=[],
            dedup_output=dedup_output,
            request_id="req1",
            agent_version="v1",
        )

        # Decision failed, but safety fallback adds NEW-0 as-is
        assert len(result) == 1
        assert delete_ids == []

    def test_source_interaction_ids_combined_from_new_and_existing(
        self, mock_consolidator
    ):
        """Source interaction ids combine across NEW + EXISTING unify members."""
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[1, 2]),
        ]
        existing_playbooks = [
            _make_user_playbook(1, user_playbook_id=100, source_interaction_ids=[3, 4]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[_unify("NEW-0", archive_existing_ids=[0], content="merged")],
        )

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
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
        """Duplicate source interaction ids across NEW + EXISTING are not repeated.

        Under the 4-kind redesign, ``unify`` archives only EXISTING rows (it
        requires exactly one NEW), so this test now combines the candidate's
        ids with an EXISTING row that shares one id.
        """
        new_playbooks = [
            _make_user_playbook(0, source_interaction_ids=[1, 2]),
        ]
        existing_playbooks = [
            _make_user_playbook(1, user_playbook_id=200, source_interaction_ids=[2, 3]),
        ]

        dedup_output = PlaybookConsolidationOutput(
            decisions=[_unify("NEW-0", archive_existing_ids=[0], content="merged")],
        )

        result, _, _ = mock_consolidator._build_deduplicated_results(
            new_playbooks=new_playbooks,
            existing_playbooks=existing_playbooks,
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

        result, _, _ = mock_consolidator._build_deduplicated_results(
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

        result, delete_ids, _ = mock_consolidator._build_deduplicated_results(
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
            result, delete_ids, _ = mock_consolidator.deduplicate(
                results=[[fb]], request_id="req1", agent_version="v1"
            )

        assert len(result) == 1
        assert delete_ids == []

    def test_mock_mode_case_insensitive(self, mock_consolidator):
        """Test that mock mode check is case insensitive."""
        fb = _make_user_playbook(0)

        with patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "True"}):
            result, delete_ids, _ = mock_consolidator.deduplicate(
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
            result, _, _ = mock_consolidator.deduplicate(
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
