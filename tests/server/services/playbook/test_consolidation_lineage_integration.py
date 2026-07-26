"""Integration test: consolidation merges route through atomic ``merge_records``.

Drives the generation service's ``_finalize_extracted_items`` apply path against
a real ``SQLiteStorage`` with the consolidation LLM step mocked to return ONE
``unify`` decision ("merge new NEW-0 with existing EXISTING-0"). Asserts the
lineage-aware outcome (Task 10):

* the existing source becomes a ``MERGED`` tombstone with ``merged_into`` ->
  survivor (not hard-deleted as in the legacy save-then-delete path),
* a ``merge`` lineage event keyed on the survivor exists, and
* ``resolve_current("user_playbook", old_id)`` returns the survivor id.

Mirrors the real-SQLite + mocked-LLM fixture style of
``test_playbook_consolidator_integration.py``.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import CompletionResult, ModelProvenance
from reflexio.server.services.lineage.resolve import resolve_current
from reflexio.server.services.playbook.components.consolidator import (
    DifferentiateDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
    UnifyDecision,
)
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def temp_storage_dir():
    """Per-test temp directory for SQLite isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sqlite_storage(temp_storage_dir, worker_id):
    """Real SQLite storage in a per-test temp dir + per-worker org id."""
    return SQLiteStorage(
        org_id=f"test-consolidation-lineage-{worker_id}",
        db_path=os.path.join(temp_storage_dir, "consolidation_lineage.db"),
    )


@pytest.fixture
def request_context(sqlite_storage, temp_storage_dir, worker_id):
    """RequestContext wired to real SQLite storage + a mocked prompt manager."""
    context = RequestContext(
        org_id=f"test-consolidation-lineage-{worker_id}",
        storage_base_dir=temp_storage_dir,
    )
    context.storage = sqlite_storage
    context.prompt_manager = MagicMock()
    context.prompt_manager.render_prompt.return_value = "mock prompt"
    context.configurator = MagicMock()
    return context


@pytest.fixture
def generation_service(request_context):
    """A PlaybookGenerationService with optimization/aggregation side effects stubbed."""
    service = PlaybookGenerationService(
        llm_client=MagicMock(), request_context=request_context
    )
    service.service_config = PlaybookGenerationServiceConfig(
        request_id="req_merge",
        agent_version="v0",
        user_id="u1",
        source="chat",
    )
    # Keep the test focused on the merge routing — stub the downstream side
    # effects that need a fully-wired configurator / scheduler.
    service._enqueue_user_playbook_optimization = MagicMock()  # type: ignore[method-assign]
    service._trigger_playbook_aggregation = MagicMock()  # type: ignore[method-assign]
    return service


def _seed_existing(storage: SQLiteStorage) -> UserPlaybook:
    """Insert one existing CURRENT user playbook and return the persisted row."""
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version="v0",
        request_id="r0",
        playbook_name="default",
        content="Recommend X.",
        trigger="when Y",
        rationale="r",
        status=None,
        source="chat",
        source_interaction_ids=[],
    )
    storage.save_user_playbooks([pb])
    saved = storage.get_user_playbooks(user_id="u1")
    assert len(saved) == 1
    return saved[0]


def _candidate() -> UserPlaybook:
    """Build a NEW (unpersisted) candidate user playbook."""
    return UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version="v0",
        request_id="req_merge",
        playbook_name="default",
        content="Recommend X (canonical).",
        trigger="when Y",
        rationale="r",
        status=None,
        source="chat",
        source_interaction_ids=[],
    )


def _candidate_with_content(
    *,
    content: str,
    request_id: str = "req_merge",
    source_interaction_ids: list[int] | None = None,
) -> UserPlaybook:
    return UserPlaybook(
        user_playbook_id=0,
        user_id="u1",
        agent_version="v0",
        request_id=request_id,
        playbook_name="default",
        content=content,
        trigger="when changing AWS deploy wiring",
        rationale="r",
        status=None,
        source="chat",
        source_interaction_ids=source_interaction_ids or [],
    )


def test_consolidation_merge_routes_through_merge_records(
    sqlite_storage, generation_service
):
    """One ``unify`` decision tombstones the existing source into the survivor.

    The legacy path hard-deleted the source (so it would be ``None`` even with
    ``include_tombstones=True``). The lineage-aware path leaves a MERGED
    tombstone whose ``merged_into`` points at the freshly-saved survivor, emits
    a ``merge`` lineage event, and ``resolve_current`` follows the pointer.
    """
    existing = _seed_existing(sqlite_storage)
    old_id = existing.user_playbook_id

    decision_output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[0],
                content="Recommend X (canonical).",
                trigger="when Y",
                rationale="merged",
            )
        ]
    )

    observed_provenance = ModelProvenance(
        model_name="MiniMax-M3",
        provider="minimax",
    )
    real_deduplicate = PlaybookConsolidator.deduplicate

    def _deduplicate_with_observed_provenance(self, *args, **kwargs):
        result = real_deduplicate(self, *args, **kwargs)
        # Stamp observed consolidator attribution so the merge event path is
        # exercised without a live LLM completion.
        self.model_provenance = observed_provenance
        return result

    with (
        patch.object(
            PlaybookGenerationService,
            "_configured_playbook_config",
            return_value=None,
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._retrieve_existing_playbooks",
            return_value=[existing],
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._consolidation_decisions",
            return_value=decision_output,
        ),
        patch.object(
            PlaybookConsolidator,
            "deduplicate",
            _deduplicate_with_observed_provenance,
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
    ):
        generation_service._finalize_extracted_items([_candidate()])

    # The survivor is the single surviving CURRENT row.
    current = sqlite_storage.get_user_playbooks(user_id="u1")
    assert len(current) == 1, [p.content for p in current]
    survivor = current[0]
    assert survivor.content == "Recommend X (canonical)."
    assert survivor.user_playbook_id != old_id

    # The old source is a MERGED tombstone pointing at the survivor — NOT
    # hard-deleted (the legacy path would make this None).
    tombstone = sqlite_storage.get_user_playbook_by_id(old_id, include_tombstones=True)
    assert tombstone is not None, "source was hard-deleted, not tombstoned"
    assert tombstone.status == Status.MERGED
    assert tombstone.merged_into == survivor.user_playbook_id

    # A merge lineage event keyed on the survivor exists, with consolidator model.
    events = sqlite_storage.get_lineage_events(
        entity_type="user_playbook", entity_id=str(survivor.user_playbook_id)
    )
    merge_events = [e for e in events if e.op == "merge"]
    assert len(merge_events) == 1, events
    assert str(old_id) in merge_events[0].source_ids
    assert merge_events[0].model_name == observed_provenance.model_name
    assert merge_events[0].provider == observed_provenance.provider

    # resolve_current follows merged_into to the live survivor.
    ref = resolve_current(sqlite_storage, "user_playbook", old_id)
    assert ref is not None
    assert ref.id == str(survivor.user_playbook_id)


def test_consolidation_repair_persists_only_repaired_multi_new_unify(
    sqlite_storage, generation_service
):
    """Repair an under-consumed same-batch merge before the service saves rows.

    This drives the production finalization boundary with real SQLite storage:
    the initial consolidation output merges facts from two NEW candidates but
    lists only ``NEW-0``; the repair output lists both. The persisted state must
    contain the merged survivor and not the raw ``NEW-1`` fallback row.
    """
    candidate_0 = _candidate_with_content(
        content="Always update target groups and security groups.",
        source_interaction_ids=[15, 16, 19, 20],
    )
    candidate_1 = _candidate_with_content(
        content="Always update security groups.",
        source_interaction_ids=[15, 16, 19, 20],
    )
    initial_output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[],
                content="Always update target groups and security groups.",
                trigger="when changing AWS deploy wiring",
                rationale="merged",
            )
        ]
    )
    repaired_output = PlaybookConsolidationOutput(
        decisions=[
            UnifyDecision(
                new_id=["NEW-0", "NEW-1"],
                archive_existing_ids=[],
                content="Always update target groups and security groups.",
                trigger="when changing AWS deploy wiring",
                rationale="merged",
            )
        ]
    )

    def repaired_consolidation(*, structured_output_validator, **_kwargs):
        assert structured_output_validator(initial_output)
        assert structured_output_validator(repaired_output) == []
        return CompletionResult(repaired_output, ModelProvenance())

    generation_service.client.generate_chat_response_with_provenance.side_effect = (
        repaired_consolidation
    )

    with (
        patch.object(
            PlaybookGenerationService,
            "_configured_playbook_config",
            return_value=None,
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._retrieve_existing_playbooks",
            return_value=[],
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
    ):
        generation_service._finalize_extracted_items([candidate_0, candidate_1])

    current = sqlite_storage.get_user_playbooks(user_id="u1")
    assert len(current) == 1, [p.content for p in current]
    survivor = current[0]
    assert survivor.content == "Always update target groups and security groups."
    assert survivor.source_interaction_ids == [15, 16, 19, 20]
    generation_service.client.generate_chat_response_with_provenance.assert_called_once()


def test_consolidation_differentiate_tombstones_split_source(
    sqlite_storage, generation_service
):
    """A differentiate split must preserve the old source as a tombstone.

    The legacy leftover-delete path hard-deletes the original row, which makes
    point-in-time attribution impossible because there is no dead source record
    left to read. The split source must survive as a tombstone instead.
    """
    existing = _seed_existing(sqlite_storage)
    old_id = existing.user_playbook_id

    decision_output = PlaybookConsolidationOutput(
        decisions=[
            DifferentiateDecision(
                new_id="NEW-0",
                existing_id=0,
                refined_new_trigger="when user asks for canonical guidance",
                refined_existing_trigger="when user asks for quick advice",
                reason="same advice, different situations",
            )
        ]
    )

    with (
        patch.object(
            PlaybookGenerationService,
            "_configured_playbook_config",
            return_value=None,
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._retrieve_existing_playbooks",
            return_value=[existing],
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._consolidation_decisions",
            return_value=decision_output,
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
    ):
        generation_service._finalize_extracted_items([_candidate()])

    tombstone = sqlite_storage.get_user_playbook_by_id(old_id, include_tombstones=True)
    assert tombstone is not None, "differentiate split source was hard-deleted"
    assert tombstone.status == Status.SUPERSEDED
    assert tombstone.content == "Recommend X."


def test_consolidation_differentiate_propagates_tombstone_failure(
    sqlite_storage, generation_service
):
    """A failed split-source tombstone must abort before aggregation."""
    existing = _seed_existing(sqlite_storage)
    decision_output = PlaybookConsolidationOutput(
        decisions=[
            DifferentiateDecision(
                new_id="NEW-0",
                existing_id=0,
                refined_new_trigger="when user asks for canonical guidance",
                refined_existing_trigger="when user asks for quick advice",
                reason="same advice, different situations",
            )
        ]
    )

    with (
        patch.object(
            PlaybookGenerationService,
            "_configured_playbook_config",
            return_value=None,
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._retrieve_existing_playbooks",
            return_value=[existing],
        ),
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator._consolidation_decisions",
            return_value=decision_output,
        ),
        patch.object(
            sqlite_storage,
            "supersede_user_playbooks_by_ids",
            side_effect=RuntimeError("tombstone failed"),
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
        pytest.raises(RuntimeError, match="tombstone failed"),
    ):
        generation_service._finalize_extracted_items([_candidate()])

    generation_service._trigger_playbook_aggregation.assert_not_called()
