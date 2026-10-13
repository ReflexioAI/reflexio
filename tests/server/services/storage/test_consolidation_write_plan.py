"""Audited retirement/replacement writes obey the existing commit fence."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import LineageContext, UserPlaybook
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm._litellm_types import ModelProvenance
from reflexio.server.services.deferred_learning_plan import PlaybookWritePlan
from reflexio.server.services.playbook.service import (
    PlaybookGenerationService,
    PlaybookGenerationServiceConfig,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def service(storage, tmp_path):
    context = RequestContext(org_id=storage.org_id, storage_base_dir=str(tmp_path))
    context.storage = storage
    instance = PlaybookGenerationService(MagicMock(), context)
    instance.service_config = PlaybookGenerationServiceConfig(
        request_id="reconcile-request", agent_version="v1", user_id="owner"
    )
    return instance


def _row(content: str) -> UserPlaybook:
    return UserPlaybook(
        user_id="owner",
        agent_version="v1",
        request_id="reconcile-request",
        content=content,
        trigger="When reconciling advice",
    )


def _plan(rows: list[UserPlaybook], old_ids: list[int]) -> PlaybookWritePlan:
    return PlaybookWritePlan(
        request_id="reconcile-request",
        output_pending_status=False,
        skip_aggregation=True,
        new_playbooks=rows,
        superseded_ids=old_ids,
        merge_groups=[],
        lineage_contexts=[
            LineageContext(
                op_kind="create",
                actor="consolidator",
                source_ids=[str(value) for value in old_ids],
                request_id="reconcile-request",
                model_name="audited-model",
                provider="scripted",
            )
            for _ in rows
        ],
    )


@pytest.mark.parametrize("rollback", [False, True])
@pytest.mark.parametrize("output_count", [0, 2])
def test_retirement_and_all_replacement_lineage_share_commit_fence(
    service, storage, rollback, output_count
):
    old = [_row("Old advice one"), _row("Old advice two")]
    storage.save_user_playbooks(old, skip_embedding=True)
    old_ids = [row.user_playbook_id for row in old]
    rows = [_row(f"Scoped replacement {index}") for index in range(output_count)]
    plan = _plan(rows, old_ids)

    def persist():
        with storage.commit_scope():
            service._persist_write_plan(plan)
            if rollback:
                raise RuntimeError("fence lost")

    if rollback:
        with pytest.raises(RuntimeError, match="fence lost"):
            persist()
    else:
        persist()

    for old_id in old_ids:
        incumbent = storage.get_user_playbook_by_id(old_id, include_tombstones=True)
        assert incumbent is not None
        assert incumbent.status is (None if rollback else Status.SUPERSEDED)
        assert incumbent.merged_into is None
        assert incumbent.content in {"Old advice one", "Old advice two"}
    current = storage.get_user_playbooks(user_id="owner")
    assert len(current) == (2 if rollback else output_count)
    events = storage.get_lineage_events(request_id="reconcile-request")
    output_events = [event for event in events if event.actor == "consolidator"]
    if rollback:
        assert output_events == []
    else:
        for row in rows:
            creates = [
                event
                for event in output_events
                if event.entity_id == str(row.user_playbook_id)
            ]
            assert len(creates) == 1
            assert creates[0].source_ids == [str(value) for value in old_ids]
            assert creates[0].model_name == "audited-model"
        assert not any(event.op == "merge" for event in events)


def test_compute_preserves_removal_only_plan_without_embedding(service, storage):
    incumbent = _row("Obsolete advice")
    storage.save_user_playbooks([incumbent], skip_embedding=True)
    with (
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator"
        ) as cls,
        patch.object(storage, "precompute_user_playbook_embeddings") as embed,
    ):
        cls.return_value.deduplicate.return_value = (
            [],
            [incumbent.user_playbook_id],
            [],
        )
        cls.return_value.model_provenance = ModelProvenance()
        cls.return_value.consolidated_output_indices = set()
        plan = service._resolve_write_plan([[_row("New invalid advice")]])
    assert plan is not None
    assert plan.new_playbooks == []
    assert plan.superseded_ids == [incumbent.user_playbook_id]
    embed.assert_not_called()
    assert storage.get_user_playbook_by_id(incumbent.user_playbook_id) is not None


@pytest.mark.parametrize("status", [None, Status.PENDING, Status.ARCHIVED])
def test_compute_carries_each_replacement_source_into_create_lineage(
    service, storage, status
):
    rows = [_row("Scope one"), _row("Scope two")]
    service.service_config.source = "service-source-B"
    for index, row in enumerate(rows):
        row.status = status
        row.source = "evidence-source-A" if index == 0 else None
        row.source_interaction_ids = [40 + index]
        row.source_span = f"source-span-{index}"
    with (
        patch(
            "reflexio.server.services.playbook.components.consolidator.PlaybookConsolidator"
        ) as cls,
        patch.object(storage, "precompute_user_playbook_embeddings"),
    ):
        cls.return_value.deduplicate.return_value = (rows, [10, 20], [])
        cls.return_value.model_provenance = ModelProvenance(model_name="audited-model")
        cls.return_value.consolidated_output_indices = {0, 1}
        cls.return_value.lineage_sources_by_output_index = {
            0: ["10", "20"],
            1: ["10", "20"],
        }
        plan = service._resolve_write_plan([[_row("Candidate")]])
    assert plan is not None
    assert [context.source_ids for context in plan.lineage_contexts] == [
        ["10", "20"],
        ["10", "20"],
    ]
    assert all(context.actor == "consolidator" for context in plan.lineage_contexts)
    assert plan.merge_groups == []
    assert [row.status for row in plan.new_playbooks] == [status, status]
    assert [row.source for row in plan.new_playbooks] == ["evidence-source-A", None]
    assert [row.source_interaction_ids for row in plan.new_playbooks] == [[40], [41]]
    assert [row.source_span for row in plan.new_playbooks] == [
        "source-span-0",
        "source-span-1",
    ]
