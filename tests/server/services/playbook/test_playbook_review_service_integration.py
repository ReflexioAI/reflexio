from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest

from reflexio.models.api_schema.domain import Interaction, Request, Status, UserPlaybook
from reflexio.models.api_schema.domain.entities import ReviewUserPlaybooksRequest
from reflexio.models.api_schema.domain.enums import UserActionType
from reflexio.server.services.playbook.components.reviewer import (
    CandidateEvidenceUnit,
    CandidateReviewDecision,
    CandidateRevision,
    PlaybookCandidateReviewOutput,
    PlaybookReviewOutcome,
)
from reflexio.server.services.playbook.review_service import UserPlaybookReviewService
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _storage(tmp_path) -> SQLiteStorage:
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(
            org_id="playbook-review-test",
            db_path=str(tmp_path / "reflexio.db"),
        )
    storage._get_embedding = Mock(return_value=[0.0] * 512)  # noqa: SLF001
    storage.llm_client.get_embeddings = Mock(return_value=[[0.0] * 512])
    return storage


def _seed_source_interactions(storage: SQLiteStorage, count: int) -> None:
    for index in range(1, count + 1):
        request_id = f"source-request-{index}"
        storage.add_request(
            Request(
                request_id=request_id,
                user_id="user-1",
                created_at=100 + index,
                source="test",
                agent_version="1.0",
                session_id="session-1",
            )
        )
        storage.add_user_interactions_bulk(
            "user-1",
            [
                Interaction(
                    interaction_id=index,
                    user_id="user-1",
                    request_id=request_id,
                    content=f"Grounded evidence {index}.",
                    role="user",
                    created_at=100 + index,
                    user_action=UserActionType.NONE,
                    user_action_description="",
                    embedding=[0.0] * 512,
                )
            ],
            embeddings_prepared=True,
        )


def _playbook(
    index: int,
    *,
    created_at: int,
    request_id: str = "generation-request",
) -> UserPlaybook:
    return UserPlaybook(
        user_id="user-1",
        agent_version="1.0",
        request_id=request_id,
        created_at=created_at,
        source="test",
        content=f"Original lesson {index}.",
        trigger=f"When condition {index} applies",
        rationale=f"Grounded evidence {index} supports this lesson.",
        source_interaction_ids=[index],
        source_span=f"Grounded evidence {index}.",
        reader_angle="correction",
    )


def _service(storage: SQLiteStorage) -> UserPlaybookReviewService:
    playbook_config = SimpleNamespace(
        window_size_override=10,
        stride_size_override=8,
        deduplication_config=None,
        extraction_definition_prompt="Extract reusable agent guidance.",
    )
    root_config = SimpleNamespace(
        user_playbook_extractor_config=playbook_config,
        window_size=10,
        stride_size=8,
        tool_can_use=None,
        api_key_config=None,
    )
    request_context = SimpleNamespace(
        storage=storage,
        org_id="playbook-review-test",
        configurator=SimpleNamespace(
            get_config=lambda: root_config,
            get_agent_context=lambda: "Test agent",
        ),
        prompt_manager=Mock(),
    )
    return UserPlaybookReviewService(
        request_context=cast(Any, request_context),
        llm_client=cast(Any, Mock()),
    )


def _request(*, report_only: bool, top_k: int = 10) -> ReviewUserPlaybooksRequest:
    return ReviewUserPlaybooksRequest(
        start_time=datetime.fromtimestamp(190, UTC),
        end_time=datetime.fromtimestamp(250, UTC),
        top_k=top_k,
        report_only=report_only,
    )


def _units(interaction_id: int) -> dict[str, list[CandidateEvidenceUnit]]:
    """The evidence units ``decide`` resolves and ``apply_decisions`` consumes."""
    return {
        "C1": [
            CandidateEvidenceUnit(
                evidence_id="C1-E1",
                turn_ref="T1",
                source_span=f"Grounded evidence {interaction_id}.",
                interaction_id=interaction_id,
            )
        ]
    }


def _accept_output(interaction_id: int = 1) -> PlaybookReviewOutcome:
    return PlaybookReviewOutcome(
        output=PlaybookCandidateReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    id="C1",
                    decision="accept",
                    reason_code="grounded_useful",
                    evidence_ids=["C1-E1"],
                )
            ]
        ),
        units_by_candidate=_units(interaction_id),
    )


def _edit_output(interaction_id: int = 1) -> PlaybookReviewOutcome:
    return PlaybookReviewOutcome(
        output=PlaybookCandidateReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    id="C1",
                    decision="revise",
                    reason_code="generic",
                    evidence_ids=["C1-E1"],
                    revision=CandidateRevision(
                        content="Narrow revised lesson.",
                        trigger="When the narrow condition applies",
                        rationale="The cited evidence supports the narrow lesson.",
                    ),
                )
            ]
        ),
        units_by_candidate=_units(interaction_id),
    )


def _reject_output(interaction_id: int = 1) -> PlaybookReviewOutcome:
    return PlaybookReviewOutcome(
        output=PlaybookCandidateReviewOutput(
            decisions=[
                CandidateReviewDecision(
                    id="C1",
                    decision="reject",
                    reason_code="unsupported_evidence",
                )
            ]
        ),
        units_by_candidate=_units(interaction_id),
    )


def _patch_review_dependencies(outputs: list[object]):
    return (
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.is_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.decide",
            side_effect=outputs,
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookConsolidator.retrieve_existing_playbooks",
            return_value=[],
        ),
    )


def test_report_mode_respects_window_and_top_k_without_writes(tmp_path) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 3)
    rows = [
        _playbook(1, created_at=200),
        _playbook(2, created_at=201),
        _playbook(3, created_at=202),
    ]
    storage.save_user_playbooks(rows)
    enabled, decide, existing = _patch_review_dependencies(
        [_accept_output(), _reject_output()]
    )

    with enabled, decide, existing:
        response = _service(storage).run(_request(report_only=True, top_k=2))

    assert response.success is True
    assert response.report_only is True
    assert [result.user_playbook_id for result in response.results] == [
        rows[2].user_playbook_id,
        rows[1].user_playbook_id,
    ]
    assert response.accepted_count == 1
    assert response.rejected_count == 1
    assert all(result.applied is False for result in response.results)
    assert len(storage.get_user_playbooks(status_filter=[None])) == 3
    assert storage.get_user_playbooks(status_filter=[Status.PENDING]) == []


def test_apply_mode_commits_each_accept_edit_and_reject_in_order(tmp_path) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 3)
    rows = [
        _playbook(1, created_at=202),
        _playbook(2, created_at=201),
        _playbook(3, created_at=200),
    ]
    storage.save_user_playbooks(rows)
    enabled, decide, existing = _patch_review_dependencies(
        # Reviewed newest-first, so the edit lands on the middle row, whose
        # cited evidence is source interaction 2.
        [_accept_output(1), _edit_output(2), _reject_output(3)]
    )

    with enabled, decide, existing:
        response = _service(storage).run(_request(report_only=False))

    assert response.success is True
    assert response.report_only is False
    assert (
        response.accepted_count,
        response.edited_count,
        response.rejected_count,
        response.skipped_count,
    ) == (1, 1, 1, 0)
    assert all(result.applied is True for result in response.results)
    assert response.results[0].successor_user_playbook_id is None
    successor_id = response.results[1].successor_user_playbook_id
    assert successor_id is not None
    assert response.results[2].successor_user_playbook_id is None

    # The replacement is CURRENT, not PENDING: an edit must not take the
    # guidance out of retrieval, which reads user playbooks with status IS NULL.
    current = storage.get_user_playbooks(status_filter=[None])
    assert {item.user_playbook_id for item in current} == {
        rows[0].user_playbook_id,
        successor_id,
    }
    assert storage.get_user_playbooks(status_filter=[Status.PENDING]) == []
    successor = next(item for item in current if item.user_playbook_id == successor_id)
    assert successor.content == "Narrow revised lesson."
    assert successor.source == "test"

    # The edited incumbent is SUPERSEDED and points at its replacement; only the
    # rejected row is ARCHIVED.
    archived = storage.get_user_playbooks(status_filter=[Status.ARCHIVED])
    assert [item.user_playbook_id for item in archived] == [rows[2].user_playbook_id]
    superseded = storage.get_user_playbooks(status_filter=[Status.SUPERSEDED])
    assert [item.user_playbook_id for item in superseded] == [rows[1].user_playbook_id]
    assert superseded[0].superseded_by == successor_id

    # The revise event is what durably attributes the edit to this run, so an
    # operator can reconstruct it from lineage when the response is lost.
    successor_events = storage.get_lineage_events(
        entity_type="user_playbook",
        entity_id=str(successor_id),
    )
    revise_events = [event for event in successor_events if event.op == "revise"]
    assert len(revise_events) == 1
    assert revise_events[0].actor == "playbook_review"
    assert revise_events[0].request_id == response.run_id


@pytest.mark.parametrize("report_only", [False, True])
def test_accepted_same_generation_playbook_is_visible_to_next_review(
    tmp_path,
    report_only: bool,
) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 2)
    rows = [
        _playbook(1, created_at=201, request_id="shared-generation"),
        _playbook(2, created_at=200, request_id="shared-generation"),
    ]
    storage.save_user_playbooks(rows)
    seen_existing_ids: list[list[int]] = []

    def retrieve_existing(*args, **kwargs) -> list[UserPlaybook]:
        return storage.get_user_playbooks(status_filter=[None])

    def decide(*args, **kwargs) -> PlaybookReviewOutcome:
        seen_existing_ids.append(
            [playbook.user_playbook_id for playbook in kwargs["existing_playbooks"]]
        )
        return _accept_output(1) if len(seen_existing_ids) == 1 else _reject_output(2)

    with (
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.is_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.decide",
            side_effect=decide,
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookConsolidator.retrieve_existing_playbooks",
            side_effect=retrieve_existing,
        ),
    ):
        response = _service(storage).run(_request(report_only=report_only))

    assert response.success is True
    assert seen_existing_ids == [
        [rows[1].user_playbook_id],
        [rows[0].user_playbook_id],
    ]
    assert [result.decision for result in response.results] == ["accept", "reject"]
    assert [result.applied for result in response.results] == [
        not report_only,
        not report_only,
    ]
    current_ids = [
        playbook.user_playbook_id
        for playbook in storage.get_user_playbooks(status_filter=[None])
    ]
    assert current_ids == (
        [rows[0].user_playbook_id, rows[1].user_playbook_id]
        if report_only
        else [rows[0].user_playbook_id]
    )


def test_apply_mode_keeps_prior_commits_when_later_archive_loses_race(
    tmp_path,
) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 3)
    rows = [
        _playbook(1, created_at=202),
        _playbook(2, created_at=201),
        _playbook(3, created_at=200),
    ]
    storage.save_user_playbooks(rows)
    # The third decision (reject) loses its archive race; the accept and the
    # edit committed before it must survive.
    storage.archive_user_playbook_by_id = Mock(return_value=False)  # type: ignore[method-assign]
    enabled, decide, existing = _patch_review_dependencies(
        # Reviewed newest-first, so the edit lands on the middle row, whose
        # cited evidence is source interaction 2.
        [_accept_output(1), _edit_output(2), _reject_output(3)]
    )

    with enabled, decide, existing:
        response = _service(storage).run(_request(report_only=False))

    assert response.success is False
    assert "2 decision(s) remain committed" in (response.msg or "")
    assert [result.applied for result in response.results] == [True, True, False]
    successor_id = response.results[1].successor_user_playbook_id
    assert successor_id is not None
    assert {
        item.user_playbook_id
        for item in storage.get_user_playbooks(status_filter=[None])
    } == {rows[0].user_playbook_id, rows[2].user_playbook_id, successor_id}
    assert storage.get_user_playbooks(status_filter=[Status.ARCHIVED]) == []
    assert {
        item.user_playbook_id
        for item in storage.get_user_playbooks(status_filter=[Status.SUPERSEDED])
    } == {rows[1].user_playbook_id}


def test_edit_decision_rolls_back_successor_when_its_supersede_loses_race(
    tmp_path,
) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 1)
    row = _playbook(1, created_at=200)
    storage.save_user_playbooks([row])
    storage.supersede_record = Mock(return_value=False)  # type: ignore[method-assign]
    enabled, decide, existing = _patch_review_dependencies([_edit_output()])

    with enabled, decide, existing:
        response = _service(storage).run(_request(report_only=False))

    assert response.success is False
    assert response.results[0].decision == "edit"
    assert response.results[0].applied is False
    assert [
        item.user_playbook_id
        for item in storage.get_user_playbooks(status_filter=[None])
    ] == [row.user_playbook_id]
    assert storage.get_user_playbooks(status_filter=[Status.ARCHIVED]) == []
    assert storage.get_user_playbooks(status_filter=[Status.PENDING]) == []
    assert len(storage.get_lineage_events(entity_type="user_playbook")) == 1


def test_review_failure_keeps_decisions_committed_before_the_failure(tmp_path) -> None:
    storage = _storage(tmp_path)
    _seed_source_interactions(storage, 2)
    rows = [
        _playbook(1, created_at=201),
        _playbook(2, created_at=200),
    ]
    storage.save_user_playbooks(rows)

    with (
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.is_enabled",
            return_value=True,
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookCandidateReviewer.decide",
            side_effect=[_reject_output(), RuntimeError("review failed")],
        ),
        patch(
            "reflexio.server.services.playbook.review_service."
            "PlaybookConsolidator.retrieve_existing_playbooks",
            return_value=[],
        ),
    ):
        response = _service(storage).run(_request(report_only=False))

    assert response.success is False
    assert response.selected_count == 2
    assert "Reviewed 1 of 2 selected playbook(s)" in (response.msg or "")
    assert "1 decision(s) remain committed" in (response.msg or "")
    assert response.results[0].decision == "reject"
    assert response.results[0].applied is True
    assert [
        item.user_playbook_id
        for item in storage.get_user_playbooks(status_filter=[None])
    ] == [rows[1].user_playbook_id]
    assert [
        item.user_playbook_id
        for item in storage.get_user_playbooks(status_filter=[Status.ARCHIVED])
    ] == [rows[0].user_playbook_id]
    assert storage.get_user_playbooks(status_filter=[Status.PENDING]) == []
