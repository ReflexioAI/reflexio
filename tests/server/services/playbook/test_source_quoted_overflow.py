"""Overflow publication cannot invent or rewrite original instructions."""

import copy

import pytest

from reflexio.models.api_schema.internal_schema import RequestInteractionDataModel
from reflexio.models.api_schema.service_schemas import Interaction, Request
from reflexio.server.services.extraction.input_preparation import (
    prepare_extraction_input,
)
from reflexio.server.services.extraction.protected_record import (
    EvidenceKind,
    ProtectedEvidence,
    ProtectedRecord,
)


def sources(content="Keep STAGING_APPROVAL_REQUIRED. Inspection is not execution."):
    return [
        RequestInteractionDataModel(
            request=Request(
                request_id="r", user_id="u", session_id="s", agent_version="v"
            ),
            session_id="s",
            arrival_order=True,
            interactions=[
                Interaction(
                    interaction_id=1,
                    request_id="r",
                    user_id="u",
                    role="User",
                    content=content,
                ),
                Interaction(
                    interaction_id=2,
                    request_id="r",
                    user_id="u",
                    role="Assistant",
                    content="User corrected an earlier deployment mistake.",
                ),
            ],
        )
    ]


def select(
    prepared, identity, *, kind: EvidenceKind = "instruction", start=None, end=None
):
    span = next(
        s for c in prepared.chunks for s in c.spans if s.interaction_id == identity
    )
    if start is not None:
        from dataclasses import replace

        span = replace(span, start=start, end=end, text=span.text[start:end])
    return ProtectedEvidence(
        kind=kind, evidence_refs=(span.evidence_ref,), quote=span.text
    )


def test_publication_copies_complete_source_not_selected_fragment_or_model_history():
    from reflexio.server.services.playbook.source_quoted_overflow import (
        project_source_messages,
    )

    groups = sources(
        "Do not deploy. Only exact environment approval permits it. STAGING_APPROVAL_REQUIRED."
    )
    before = copy.deepcopy(groups)
    prepared = prepare_extraction_input(groups, user_id="u")
    # A selection excluding 'Do not' must never turn the copied instruction into permission.
    record = ProtectedRecord(
        evidence=(select(prepared, 1, start=7, end=13), select(prepared, 2))
    )
    projected = project_source_messages(prepared, groups, record, user_id="u")
    assert len(projected) == 1
    assert projected[0].content == groups[0].interactions[0].content
    assert projected[0].interaction_id == 1 and projected[0].request_id == "r"
    assert groups == before


def test_duplicate_selections_copy_once_and_oversized_messages_are_not_truncated():
    from reflexio.server.services.playbook.source_quoted_overflow import (
        project_source_messages,
    )

    groups = sources("Instruction " * 9000)
    prepared = prepare_extraction_input(groups, user_id="u")
    evidence = select(prepared, 1)
    assert (
        project_source_messages(
            prepared,
            groups,
            ProtectedRecord(evidence=(evidence, evidence)),
            user_id="u",
        )
        == []
    )
    groups = sources()
    prepared = prepare_extraction_input(groups, user_id="u")
    evidence = select(prepared, 1)
    assert (
        len(
            project_source_messages(
                prepared,
                groups,
                ProtectedRecord(evidence=(evidence, evidence)),
                user_id="u",
            )
        )
        == 1
    )


def test_changed_sources_and_unresolvable_selection_fail_closed():
    from reflexio.server.services.playbook.source_quoted_overflow import (
        project_source_messages,
    )

    groups = sources()
    prepared = prepare_extraction_input(groups, user_id="u")
    record = ProtectedRecord(evidence=(select(prepared, 1),))
    groups[0].interactions[0].content = "Deploy without permission."
    with pytest.raises(ValueError):
        project_source_messages(prepared, groups, record, user_id="u")
    with pytest.raises(ValueError):
        project_source_messages(
            prepared,
            sources(),
            ProtectedRecord(
                evidence=(
                    ProtectedEvidence(
                        kind="instruction",
                        evidence_refs=("i999:content:0:5",),
                        quote="hello",
                    ),
                )
            ),
            user_id="u",
        )


@pytest.mark.parametrize("kind", ["action", "outcome", "uncertainty"])
def test_non_instruction_selections_do_not_become_rules(kind):
    from reflexio.server.services.playbook.source_quoted_overflow import (
        project_source_messages,
    )

    groups = sources()
    prepared = prepare_extraction_input(groups, user_id="u")
    assert (
        project_source_messages(
            prepared,
            groups,
            ProtectedRecord(evidence=(select(prepared, 1, kind=kind),)),
            user_id="u",
        )
        == []
    )


def test_publication_budget_counts_all_text_copies_and_keeps_later_small_message():
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    import tiktoken

    from reflexio.server.services.extraction.overflow_coordinator import (
        OverflowCoordinator,
    )
    from reflexio.server.services.playbook.source_quoted_overflow import (
        PUBLICATION_TOKENS,
        _projection,
        _row_witness,
    )

    groups = sources("Keep the approval gate. " + "context " * 5000)
    groups[0].interactions.append(
        Interaction(
            interaction_id=3,
            request_id="r",
            user_id="u",
            role="User",
            content="Do not deploy without approval.",
        )
    )
    prepared = prepare_extraction_input(groups, user_id="u", chunk_tokens=1500)
    assert len(prepared.chunks) > 1
    record = ProtectedRecord(
        evidence=(select(prepared, 1, start=0, end=23), select(prepared, 3))
    )
    manifest = OverflowCoordinator.initial_manifest(prepared.source_digest)
    manifest["checkpoint"].update(
        protected_record=record.model_dump(mode="json"),
        protected_coverage=[
            OverflowCoordinator.candidate_key(c.text) for c in prepared.chunks
        ],
    )
    coordinator = OverflowCoordinator(
        Mock(org_id="org"),
        Mock(kind="playbook", user_id="u", window_id="w", project_id="p"),
        "token",
        manifest,
        source_digest=prepared.source_digest,
        context_limits={},
        prepared=prepared,
    )
    service = Mock(
        service_config=SimpleNamespace(
            user_id="u",
            agent_version="v",
            request_id="r",
            source="test",
            window_interactions=groups,
        )
    )
    rows, _ = _projection(coordinator, service)
    assert len(rows) == 1 and rows[0].source_interaction_ids == [3]
    assert rows[0].content == groups[0].interactions[-1].content
    packed = json.dumps(
        [_row_witness(row) for row in rows], ensure_ascii=False, separators=(",", ":")
    )
    assert (
        len(tiktoken.get_encoding("cl100k_base").encode(packed)) <= PUBLICATION_TOKENS
    )


def test_tool_field_selection_cannot_select_unrelated_user_content():
    from reflexio.models.api_schema.common import ToolUsed
    from reflexio.server.services.playbook.source_quoted_overflow import (
        project_source_messages,
    )

    groups = sources("Unrelated user message.")
    groups[0].interactions[0].tools_used = [
        ToolUsed(
            tool_name="inspect", tool_data={"output": "Do not deploy without approval."}
        )
    ]
    prepared = prepare_extraction_input(groups, user_id="u")
    span = next(
        s
        for c in prepared.chunks
        for s in c.spans
        if s.interaction_id == 1 and s.field != "content"
    )
    record = ProtectedRecord(
        evidence=(
            ProtectedEvidence(
                kind="instruction", evidence_refs=(span.evidence_ref,), quote=span.text
            ),
        )
    )
    assert project_source_messages(prepared, groups, record, user_id="u") == []


def test_postcommit_classification_tags_do_not_rewrite_source_witness():
    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.server.services.playbook.source_quoted_overflow import _row_witness

    row = UserPlaybook(
        user_id="u", agent_version="v", request_id="r", content="Do not deploy."
    )
    before = _row_witness(row)
    row.tags = ["deployment"]
    assert _row_witness(row) == before
    row.expanded_terms = "production execution"
    assert _row_witness(row) != before


def test_vector_receipt_uses_storage_precision_but_rejects_changed_search_vector():
    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.services.playbook.source_quoted_overflow import (
        _embedding_digest,
    )

    row = UserPlaybook.model_construct(
        embedding=[0.10000000149011612] * EMBEDDING_DIMENSIONS
    )
    expected = _embedding_digest(row)
    row.embedding = [0.1] * EMBEDDING_DIMENSIONS  # pgvector text round trip.
    assert _embedding_digest(row) == expected
    row.embedding[0] = 0.9
    assert _embedding_digest(row) != expected


@pytest.mark.parametrize("component", [True, float("nan"), float("inf"), 1e300])
def test_invalid_source_projection_vectors_cannot_be_sealed(component):
    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.models.config_schema import EMBEDDING_DIMENSIONS
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.playbook.source_quoted_overflow import (
        _embedding_digest,
    )

    row = UserPlaybook.model_construct(embedding=[component] * EMBEDDING_DIMENSIONS)
    with pytest.raises(InputBudgetConfigurationError):
        _embedding_digest(row)


def test_source_quoted_playbooks_are_not_automatically_enqueued_for_optimization():
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock, patch

    from reflexio.models.api_schema.service_schemas import UserPlaybook
    from reflexio.models.config_schema import PlaybookOptimizerConfig
    from reflexio.server.services.playbook.service import PlaybookGenerationService
    from reflexio.server.services.playbook.source_quoted_overflow import POLICY
    from reflexio.server.services.playbook_optimizer import (
        PlaybookOptimizationScheduler,
    )

    service = object.__new__(PlaybookGenerationService)
    service.configurator = Mock(
        get_config=lambda: SimpleNamespace(
            playbook_optimizer_config=PlaybookOptimizerConfig(
                enabled=True,
                user_playbooks_enabled=True,
            )
        )
    )
    row = UserPlaybook(
        user_playbook_id=3,
        user_id="u",
        agent_version="v",
        request_id="r",
        content="Keep the approval gate.",
        source_interaction_ids=[1],
        notes=json.dumps({"policy": POLICY}),
    )
    with patch.object(PlaybookOptimizationScheduler, "get_instance") as scheduler:
        service._enqueue_user_playbook_optimization([row])
    scheduler.return_value.enqueue.assert_not_called()


@pytest.mark.parametrize(
    "notes",
    [
        None,
        "plain notes",
        '{"policy":"other"}',
        '{"policy":"source-quoted-playbooks-v1"}',
        ' {"policy":"source-quoted-playbooks-v1","details":"keep all text"}',
    ],
)
def test_ordinary_notes_reserve_policy_without_losing_text_or_growing(notes):
    from reflexio.models.api_schema.domain.entities import UserPlaybook
    from reflexio.server.services.playbook.source_quoted_overflow import (
        is_source_quoted_playbook,
        ordinary_playbook_notes,
    )

    preserved = ordinary_playbook_notes(notes)
    row = UserPlaybook(agent_version="v", request_id="r", notes=preserved)
    assert not is_source_quoted_playbook(row)
    assert ordinary_playbook_notes(preserved) == preserved
    if notes is not None:
        assert preserved is not None
        assert preserved.endswith(notes)
