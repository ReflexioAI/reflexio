"""Regression coverage for excluding rebuilding agent playbooks from default reads."""

from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.retriever_schema import (
    SearchAgentPlaybookRequest,
    UnifiedSearchRequest,
)
from reflexio.models.api_schema.service_schemas import (
    AgentPlaybook,
    PlaybookStatus,
    Status,
)
from reflexio.server.services.pre_retrieval import ReformulationResult
from reflexio.server.services.unified_search_service import run_unified_search

pytestmark = pytest.mark.integration


def _make_agent_playbook(
    playbook_id: int,
    *,
    playbook_name: str,
    content: str,
) -> AgentPlaybook:
    return AgentPlaybook(
        agent_playbook_id=playbook_id,
        playbook_name=playbook_name,
        agent_version="claude-code",
        content=content,
        created_at=1_700_000_000 + playbook_id,
        playbook_status=PlaybookStatus.APPROVED,
    )


def _mark_archive_in_progress(storage, agent_playbook_id: int) -> None:
    storage.conn.execute(
        "UPDATE agent_playbooks SET status = ? WHERE agent_playbook_id = ?",
        (Status.ARCHIVE_IN_PROGRESS.value, agent_playbook_id),
    )
    storage.conn.commit()


def test_rebuilding_agent_playbook_is_excluded_from_default_reads_and_search(
    storage,
) -> None:
    current = _make_agent_playbook(
        1,
        playbook_name="visible-rule",
        content="visible guidance for escalations only",
    )
    rebuilding = _make_agent_playbook(
        2,
        playbook_name="rebuild-rule",
        content="rtbf_rebuild_hidden_token guidance should stay hidden",
    )
    storage.save_agent_playbooks([current, rebuilding])
    _mark_archive_in_progress(storage, rebuilding.agent_playbook_id)

    assert storage.get_agent_playbook_by_id(rebuilding.agent_playbook_id) is None

    listed = storage.get_agent_playbooks()
    assert [playbook.agent_playbook_id for playbook in listed] == [
        current.agent_playbook_id
    ]

    search_results = storage.search_agent_playbooks(
        SearchAgentPlaybookRequest(
            query="rtbf_rebuild_hidden_token",
            top_k=5,
        )
    )
    assert search_results == []


def test_rebuilding_agent_playbook_is_excluded_from_unified_search(storage) -> None:
    rebuilding = _make_agent_playbook(
        3,
        playbook_name="rebuild-only",
        content="rtbf_unified_hidden_token guidance",
    )
    storage.save_agent_playbooks([rebuilding])
    _mark_archive_in_progress(storage, rebuilding.agent_playbook_id)

    with patch(
        "reflexio.server.services.unified_search_service.QueryReformulator"
    ) as reformulator_cls:
        reformulator_cls.return_value.rewrite.return_value = ReformulationResult(
            standalone_query="rtbf_unified_hidden_token"
        )
        result = run_unified_search(
            request=UnifiedSearchRequest(
                query="rtbf_unified_hidden_token",
                entity_types=["agent_playbooks"],
                top_k=5,
            ),
            org_id="contract_test",
            storage=storage,
            llm_client=MagicMock(),
            prompt_manager=MagicMock(),
        )

    assert result.success is True
    assert result.agent_playbooks == []
