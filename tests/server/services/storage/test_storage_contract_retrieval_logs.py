"""Contract tests for playbook retrieval-log scaffolding in OSS storage."""

import pytest

from reflexio.models.api_schema.domain import (
    PlaybookRetrievalLog,
    PlaybookRetrievalLogItem,
)
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.integration


def test_retrieval_log_models_expose_header_and_item_defaults() -> None:
    item = PlaybookRetrievalLogItem(ordinal=0, agent_playbook_id=101)
    log = PlaybookRetrievalLog(request_id="req-1", session_id="sess-1", user_id="u1")

    assert item.retrieval_log_item_id == 0
    assert item.retrieval_log_id == 0
    assert item.source_user_playbook_ids == []
    assert item.source_interaction_ids_by_user_playbook_id == {}

    assert log.retrieval_log_id == 0
    assert log.interaction_id is None
    assert log.query is None
    assert log.agent_version is None
    assert log.shown_items == []
    assert log.created_at == 0


def test_sqlite_retrieval_log_methods_remain_unimplemented(
    storage: BaseStorage,
) -> None:
    log = PlaybookRetrievalLog(
        request_id="req-1",
        session_id="sess-1",
        user_id="u1",
        shown_items=[
            PlaybookRetrievalLogItem(
                ordinal=0,
                agent_playbook_id=101,
                source_user_playbook_ids=[11],
                source_interaction_ids_by_user_playbook_id={"11": [201]},
            )
        ],
    )

    with pytest.raises(NotImplementedError):
        storage.save_playbook_retrieval_log(log)

    with pytest.raises(NotImplementedError):
        storage.get_playbook_retrieval_logs(user_id="u1")
