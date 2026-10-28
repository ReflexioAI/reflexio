"""Source provenance follows its parents; it is not evictable operational history."""

from unittest.mock import patch

import pytest

from reflexio.models.api_schema.retriever_schema import SearchAgentPlaybookRequest
from reflexio.models.api_schema.service_schemas import AgentPlaybook, UserPlaybook
from reflexio.server.services.storage import retention
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

_SOURCE_TABLE = "agent_playbook_source_user_playbooks"
_RETIRED_CAP = "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOK_SOURCE_USER_PLAYBOOKS"


@pytest.fixture
def graph(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-provenance-fixture")
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        store = SQLiteStorage(
            org_id="owned-provenance", db_path=str(tmp_path / "owned.db")
        )
        try:
            store.save_user_playbooks(
                [
                    UserPlaybook(
                        user_id="u1",
                        playbook_name="pb",
                        agent_version="v1",
                        request_id=f"r{i}",
                        content=f"source {i}",
                        trigger="trigger",
                        created_at=i,
                        source="test",
                        source_interaction_ids=[],
                    )
                    for i in (1, 2)
                ]
            )
            user_ids = [
                int(row[0])
                for row in store.conn.execute(
                    "SELECT user_playbook_id FROM user_playbooks ORDER BY user_playbook_id"
                ).fetchall()
            ]
            store.save_agent_playbooks(
                [
                    AgentPlaybook(
                        playbook_name="pb",
                        agent_version="v1",
                        content="aggregated guidance",
                        created_at=3,
                    )
                ]
            )
            agent_id = int(
                store.conn.execute(
                    "SELECT agent_playbook_id FROM agent_playbooks LIMIT 1"
                ).fetchone()[0]
            )
            store.set_source_user_playbook_ids_for_agent_playbook(agent_id, user_ids)
            yield store, agent_id, user_ids
        finally:
            store.conn.close()


def test_source_links_cannot_be_independently_evicted_from_live_playbooks(graph):
    store, agent_id, user_ids = graph
    request = SearchAgentPlaybookRequest(source="test", top_k=10)
    assert [
        item.agent_playbook_id for item in store.search_agent_playbooks(request)
    ] == [agent_id]
    with pytest.raises(ValueError, match="Unknown retention target"):
        store.delete_oldest_retention_target_rows(_SOURCE_TABLE, 2)
    assert store.get_source_user_playbook_ids_for_agent_playbook(agent_id) == user_ids
    assert [
        item.agent_playbook_id for item in store.search_agent_playbooks(request)
    ] == [agent_id]
    assert _SOURCE_TABLE not in retention.get_row_retention_limits(env={})


def test_agent_parent_retention_removes_links_but_preserves_source_playbooks(graph):
    store, agent_id, user_ids = graph
    assert store.delete_oldest_retention_target_rows("agent_playbooks", 1) == 1
    assert store.get_agent_playbook_by_id(agent_id) is None
    assert store.get_source_user_playbook_ids_for_agent_playbook(agent_id) == []
    assert all(store.get_user_playbook_by_id(key) is not None for key in user_ids)


def test_source_parent_retention_keeps_other_live_provenance(graph):
    store, agent_id, user_ids = graph
    assert store.delete_oldest_retention_target_rows("user_playbooks", 1) == 1
    assert store.get_agent_playbook_by_id(agent_id) is not None
    assert (
        store.get_source_user_playbook_ids_for_agent_playbook(agent_id) == user_ids[1:]
    )
    assert (
        len(store.search_agent_playbooks(SearchAgentPlaybookRequest(source="test")))
        == 1
    )


@pytest.mark.parametrize("old", ["", "0", "1", "private-operator-value"])
def test_old_cap_is_rejected_without_reinterpreting_parent_policy(old):
    with pytest.raises(ValueError, match=_RETIRED_CAP) as exc:
        retention.get_row_retention_limits(
            env={_RETIRED_CAP: old, "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOKS": "0"}
        )
    message = str(exc.value)
    assert "private-operator-value" not in message
    assert f"{_RETIRED_CAP} ->" not in message
    assert "do not copy its value" in message
    assert "Migrate the effective policy to the canonical setting" not in message


def test_parent_and_operational_policies_keep_their_original_values():
    limits = retention.get_row_retention_limits(
        env={
            "REFLEXIO_ROW_LIMIT_AGENT_PLAYBOOKS": "0",
            "REFLEXIO_ROW_LIMIT_USER_PLAYBOOKS": "17",
            "REFLEXIO_OPERATIONAL_ROW_LIMIT": "9",
        }
    )
    assert limits["agent_playbooks"] == 0
    assert limits["user_playbooks"] == 17
    assert limits["share_links"] == 9
    assert _SOURCE_TABLE not in limits
