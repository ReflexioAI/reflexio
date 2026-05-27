from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain import AgentSuccessEvaluationResult
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def storage(tmp_path):
    db = tmp_path / "test.db"
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        s = SQLiteStorage(org_id="test_org", db_path=str(db))
        yield s


@pytest.mark.parametrize(
    "shadow_is_success,shadow_is_escalated",
    [(True, False), (False, True), (None, None)],
)
def test_shadow_outcome_roundtrip(storage, shadow_is_success, shadow_is_escalated):
    storage.save_agent_success_evaluation_results(
        [
            AgentSuccessEvaluationResult(
                session_id="s1",
                agent_version="v1",
                evaluation_name="overall_success",
                is_success=True,
                is_escalated=False,
                shadow_is_success=shadow_is_success,
                shadow_is_escalated=shadow_is_escalated,
            )
        ]
    )
    [result] = storage.get_agent_success_evaluation_results(limit=1)
    assert result.shadow_is_success == shadow_is_success
    assert result.shadow_is_escalated == shadow_is_escalated
