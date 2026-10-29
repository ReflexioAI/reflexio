"""Saved publication plans must preserve source-quoted recovery rows.

Current decision, single/multi-output lineage, atomic rollback and provider
fallback contracts live in test_audited_consolidator_integration.py.
"""

import os
import tempfile

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("operation", ["overwrite", "retire", "merge"])
def test_cached_normal_plan_cannot_replace_source_quoted_rule(
    sqlite_storage, operation
):
    import json

    from reflexio.models.api_schema.domain.entities import LineageContext
    from reflexio.server.llm.input_budget import InputBudgetConfigurationError
    from reflexio.server.services.deferred_learning_plan import PlaybookWritePlan
    from reflexio.server.services.playbook.service import PlaybookGenerationService
    from reflexio.server.services.playbook.source_quoted_overflow import POLICY

    copied = _make_candidate(content="Do not deploy without named approval.")
    copied.notes = json.dumps({"policy": POLICY})
    sqlite_storage.save_user_playbooks([copied], skip_embedding=True)
    before = sqlite_storage.get_user_playbook_by_id(copied.user_playbook_id).model_dump(
        mode="json"
    )
    candidate = _make_candidate(content="Model rewrite.")
    if operation == "overwrite":
        candidate.user_playbook_id = copied.user_playbook_id
    plan = PlaybookWritePlan(
        request_id="later",
        output_pending_status=False,
        skip_aggregation=True,
        new_playbooks=[candidate],
        superseded_ids=[copied.user_playbook_id] if operation == "retire" else [],
        merge_groups=[(0, [copied.user_playbook_id])] if operation == "merge" else [],
        lineage_contexts=[
            LineageContext(op_kind="create", actor="extractor", request_id="later")
        ],
    )
    service = object.__new__(PlaybookGenerationService)
    service.storage = sqlite_storage
    with pytest.raises(InputBudgetConfigurationError), sqlite_storage.commit_scope():
        service._persist_write_plan(plan)
    assert (
        sqlite_storage.get_user_playbook_by_id(copied.user_playbook_id).model_dump(
            mode="json"
        )
        == before
    )
    assert len(sqlite_storage.get_user_playbooks(user_id="u1")) == 1


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for SQLite isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sqlite_storage(temp_storage_dir, worker_id):
    """Real SQLite storage in a per-test temp dir + per-worker org id."""
    return SQLiteStorage(
        org_id=f"test-consolidator-apply-{worker_id}",
        db_path=os.path.join(temp_storage_dir, "consolidator_apply.db"),
    )


def _make_candidate(
    *,
    user_id: str = "u1",
    content: str = "Recommend X.",
    trigger: str = "when Y",
    polarity: str = "positive",
    request_id: str = "r1",
) -> UserPlaybook:
    """Build a NEW candidate UserPlaybook (not persisted).

    Args:
        user_id: User id scoping the candidate.
        content: Candidate content body.
        trigger: Trigger string.
        polarity: ``"positive"`` or ``"negative"``.
        request_id: Request id for the candidate.

    Returns:
        A fresh ``UserPlaybook`` ready to flow through ``deduplicate``.
    """
    # Orientation is a wording convention: a negative candidate uses avoidance
    # framing ("Avoid …") and carries a failure signal in its rationale,
    # mirroring what the extractor actually writes.
    rationale = "user pushback observed" if polarity == "negative" else "r"
    return UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v0",
        request_id=request_id,
        playbook_name="default",
        content=content,
        trigger=trigger,
        rationale=rationale,
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
    )
