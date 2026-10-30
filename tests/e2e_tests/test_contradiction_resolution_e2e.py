"""An unsafe draft cannot pass real service planning/publication without audit.

The method, service, source reads, SQLite transaction and lineage are real.
Provider responses are scripted; this verifies control flow, not AI quality.
"""

import pytest
from litellm.exceptions import APIConnectionError

from reflexio.server.llm.litellm_client import LiteLLMClientError
from tests.server.services.playbook.test_audited_consolidator_integration import (
    _publication_case,
    response,
)
from tests.server.services.playbook.test_audited_consolidator_integration import (
    consolidator as consolidator,
)
from tests.server.services.playbook.test_audited_consolidator_integration import (
    storage as storage,
)

pytestmark = pytest.mark.e2e


@pytest.mark.parametrize("final_audit_clear", [False, True])
def test_contradictory_draft_only_publishes_after_independent_correction(
    consolidator, storage, monkeypatch, final_audit_clear
):
    case = _publication_case(consolidator, storage, monkeypatch, output_count=1)
    ref = {"session_id": "evidence", "turn": case.interaction_id}

    def proposal(content):
        return {
            "decisions": [
                {
                    "kind": "replace",
                    "new_ids": ["NEW-0"],
                    "existing_ids": ["EXISTING-0"],
                    "reason": "scripted proposal",
                    "outputs": [
                        {
                            "trigger": "before submitting",
                            "content": content,
                            "rationale": "See cited evidence.",
                            "evidence_refs": [ref],
                        }
                    ],
                }
            ]
        }

    finding = {
        "findings": [
            {
                "reason": "The proposal must preserve the authorization requirement.",
                "playbook_ids": ["NEW-0"],
                "evidence_refs": [ref],
            }
        ]
    }
    answers = iter(
        [
            proposal("Submit without checking authorization."),
            finding,
            proposal("Check authorization."),
            {"findings": []} if final_audit_clear else finding,
        ]
    )
    attempts = []
    original_events = storage.get_lineage_events()

    def complete(params, _timeout):
        attempts.append(params)
        assert storage.get_user_playbooks() == [case.incumbent]
        assert storage.get_lineage_events() == original_events
        try:
            answer = next(answers)
        except StopIteration:
            raise APIConnectionError(
                message="scripted provider unavailable after unresolved audit",
                llm_provider="openai",
                model="gpt-4o",
            ) from None
        return response(answer)

    monkeypatch.setattr(consolidator.client, "_completion_with_deadline", complete)
    if not final_audit_clear:
        with pytest.raises(LiteLLMClientError):
            case.service._resolve_write_plan([[case.candidate]])
        assert len(attempts) == 5
        assert storage.get_user_playbooks() == [case.incumbent]
        assert storage.get_lineage_events() == original_events
        return
    plan = case.service._resolve_write_plan([[case.candidate]])
    assert plan is not None and len(attempts) == 4
    assert [row.content for row in plan.new_playbooks] == ["Check authorization."]
    with storage.commit_scope():
        case.service._persist_write_plan(plan)
    assert [
        row.content
        for row in storage.get_user_playbooks(user_id="owner", agent_version="v1")
    ] == ["Check authorization."]
    assert all(
        "without checking" not in row.content for row in storage.get_user_playbooks()
    )
