"""Exact selected normal-worker turns over synthetic, source-backed SQLite."""

from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.durable_learning.learning_admission import (
    LearningWorkSelection,
    preview_learning_work,
    run_selected_learning_work,
)
from reflexio.server.services.storage.storage_base import AgentRunStatus
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionIntegrityError,
)
from tests.server.services.extraction.test_selected_recovery_integration import (
    context as context,
)
from tests.server.services.extraction.test_selected_recovery_integration import (
    seed,
)


def grant(context, selection):
    from reflexio.server.services.storage.storage_base.learning_transition import (
        _change_learning_policy_authority_db,
    )

    storage = context.storage
    before = storage.read_learning_transition()
    assert before is not None
    with storage._learning_policy_authority_scope() as (db, authority):
        assert authority is not None
        sealed = _change_learning_policy_authority_db(
            db,
            action="seal",
            operation_id="release",
            expected_nonce=authority.authority_nonce,
            expected_generation=authority.generation,
            target_policy=authority.active_policy,
        )
    project = before.project
    return storage.grant_learning_transition(
        "release",
        expected_authority_nonce=sealed.authority_nonce,
        expected_authority_generation=sealed.generation,
        expected_nonce=project.nonce if project else None,
        expected_revision=project.revision if project else None,
        selection_sha256=selection.fingerprint(),
    )


def invoke(context, selection, state):
    assert state.project is not None
    return run_selected_learning_work(
        context,
        selection,
        operation_id="release",
        expected_nonce=state.project.nonce,
        expected_revision=state.project.revision,
        expected_authority_nonce=state.authority.authority_nonce,
        expected_authority_generation=state.authority.generation,
    )


def test_selected_turn_uses_normal_worker_without_unrelated_class_rotation(
    context, monkeypatch
):
    selected, other = seed(context), seed(context, "other")
    selection = preview_learning_work(context, run_id=selected.id)
    assert selection.eligible
    assert LearningWorkSelection.from_dict(selection.to_dict()) == selection
    state = grant(context, selection)
    provider = MagicMock(side_effect=AssertionError("Provider execution forbidden"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", provider)
    result = invoke(context, selection, state)
    assert result.status == "completed", result.to_dict()
    assert context.storage.get_agent_run(selected.id).status == AgentRunStatus.FINALIZED
    assert context.storage.get_agent_run(other.id) == other
    assert context.storage.read_learning_transition() == state
    provider.assert_not_called()


def test_known_digest_does_not_authorize_another_root(context):
    selected, other = seed(context), seed(context, "other")
    selection = preview_learning_work(context, run_id=selected.id)
    state = grant(context, selection)
    other_selection = preview_learning_work(context, run_id=other.id)
    result = invoke(context, other_selection, state)
    assert result.status == "held"
    assert context.storage.get_agent_run(other.id) == other


def test_changed_grant_revision_holds_without_adopting_selected_root(context):
    selected = seed(context)
    selection = preview_learning_work(context, run_id=selected.id)
    state = grant(context, selection)
    context.storage.grant_learning_transition(
        "release",
        expected_nonce=state.project.nonce,
        expected_revision=state.project.revision,
        expected_authority_nonce=state.authority.authority_nonce,
        expected_authority_generation=state.authority.generation,
        selection_sha256=None,
    )
    assert invoke(context, selection, state).status == "held"
    assert context.storage.get_agent_run(selected.id) == selected


def test_unknown_generation_witness_is_held_not_relabelled_finalization(context):
    run = seed(context)
    # Restore a supported ordinary generation state in this synthetic fixture,
    # retaining its actual original row. No old schema/instruction is invented.
    context.storage.update_agent_run_status(run.id, AgentRunStatus.RESUME_READY)
    selection = preview_learning_work(context, run_id=run.id)
    assert not selection.eligible
    assert "not_finalization_candidate" in selection.blockers


@pytest.mark.parametrize(
    "mutation", ["unknown_field", "wrong_version", "bool_revision", "bad_digest"]
)
def test_selection_decode_is_strict(context, mutation):
    selection = preview_learning_work(context, run_id=seed(context).id)
    raw = selection.to_dict()
    if mutation == "unknown_field":
        raw["allow_drain"] = True
    elif mutation == "wrong_version":
        raw["schema_version"] = True
    elif mutation == "bool_revision":
        raw["revision_fingerprint"] = False
    else:
        raw["instruction_fingerprint"] = "known-but-not-a-source-witness"
    with pytest.raises(LearningTransitionIntegrityError):
        LearningWorkSelection.from_dict(raw)
