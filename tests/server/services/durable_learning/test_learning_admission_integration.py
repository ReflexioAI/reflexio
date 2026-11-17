"""Synthetic SQLite fences; provider entry is always a hard failure."""

from contextvars import copy_context
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from reflexio.server.llm.compute_admission import (
    ComputeAdmissionError,
    admit_external_compute,
    compute_attempt_scope,
    external_compute_admission,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.durable_learning.learning_admission import (
    LearningWork,
    fence_learning_work,
    learning_business_write,
    learning_work_scope,
    require_learning_run,
)
from reflexio.server.services.durable_learning.user_lease import (
    create_owned_agent_run,
    owned_agent_run_write,
    user_extraction_lease,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    AgentBinding,
    AgentRunRecord,
    AgentRunStatus,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningTransitionBlockedError,
    LearningTransitionIntegrityError,
)
from tests.server.services import test_durable_window_pipeline as durable

pipeline = durable.pipeline


@pytest.fixture
def storage(tmp_path, monkeypatch):
    fail = MagicMock(side_effect=AssertionError("Provider execution forbidden"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", fail)
    monkeypatch.setattr(SQLiteStorage, "_get_embedding", fail)
    store = SQLiteStorage(org_id="org", db_path=str(tmp_path / "learning.db"))
    from reflexio.test_support.learning_transition import qualify_empty_learning_scope

    qualify_empty_learning_scope(store)
    yield store
    store.conn.close()
    fail.assert_not_called()


def work(storage, user="user", reference="request"):
    return LearningWork(storage.org_id, "", user, "request", reference)


def seal(storage):
    from reflexio.server.services.storage.storage_base.learning_transition import (
        _change_learning_policy_authority_db,
    )

    with storage._learning_policy_authority_scope() as (db, root):
        assert root is not None
        return _change_learning_policy_authority_db(
            db,
            action="seal",
            operation_id="release",
            target_policy=root.active_policy,
            expected_nonce=root.authority_nonce,
            expected_generation=root.generation,
        )


def test_dispatch_composes_before_callback_and_debit(storage):
    calls = []
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
    ):
        token = external_compute_admission.set(lambda: calls.append("prepare"))
        try:
            with compute_attempt_scope(
                reserve=lambda: calls.append("debit")
            ) as progress:
                admit_external_compute()
                progress.retain()
            assert calls == ["prepare", "debit"]
            seal(storage)
            with (
                compute_attempt_scope(reserve=lambda: calls.append("new_debit")),
                pytest.raises(ComputeAdmissionError),
            ):
                admit_external_compute()
            assert calls == ["prepare", "debit"]
        finally:
            external_compute_admission.reset(token)


def test_seal_does_not_veto_returned_output_evidence(storage):
    run = AgentRunRecord(
        id="run",
        binding=AgentBinding(
            org_id="org",
            user_id="user",
            extractor_kind="profile",
            request_id="request",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={},
    )
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
    ):
        run = create_owned_agent_run(storage, run)
        seal(storage)
        returned = {"profiles": [], "usage": {"input_tokens": 7}}
        with owned_agent_run_write(storage, run):
            saved = storage.update_agent_run_status(
                run.id,
                AgentRunStatus.AGENT_COMPLETED,
                committed_output=returned,
                expected_statuses=(run.status,),
                expected_claimed_by=run.claimed_by,
                expected_claimed_at=run.claimed_at,
            )
        assert saved is not None and saved.committed_output == returned
        with (
            pytest.raises(LearningTransitionBlockedError),
            learning_business_write(storage),
        ):
            pytest.fail("Sealed business writer entered")
        assert storage.get_agent_run(run.id).committed_output == returned


def test_seal_and_reopen_do_not_revive_copied_old_work(storage):
    with user_extraction_lease(storage, "user", max_wait_seconds=0):
        with learning_work_scope(storage, work(storage)):
            copied = copy_context()
            state = seal(storage)
            from reflexio.server.services.storage.storage_base.learning_transition import (
                _change_learning_policy_authority_db,
            )

            with storage._learning_policy_authority_scope() as (db, _root):
                _change_learning_policy_authority_db(
                    db,
                    action="abort",
                    operation_id="release",
                    expected_nonce=state.authority_nonce,
                    expected_generation=state.generation,
                )
            with pytest.raises(LearningTransitionBlockedError):
                copied.run(fence_learning_work, storage)
        with learning_work_scope(storage, work(storage)):
            fence_learning_work(storage)


def test_other_run_cannot_borrow_bound_request(storage):
    run = AgentRunRecord(
        id="run",
        binding=AgentBinding(
            org_id="org",
            user_id="user",
            extractor_kind="profile",
            request_id="request",
            agent_version="v1",
            source="api",
        ),
        status=AgentRunStatus.RUNNING,
        generation_request_snapshot={},
    )
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
    ):
        require_learning_run(storage, run)
        with pytest.raises(LearningTransitionIntegrityError):
            require_learning_run(
                storage, replace(run, binding=replace(run.binding, request_id="other"))
            )


def test_unbound_query_compute_remains_available_when_learning_is_sealed(storage):
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
    ):
        seal(storage)
    admit_external_compute()
    with (
        pytest.raises(LearningTransitionBlockedError),
        user_extraction_lease(storage, "new-user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage, "new-user")),
    ):
        pytest.fail("A new user bypassed the project seal")


def test_text_refuses_before_overflow_guard_can_reserve(storage):
    guard = MagicMock()
    client = LiteLLMClient(
        LiteLLMConfig(model="gpt-4o", fallback_models=["gpt-4o-mini"], max_retries=0)
    )
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
    ):
        seal(storage)
        with pytest.raises(ComputeAdmissionError):
            client.generate_chat_response(
                [{"role": "user", "content": "synthetic learning"}],
                provider_request_guard=guard,
            )
    guard.assert_not_called()


def test_learning_compute_cannot_retain_business_writer_locks(storage):
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
        storage.commit_scope(),
        pytest.raises(ComputeAdmissionError),
    ):
        admit_external_compute()


def test_gate_false_outcome_is_saved_but_window_cannot_complete_after_seal(
    pipeline, monkeypatch
):
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.profile.service import ProfileGenerationService

    engine, worker, client = pipeline
    store = engine.get_storage()
    provider = MagicMock(side_effect=AssertionError("Provider execution forbidden"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", provider)
    # This control exercises the maintained no-overflow/gate-false branch. It
    # scripts only the semantic gate result, not publication/lease ownership.
    monkeypatch.setattr(WindowExecutor, "_coordinator", lambda *_args: None)

    def gate_false(_service, _request):
        seal(store)
        return

    monkeypatch.setattr(ProfileGenerationService, "compute_generation", gate_false)
    assert client.post(
        "/api/publish_interaction", json=durable.payload("sealed-gate", force=True)
    ).json()["success"]
    assert worker.drain_org("stream-test", 1, 300) == 0
    row = store.conn.execute(
        "SELECT completed,outcome FROM extraction_windows WHERE kind='profile'"
    ).fetchone()
    assert row is not None and not row["completed"] and row["outcome"] is not None
    assert store.count_all_profiles() == 0
    provider.assert_not_called()


def _wrapped_transition_refusal(*_args, **_kwargs):
    try:
        raise LearningTransitionBlockedError("Learning transition is sealed")
    except LearningTransitionBlockedError as error:
        raise ComputeAdmissionError(error) from error


def test_wrapped_dispatch_hold_cannot_report_successful_empty_generation():
    from reflexio.server.services.base_generation_service import BaseGenerationService

    service = MagicMock(spec=BaseGenerationService)
    service.compute_generation.side_effect = _wrapped_transition_refusal
    with pytest.raises(ComputeAdmissionError):
        BaseGenerationService._run_generation(service, object())
    service.persist_generation.assert_not_called()
    service.emit_generation_side_effects.assert_not_called()
    service._record_generation_event.assert_not_called()


def test_wrapped_gate_hold_cannot_fall_back_to_run(monkeypatch):
    from types import SimpleNamespace

    from reflexio.server.services.base_generation._should_run import (
        ShouldRunPrecheckMixin,
    )

    service = MagicMock()
    service.service_config = SimpleNamespace(
        auto_run=True, force_extraction=False, window_interactions=None
    )
    service._window_skip_should_run = False
    service._collect_scoped_interactions_for_precheck.return_value = (
        [object()],
        object(),
    )
    service._build_should_run_prompt.return_value = "synthetic gate"
    service._resolve_should_run_model.return_value = "synthetic"
    service.client.generate_chat_response.side_effect = _wrapped_transition_refusal
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.base_generation_service._cheap_should_run_reject",
        lambda _sessions: None,
    )
    with pytest.raises(ComputeAdmissionError):
        ShouldRunPrecheckMixin._should_run_before_extraction(
            service, SimpleNamespace(extractor_name="synthetic")
        )


def test_partial_compute_then_seal_preserves_charge_and_typed_hold(storage):
    from reflexio.server.llm.compute_admission import (
        ComputeTransitionInterruptedError,
        record_compute_result,
    )
    from reflexio.server.services.durable_learning.learning_admission import (
        learning_transition_is_held,
    )
    from reflexio.server.services.storage.error import execution_input_is_busy

    charges = []
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0),
        learning_work_scope(storage, work(storage)),
        pytest.raises(ComputeTransitionInterruptedError) as raised,
        compute_attempt_scope(reserve=lambda: charges.append(1)),
    ):
        admit_external_compute()
        record_compute_result()
        seal(storage)
        admit_external_compute()
    assert charges == [1]
    assert learning_transition_is_held(raised.value)
    assert not execution_input_is_busy(raised.value)
    assert raised.value.__cause__ is None


def test_actual_window_partial_compute_hold_spends_no_failure_retry(
    pipeline, monkeypatch
):
    from reflexio.server.llm.compute_admission import record_compute_result
    from reflexio.server.services.durable_learning.window_executor import WindowExecutor
    from reflexio.server.services.durable_learning.worker import DurableLearningWorker
    from reflexio.server.services.profile.service import ProfileGenerationService

    engine, worker, client = pipeline
    storage = engine.get_storage()
    provider = MagicMock(side_effect=AssertionError("No physical provider allowed"))
    monkeypatch.setattr(LiteLLMClient, "_completion_with_hard_timeout", provider)
    monkeypatch.setattr(WindowExecutor, "_coordinator", lambda *_: None)
    admitted = []

    def partial(_service, _request):
        admit_external_compute()
        admitted.append("returned")
        record_compute_result()
        seal(storage)
        admit_external_compute()

    monkeypatch.setattr(ProfileGenerationService, "compute_generation", partial)
    original = DurableLearningWorker._handle_window_failure
    outcomes = []

    def classify(*args):
        outcome = original(*args)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(
        DurableLearningWorker, "_handle_window_failure", staticmethod(classify)
    )
    assert client.post(
        "/api/publish_interaction", json=durable.payload("partial-seal", force=True)
    ).json()["success"]
    assert worker.drain_org(storage.org_id, 1, 300) == 0
    assert outcomes == ["transition_held"]
    row = storage.conn.execute(
        "SELECT attempts,last_error,retry_at,completed_seq FROM extraction_cursors WHERE kind='profile'"
    ).fetchone()
    assert row["attempts"] == 1  # Only the real admitted turn remains charged.
    assert row["last_error"] == "interrupted_compute"
    assert row["retry_at"] != float("inf")
    assert row["completed_seq"] == 0
    assert admitted == ["returned"] and storage.count_all_profiles() == 0
    provider.assert_not_called()


@pytest.mark.parametrize("partial", [False, True])
def test_effects_only_selection_refuses_compute_before_callbacks_or_debit(
    monkeypatch, partial
):
    from reflexio.server.llm.compute_admission import (
        ComputeTransitionInterruptedError,
        record_compute_result,
    )
    from reflexio.server.services.durable_learning import selected_learning
    from reflexio.server.services.durable_learning.learning_admission import (
        learning_transition_is_held,
    )

    digest = "a" * 64
    selection = selected_learning.LearningWorkSelection(
        schema_version=1,
        org_id="org",
        project_id="",
        user_id="user",
        kind="window",
        reference="window",
        revision_fingerprint=digest,
        payload_fingerprint=digest,
        config_fingerprint=digest,
        source_fingerprint=digest,
        instruction_fingerprint=digest,
        effects_only=True,
    )
    monkeypatch.setattr(
        selected_learning, "current_selected_learning", lambda: selection
    )
    observed = []
    token = external_compute_admission.set(lambda: observed.append("callback"))
    try:
        with (
            pytest.raises(
                ComputeTransitionInterruptedError if partial else ComputeAdmissionError
            ) as raised,
            compute_attempt_scope(reserve=lambda: observed.append("debit")),
        ):
            if partial:
                record_compute_result()
            admit_external_compute()
    finally:
        external_compute_admission.reset(token)
    assert learning_transition_is_held(raised.value)
    assert observed == []


def test_effects_only_selection_cannot_enter_cached_learning_writer(monkeypatch):
    from reflexio.server.services.durable_learning import selected_learning

    digest = "a" * 64
    selected = selected_learning.LearningWorkSelection(
        schema_version=1,
        org_id="org",
        project_id="",
        user_id="user",
        kind="window",
        reference="window",
        revision_fingerprint=digest,
        payload_fingerprint=digest,
        config_fingerprint=digest,
        source_fingerprint=digest,
        instruction_fingerprint=digest,
        effects_only=True,
    )
    monkeypatch.setattr(
        selected_learning, "current_selected_learning", lambda: selected
    )
    writer = MagicMock()
    callback = MagicMock()
    with pytest.raises(LearningTransitionBlockedError), learning_business_write(writer):
        callback()
    writer.commit_scope.assert_not_called()
    callback.assert_not_called()


def _legacy_source_run(storage):
    from reflexio.models.api_schema.service_schemas import Interaction, Request

    storage.add_request(
        Request(
            request_id="legacy-request",
            user_id="user",
            created_at=100,
            source="test",
            session_id="session",
        )
    )
    for identity in (1, 2):
        storage._insert_interaction(
            Interaction(
                interaction_id=identity,
                user_id="user",
                request_id="legacy-request",
                created_at=identity,
                role="user",
                content="Use the stated deployment rule.",
            )
        )
    storage.create_agent_run(
        AgentRunRecord(
            id="legacy-run",
            binding=AgentBinding(
                org_id=storage.org_id,
                extractor_kind="playbook",
                user_id=None,
                request_id="legacy-request",
                agent_version="v1",
                source="test",
                source_interaction_ids=[1, 2],
            ),
            status=AgentRunStatus.RESUME_READY,
            generation_request_snapshot={},
        )
    )
    return storage.get_agent_run("legacy-run")


def test_legacy_admission_reuses_complete_proof_on_actual_fenced_writer(
    storage, monkeypatch
):
    from reflexio.server.services.playbook import review_window

    run = _legacy_source_run(storage)
    original = review_window.infer_playbook_review_user_id
    proofs = []
    with (
        user_extraction_lease(storage, "user", max_wait_seconds=0) as owner,
        learning_work_scope(
            storage, LearningWork(storage.org_id, "", "user", "run", run.id)
        ),
    ):

        def observed(**kwargs):
            assert storage.in_commit_scope
            owner.fence(storage)
            proofs.append(tuple(kwargs["source_interaction_ids"]))
            return original(**kwargs)

        monkeypatch.setattr(review_window, "infer_playbook_review_user_id", observed)
        require_learning_run(storage, run)
    assert proofs == [(1, 2)]
    assert (
        storage.get_agent_run(run.id).binding.user_id is None
    )  # Admission isn't adoption.


@pytest.mark.parametrize(
    "invalid",
    [
        "mixed_interactions",
        "request_owner",
        "erased_interaction",
        "missing_request",
        "empty_source",
        "profile_null",
        "new_playbook_null",
        "other_user",
        "wrong_project",
        "wrong_root",
        "cross_org",
    ],
)
def test_legacy_admission_never_guesses_missing_or_cross_scope_owner(storage, invalid):
    run = _legacy_source_run(storage)
    if invalid == "mixed_interactions":
        storage._execute(
            "UPDATE interactions SET user_id='other' WHERE interaction_id=2"
        )
    elif invalid == "request_owner":
        storage._execute(
            "UPDATE requests SET user_id='other' WHERE request_id='legacy-request'"
        )
    elif invalid == "erased_interaction":
        storage._execute("DELETE FROM interactions WHERE interaction_id=2")
    elif invalid == "missing_request":
        storage._execute("DELETE FROM requests WHERE request_id='legacy-request'")
    elif invalid == "empty_source":
        run = replace(run, binding=replace(run.binding, source_interaction_ids=[]))
    elif invalid == "profile_null":
        run = replace(run, binding=replace(run.binding, extractor_kind="profile"))
    elif invalid == "new_playbook_null":
        run = replace(run, created_at=None)
    elif invalid == "wrong_project":
        run = replace(run, project_id="other")
    elif invalid == "cross_org":
        run = replace(run, binding=replace(run.binding, org_id="other"))
    user = "other" if invalid == "other_user" else "user"
    reference = "other-run" if invalid == "wrong_root" else run.id
    before = storage.get_agent_run(run.id)
    with (
        user_extraction_lease(storage, user, max_wait_seconds=0),
        learning_work_scope(
            storage, LearningWork(storage.org_id, "", user, "run", reference)
        ),
        pytest.raises(LearningTransitionIntegrityError),
    ):
        require_learning_run(storage, run)
    assert storage.get_agent_run(run.id) == before


def test_legacy_preliminary_proof_cannot_skip_native_adoption_writer(storage):
    from reflexio.server.services.durable_learning.user_lease import agent_run_user_work
    from reflexio.server.services.storage.storage_base import LeaseLostError

    run = _legacy_source_run(storage)
    with (
        agent_run_user_work(storage, run),
        pytest.raises(LeaseLostError),
        owned_agent_run_write(storage, run),
    ):
        pytest.fail("Preliminary owner proof must not authorize a run writer")
    assert storage.get_agent_run(run.id).binding.user_id is None


def test_source_erasure_after_preliminary_proof_refuses_real_adoption(
    storage, monkeypatch
):
    from reflexio.server.services.playbook import review_window

    run = _legacy_source_run(storage)
    storage.update_agent_run_status(
        run.id, AgentRunStatus.FINALIZATION_FAILED, committed_output={"playbooks": []}
    )
    run = storage.get_agent_run(run.id)
    original = review_window.infer_playbook_review_user_id
    erased = []

    def prove_then_erase(**kwargs):
        result = original(**kwargs)
        if storage.in_commit_scope and not erased:
            storage._execute("DELETE FROM interactions WHERE interaction_id=2")
            erased.append(2)
        return result

    with user_extraction_lease(storage, "user", max_wait_seconds=0) as owner:
        position = storage._read_owned_work_position("user", owner.token)
        cursor, descriptor = storage._resume_candidate_page(
            "finalization", user_id="user", token=owner.token
        )
        assert descriptor is not None and descriptor["id"] == run.id
        monkeypatch.setattr(
            review_window, "infer_playbook_review_user_id", prove_then_erase
        )
        selected, adopted = storage._commit_owned_work_selection(
            user_id="user",
            token=owner.token,
            expected_position=position,
            work_class="finalization",
            cursor=cursor,
            expected_run=run,
            worker_id="legacy-test",
        )
        assert selected and adopted is None
    assert erased == [2]
    assert storage.get_agent_run(run.id) == run
