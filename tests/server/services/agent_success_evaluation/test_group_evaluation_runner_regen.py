"""Tests for force_regenerate on run_group_evaluation.

These cover the regenerate flow's behavior at the runner layer:
- force_regenerate=True bypasses both the operation-state "already evaluated"
  short-circuit and the completeness-delay gate so an operator can re-evaluate
  any session.
- When regenerating, the runner captures the session's prior result_ids BEFORE
  running the eval, then deletes those rows by id AFTER the new save lands. This
  guarantees that an LLM/save failure never wipes the session's eval rows.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from reflexio.models.api_schema.service_schemas import (
    AgentSuccessEvaluationResult,
    Interaction,
    Request,
)
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)


def _make_request(request_id: str, user_id: str, session_id: str) -> Request:
    """Create a request old enough to pass the completion delay gate.

    Args:
        request_id (str): Request identifier.
        user_id (str): Owner user identifier.
        session_id (str): Session identifier.

    Returns:
        Request: A Request with created_at well in the past.
    """
    now = int(datetime.now(UTC).timestamp())
    return Request(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        created_at=now - 10000,
    )


def _make_interaction(request_id: str, user_id: str) -> Interaction:
    """Create a minimal interaction tied to a request.

    Args:
        request_id (str): Owning request identifier.
        user_id (str): Owner user identifier.

    Returns:
        Interaction: A populated Interaction instance.
    """
    now = int(datetime.now(UTC).timestamp())
    return Interaction(
        interaction_id=1,
        user_id=user_id,
        request_id=request_id,
        content="test content",
        role="user",
        created_at=now - 9999,
    )


def _make_prior_result(
    result_id: int,
    session_id: str,
    evaluation_name: str,
    agent_version: str = "1.0.0",
    user_id: str = "user_a",
) -> AgentSuccessEvaluationResult:
    """Construct an AgentSuccessEvaluationResult representing a prior-run row.

    Args:
        result_id (int): Primary key value the storage layer would have assigned.
        session_id (str): Owning session.
        evaluation_name (str): Evaluator identifier.
        agent_version (str): Agent version scope.

    Returns:
        AgentSuccessEvaluationResult: Minimal valid prior-run row.
    """
    return AgentSuccessEvaluationResult(
        result_id=result_id,
        user_id=user_id,
        session_id=session_id,
        agent_version=agent_version,
        evaluation_name=evaluation_name,
        is_success=True,
        failure_type=None,
        failure_reason=None,
        regular_vs_shadow=None,
        number_of_correction_per_session=0,
        user_turns_to_resolution=None,
        is_escalated=False,
        embedding=[],
        created_at=1,
    )


def _make_storage(
    *,
    with_evaluated_marker: bool,
    prior_results: list[AgentSuccessEvaluationResult] | None = None,
) -> MagicMock:
    """Build a storage mock seeded with one request + one interaction.

    Args:
        with_evaluated_marker (bool): When True, get_operation_state returns a
            payload whose operation_state.evaluated is True — simulating a
            session that's already been evaluated.
        prior_results (list[AgentSuccessEvaluationResult] | None): Optional
            prior-run rows that get_agent_success_evaluation_results should
            return — exercises the new "capture old result_ids" path.

    Returns:
        MagicMock: A storage stub wired with the standard return values.
    """
    storage = MagicMock()
    if with_evaluated_marker:
        storage.get_operation_state.return_value = {
            "operation_state": {"evaluated": True, "evaluated_at": 1}
        }
    else:
        storage.get_operation_state.return_value = None
    storage.get_requests_by_session.return_value = [
        _make_request("req_1", "user_a", "session_a")
    ]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_1", "user_a")
    ]
    storage.get_agent_success_evaluation_results.return_value = prior_results or []

    # The runner captures prior result ids via the targeted, indexed lookup
    # get_agent_success_evaluation_result_ids (added in the evaluation-overview
    # read optimization). Mirror the storage layer's own (user_id, session_id,
    # evaluation_name, agent_version) filtering instead of a static return — so
    # the test fails if the runner wires the WRONG identity tuple, not merely if
    # it forgets to call the method.
    def _result_ids_for(
        user_id: str, session_id: str, evaluation_name: str, agent_version: str
    ) -> list[int]:
        """Return seeded result_ids matching one eval identity tuple (mirrors storage).

        Args:
            user_id (str): Owning user to match.
            session_id (str): Owning session to match.
            evaluation_name (str): Evaluator identifier to match.
            agent_version (str): Agent version scope to match.

        Returns:
            list[int]: result_ids of the seeded prior rows whose full identity
                tuple matches the arguments.
        """
        return [
            r.result_id
            for r in (prior_results or [])
            if r.user_id == user_id
            and r.session_id == session_id
            and r.evaluation_name == evaluation_name
            and r.agent_version == agent_version
        ]

    storage.get_agent_success_evaluation_result_ids.side_effect = _result_ids_for
    storage.delete_agent_success_evaluation_results_by_ids.return_value = len(
        prior_results or []
    )
    return storage


def test_force_regenerate_bypasses_already_evaluated_short_circuit() -> None:
    """force_regenerate=True must still invoke the service on a marked session."""
    storage = _make_storage(with_evaluated_marker=True)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    # Service was constructed and invoked despite the existing evaluated marker.
    service.run.assert_called_once()


def test_force_regenerate_builds_evaluation_request() -> None:
    """force_regenerate runs the eval service with a request for the session."""
    storage = _make_storage(with_evaluated_marker=False)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    service.run.assert_called_once()
    eval_request = service.run.call_args.args[0]
    assert eval_request.session_id == "session_a"
    assert eval_request.agent_version == "1.0.0"


def test_force_regenerate_deletes_prior_results() -> None:
    """force_regenerate captures the session's prior result rows then deletes them
    by id AFTER save.

    The delete is scoped to the captured prior result_ids (not the
    (session, name, version) triple) and happens AFTER the new rows have
    been saved successfully. This avoids the zero-row window that would
    otherwise be exposed if the LLM/save step failed.
    """
    prior = [
        _make_prior_result(
            result_id=42, session_id="session_a", evaluation_name="overall_success"
        )
    ]
    storage = _make_storage(with_evaluated_marker=True, prior_results=prior)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with (
        patch(
            "reflexio.server.services.agent_success_evaluation"
            ".group_evaluation_runner.AgentSuccessEvaluationService"
        ) as service_cls,
        patch(
            "reflexio.server.services.agent_success_evaluation"
            ".group_evaluation_runner.get_extractor_name",
            return_value="overall_success",
        ),
    ):
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    # By-id delete called once with the captured prior result_ids — proving the
    # runner passed the matching (user_a, session_a, overall_success, 1.0.0) identity to
    # get_agent_success_evaluation_result_ids (the mock now filters on it).
    storage.delete_agent_success_evaluation_results_by_ids.assert_called_once_with([42])
    # The old triple-scoped delete must NOT be called by the new flow.
    storage.delete_agent_success_evaluation_results_for_session.assert_not_called()


def test_force_regenerate_with_no_prior_rows_does_not_delete() -> None:
    """force_regenerate with no prior result rows must NOT call any delete.

    When the session has no prior eval rows, there is nothing to clean up, so
    the runner skips the by-id delete entirely (and never falls back to the
    old session-scoped delete).
    """
    storage = _make_storage(with_evaluated_marker=True)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    storage.delete_agent_success_evaluation_results_for_session.assert_not_called()
    storage.delete_agent_success_evaluation_results_by_ids.assert_not_called()
    service.run.assert_called_once()


def test_regenerate_preserves_old_rows_when_llm_run_fails() -> None:
    """If the eval service reports failures, the prior rows MUST NOT be deleted.

    This is the core durability guarantee: the delete-then-LLM-then-save flow
    used to wipe rows on any LLM failure. The new ordering captures ids first,
    saves new rows, and only deletes the captured ids AFTER success.
    """
    prior = [
        _make_prior_result(
            result_id=99, session_id="session_a", evaluation_name="overall_success"
        ),
        _make_prior_result(
            result_id=100, session_id="session_a", evaluation_name="overall_success"
        ),
    ]
    storage = _make_storage(with_evaluated_marker=True, prior_results=prior)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        # Simulate an LLM/save failure.
        service.has_run_failures.return_value = True
        service.last_run_save_failed = True
        service.last_run_saved_result_count = 0
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    # No delete on either path — old rows remain in storage.
    storage.delete_agent_success_evaluation_results_by_ids.assert_not_called()
    storage.delete_agent_success_evaluation_results_for_session.assert_not_called()
    # Evaluated marker must NOT be written on failure either.
    storage.upsert_operation_state.assert_not_called()


def test_regenerate_preserves_old_rows_when_zero_results_saved() -> None:
    """If has_run_failures is False but no rows were saved, preserve old rows.

    Symmetric to the failure case: an empty save (e.g. all candidates filtered
    out) is treated the same as a failure for the purpose of the cleanup
    delete, because deleting would leave the session with zero rows.
    """
    prior = [
        _make_prior_result(
            result_id=77, session_id="session_a", evaluation_name="overall_success"
        )
    ]
    storage = _make_storage(with_evaluated_marker=True, prior_results=prior)
    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 0
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    storage.delete_agent_success_evaluation_results_by_ids.assert_not_called()
    storage.upsert_operation_state.assert_not_called()


def test_regenerate_happy_path_with_real_sqlite_storage(tmp_path) -> None:
    """End-to-end: real SQLite storage shows old rows replaced after a successful regen.

    Uses real SQLiteStorage in a temp dir to verify the runner integrates
    correctly with the actual storage contract (not just MagicMock semantics).
    """
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    db_path = tmp_path / "reflexio.db"
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="test_regen_happy", db_path=str(db_path))

        # Seed: one prior result for (session_a, overall_success, 1.0.0).
        # failure_type="old_value" lets us distinguish it from the new row.
        old_row = AgentSuccessEvaluationResult(
            result_id=0,
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            evaluation_name="overall_success",
            is_success=False,
            failure_type="old_value",
            failure_reason="prior verdict",
            regular_vs_shadow=None,
            number_of_correction_per_session=0,
            user_turns_to_resolution=None,
            is_escalated=False,
            embedding=[],
            created_at=1,
        )
        storage.save_agent_success_evaluation_results([old_row])

        # Seed a request + interaction so the runner can build data models.
        now = int(datetime.now(UTC).timestamp())
        storage.add_request(
            Request(
                request_id="req_1",
                user_id="user_a",
                session_id="session_a",
                created_at=now - 10_000,
                source="api",
                agent_version="1.0.0",
            )
        )
        storage.add_user_interaction(
            "user_a",
            Interaction(
                interaction_id=0,
                user_id="user_a",
                request_id="req_1",
                content="hi",
                role="user",
                created_at=now - 9_999,
            ),
        )

        request_context = MagicMock()
        request_context.storage = storage
        llm_client = MagicMock()

        # Fake the service: on .run(), save a new row imitating the real flow.
        def fake_run(_request) -> None:
            storage.save_agent_success_evaluation_results(
                [
                    AgentSuccessEvaluationResult(
                        result_id=0,
                        session_id="session_a",
                        agent_version="1.0.0",
                        evaluation_name="overall_success",
                        is_success=True,
                        failure_type="new_value",
                        failure_reason="fresh verdict",
                        regular_vs_shadow=None,
                        number_of_correction_per_session=0,
                        user_turns_to_resolution=None,
                        is_escalated=False,
                        embedding=[],
                        created_at=now,
                    )
                ]
            )

        with (
            patch(
                "reflexio.server.services.agent_success_evaluation"
                ".group_evaluation_runner.AgentSuccessEvaluationService"
            ) as service_cls,
            patch(
                "reflexio.server.services.agent_success_evaluation"
                ".group_evaluation_runner.get_extractor_name",
                return_value="overall_success",
            ),
        ):
            service = MagicMock()
            service.run.side_effect = fake_run
            service.has_run_failures.return_value = False
            service.last_run_saved_result_count = 1
            service_cls.return_value = service

            run_group_evaluation(
                org_id="org_a",
                user_id="user_a",
                session_id="session_a",
                agent_version="1.0.0",
                source="api",
                request_context=request_context,
                llm_client=llm_client,
                force_regenerate=True,
            )

        # Old row gone, new row in place. Exactly one row for the tuple.
        remaining = storage.get_agent_success_evaluation_results(limit=100)
        matching = [
            r
            for r in remaining
            if r.session_id == "session_a"
            and r.evaluation_name == "overall_success"
            and r.agent_version == "1.0.0"
        ]
        assert len(matching) == 1
        assert matching[0].failure_type == "new_value"


def test_regenerate_failure_preserves_old_rows_with_real_sqlite_storage(
    tmp_path,
) -> None:
    """End-to-end durability: LLM failure leaves the prior row in storage.

    The bug being fixed: a failed regen used to delete old rows BEFORE the
    new save, leaving the session with zero eval rows. With the new order,
    a failing service.run() means no delete is attempted and the row is
    preserved.
    """
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    db_path = tmp_path / "reflexio.db"
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="test_regen_failure", db_path=str(db_path))

        old_row = AgentSuccessEvaluationResult(
            result_id=0,
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            evaluation_name="overall_success",
            is_success=False,
            failure_type="old_value",
            failure_reason="prior verdict",
            regular_vs_shadow=None,
            number_of_correction_per_session=0,
            user_turns_to_resolution=None,
            is_escalated=False,
            embedding=[],
            created_at=1,
        )
        storage.save_agent_success_evaluation_results([old_row])

        now = int(datetime.now(UTC).timestamp())
        storage.add_request(
            Request(
                request_id="req_1",
                user_id="user_a",
                session_id="session_a",
                created_at=now - 10_000,
                source="api",
                agent_version="1.0.0",
            )
        )
        storage.add_user_interaction(
            "user_a",
            Interaction(
                interaction_id=0,
                user_id="user_a",
                request_id="req_1",
                content="hi",
                role="user",
                created_at=now - 9_999,
            ),
        )

        request_context = MagicMock()
        request_context.storage = storage
        llm_client = MagicMock()

        with (
            patch(
                "reflexio.server.services.agent_success_evaluation"
                ".group_evaluation_runner.AgentSuccessEvaluationService"
            ) as service_cls,
            patch(
                "reflexio.server.services.agent_success_evaluation"
                ".group_evaluation_runner.get_extractor_name",
                return_value="overall_success",
            ),
        ):
            service = MagicMock()
            # Simulate the LLM failure path. Service .run() is a no-op (no save).
            service.has_run_failures.return_value = True
            service.last_run_save_failed = True
            service.last_run_saved_result_count = 0
            service_cls.return_value = service

            run_group_evaluation(
                org_id="org_a",
                user_id="user_a",
                session_id="session_a",
                agent_version="1.0.0",
                source="api",
                request_context=request_context,
                llm_client=llm_client,
                force_regenerate=True,
            )

        # The old row is STILL THERE — durability guaranteed.
        remaining = storage.get_agent_success_evaluation_results(limit=100)
        matching = [
            r
            for r in remaining
            if r.session_id == "session_a" and r.evaluation_name == "overall_success"
        ]
        assert len(matching) == 1
        assert matching[0].failure_type == "old_value"


def test_force_regenerate_bypasses_completeness_delay_gate() -> None:
    """force_regenerate=True must skip the delay check even for a fresh session."""
    # Build a request just a moment old — would normally trip the delay gate.
    storage = MagicMock()
    storage.get_operation_state.return_value = None
    now = int(datetime.now(UTC).timestamp())
    fresh_request = Request(
        request_id="req_fresh",
        user_id="user_a",
        session_id="session_a",
        created_at=now,
    )
    storage.get_requests_by_session.return_value = [fresh_request]
    storage.get_interactions_by_request_ids.return_value = [
        _make_interaction("req_fresh", "user_a")
    ]

    request_context = MagicMock()
    request_context.storage = storage
    llm_client = MagicMock()

    with patch(
        "reflexio.server.services.agent_success_evaluation"
        ".group_evaluation_runner.AgentSuccessEvaluationService"
    ) as service_cls:
        service = MagicMock()
        service.has_run_failures.return_value = False
        service.last_run_saved_result_count = 1
        service_cls.return_value = service

        run_group_evaluation(
            org_id="org_a",
            user_id="user_a",
            session_id="session_a",
            agent_version="1.0.0",
            source="api",
            request_context=request_context,
            llm_client=llm_client,
            force_regenerate=True,
        )

    service.run.assert_called_once()
