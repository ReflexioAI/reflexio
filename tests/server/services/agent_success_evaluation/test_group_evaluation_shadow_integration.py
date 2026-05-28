"""F1 integration: group_evaluation_runner dispatches the per-turn shadow judge.

When session interactions carry ``shadow_content`` and the regular success
eval succeeds, the runner must invoke
:class:`ShadowComparisonJudge.judge_turn` per shadow-bearing interaction and
persist a verdict via ``storage.save_shadow_comparison_verdict``. When the
regular eval fails, the dispatch is skipped to avoid noisy verdicts on
sessions whose success grade is unreliable.

Uses a real :class:`SQLiteStorage` in a temp dir so the verdict save lands
in real storage; the LLM judge call is stubbed to a deterministic verdict.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import Interaction, Request
from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.agent_success_evaluation.group_evaluation_runner import (
    run_group_evaluation,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


@pytest.fixture
def storage() -> Generator[SQLiteStorage]:
    """Fresh SQLite store in a temp dir with embedding stubbed.

    Stubbing ``_get_embedding`` avoids the test depending on a real
    embedding provider during ``add_user_interaction``.
    """
    with (
        tempfile.TemporaryDirectory() as tmp_dir,
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
    ):
        yield SQLiteStorage(
            org_id="f1_runner_test",
            db_path=f"{tmp_dir}/reflexio.db",
        )


def _seed_session_with_shadow_interactions(
    storage: SQLiteStorage,
    *,
    session_id: str = "sess-1",
    user_id: str = "u1",
    agent_version: str = "v1",
) -> list[Interaction]:
    """Seed one Request + 3 Interactions; 2 have shadow_content, 1 doesn't.

    Uses a request ``created_at`` deep in the past so the runner's
    completion-delay gate would also accept it — but the tests pass
    ``force_regenerate=True`` anyway to bypass that gate explicitly.
    """
    ts = 1_700_000_000
    storage.add_request(
        Request(
            request_id="req-1",
            user_id=user_id,
            session_id=session_id,
            created_at=ts,
            source="test",
            agent_version=agent_version,
        )
    )
    interactions: list[Interaction] = []
    for i in range(3):
        inter = Interaction(
            user_id=user_id,
            request_id="req-1",
            created_at=ts + i,
            role="agent",
            content=f"REGULAR-{i}",
            shadow_content=f"SHADOW-{i}" if i < 2 else "",
        )
        storage.add_user_interaction(user_id, inter)
        interactions.append(inter)
    return interactions


def _make_request_context(storage: SQLiteStorage, config: Config) -> SimpleNamespace:
    """Build the minimal request_context the runner reads.

    The runner accesses ``.storage`` and the helper accesses
    ``.configurator.get_config()`` and ``.prompt_manager``. A
    SimpleNamespace stand-in keeps the test independent of the heavy
    BaseConfigurator stack.
    """
    return SimpleNamespace(
        storage=storage,
        configurator=SimpleNamespace(get_config=lambda: config),
        prompt_manager=MagicMock(),
    )


def test_runner_writes_verdicts_for_interactions_with_shadow_content(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """3 interactions, 2 carry shadow_content -> 2 verdicts saved."""
    _seed_session_with_shadow_interactions(storage)

    # Stub the regular success-eval service so it claims success without
    # touching an LLM. The runner reads .has_run_failures() and
    # .last_run_saved_result_count to decide whether to mark evaluated.
    fake_service = MagicMock()
    fake_service.has_run_failures.return_value = False
    fake_service.last_run_saved_result_count = 1
    fake_service.last_run_save_failed = False
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation."
        "group_evaluation_runner.AgentSuccessEvaluationService",
        MagicMock(return_value=fake_service),
    )

    # Stub the F1 judge to return a deterministic verdict per call. Capture
    # the interaction_ids the judge was asked about so the test can assert
    # the runner filtered to shadow-bearing rows.
    seen_interaction_ids: list[int] = []

    def fake_judge(
        self,
        *,
        interaction: Interaction,
        session_id: str,
        agent_version: str,
        rng,  # noqa: ANN001 — random.Random, kept loose to match the real signature
        user_message: str = "",
    ) -> ShadowComparisonVerdict:
        seen_interaction_ids.append(interaction.interaction_id)
        return ShadowComparisonVerdict(
            verdict_id=0,
            interaction_id=str(interaction.interaction_id),
            session_id=session_id,
            agent_version=agent_version,
            reflexio_is_request_1=True,
            output=ShadowComparisonOutput(
                better_request="1",
                is_significantly_better=True,
            ),
            judge_prompt_version="v1.0.0",
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge."
        "ShadowComparisonJudge.judge_turn",
        fake_judge,
    )

    config = Config(storage_config=StorageConfigSQLite())
    request_context = _make_request_context(storage, config)

    run_group_evaluation(
        org_id="0",
        user_id="u1",
        session_id="sess-1",
        agent_version="v1",
        source=None,
        request_context=request_context,  # type: ignore[arg-type]
        llm_client=MagicMock(),
        force_regenerate=True,
    )

    # 2 of the 3 interactions had shadow_content; expect 2 verdicts saved.
    verdicts = storage.get_shadow_comparison_verdicts(
        from_ts=0,
        to_ts=2_000_000_000,
        judge_prompt_version="v1.0.0",
    )
    assert len(verdicts) == 2, (
        f"Expected 2 verdicts for the 2 shadow-bearing interactions, "
        f"got {len(verdicts)}"
    )

    # And the judge was only invoked on the shadow-bearing interactions.
    assert len(seen_interaction_ids) == 2
    # SQLite assigns interaction_ids starting at 1 in this fresh db. The
    # first two inserted carry shadow; the third does not.
    saved_interaction_ids = {v.interaction_id for v in verdicts}
    assert saved_interaction_ids == {
        str(seen_interaction_ids[0]),
        str(seen_interaction_ids[1]),
    }


def test_runner_does_not_dispatch_judge_when_success_eval_failed(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """When the regular eval reports failures, skip F1 to avoid noisy verdicts."""
    _seed_session_with_shadow_interactions(storage)

    fake_service = MagicMock()
    fake_service.has_run_failures.return_value = True  # failures present
    fake_service.last_run_saved_result_count = 0
    fake_service.last_run_save_failed = False
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation."
        "group_evaluation_runner.AgentSuccessEvaluationService",
        MagicMock(return_value=fake_service),
    )

    judge_call_count = 0

    def fake_judge(self, **kwargs) -> None:  # noqa: ANN003 — loose match
        nonlocal judge_call_count
        judge_call_count += 1
        return

    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge."
        "ShadowComparisonJudge.judge_turn",
        fake_judge,
    )

    config = Config(storage_config=StorageConfigSQLite())
    request_context = _make_request_context(storage, config)

    run_group_evaluation(
        org_id="0",
        user_id="u1",
        session_id="sess-1",
        agent_version="v1",
        source=None,
        request_context=request_context,  # type: ignore[arg-type]
        llm_client=MagicMock(),
        force_regenerate=True,
    )

    assert judge_call_count == 0, (
        "F1 judge should NOT be dispatched when the regular eval reported failures"
    )
    # And no verdicts written.
    verdicts = storage.get_shadow_comparison_verdicts(
        from_ts=0,
        to_ts=2_000_000_000,
        judge_prompt_version="v1.0.0",
    )
    assert verdicts == []


def test_runner_continues_when_individual_judge_call_raises(
    monkeypatch: pytest.MonkeyPatch, storage: SQLiteStorage
) -> None:
    """A failing judge call on one interaction must not abort the batch.

    Seed 2 shadow-bearing interactions; make the first call raise. The
    runner should log+continue and persist the second verdict.
    """
    _seed_session_with_shadow_interactions(storage)

    fake_service = MagicMock()
    fake_service.has_run_failures.return_value = False
    fake_service.last_run_saved_result_count = 1
    fake_service.last_run_save_failed = False
    monkeypatch.setattr(
        "reflexio.server.services.agent_success_evaluation."
        "group_evaluation_runner.AgentSuccessEvaluationService",
        MagicMock(return_value=fake_service),
    )

    calls: list[int] = []

    def fake_judge(
        self,
        *,
        interaction: Interaction,
        session_id: str,
        agent_version: str,
        rng,  # noqa: ANN001
        user_message: str = "",
    ) -> ShadowComparisonVerdict | None:
        calls.append(interaction.interaction_id)
        if len(calls) == 1:
            raise RuntimeError("simulated transient judge failure")
        return ShadowComparisonVerdict(
            verdict_id=0,
            interaction_id=str(interaction.interaction_id),
            session_id=session_id,
            agent_version=agent_version,
            reflexio_is_request_1=False,
            output=ShadowComparisonOutput(
                better_request="2",
                is_significantly_better=False,
            ),
            judge_prompt_version="v1.0.0",
            created_at=datetime.now(UTC),
        )

    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge."
        "ShadowComparisonJudge.judge_turn",
        fake_judge,
    )

    config = Config(storage_config=StorageConfigSQLite())
    request_context = _make_request_context(storage, config)

    run_group_evaluation(
        org_id="0",
        user_id="u1",
        session_id="sess-1",
        agent_version="v1",
        source=None,
        request_context=request_context,  # type: ignore[arg-type]
        llm_client=MagicMock(),
        force_regenerate=True,
    )

    assert len(calls) == 2, "Both shadow-bearing interactions should be tried"
    verdicts = storage.get_shadow_comparison_verdicts(
        from_ts=0,
        to_ts=2_000_000_000,
        judge_prompt_version="v1.0.0",
    )
    assert len(verdicts) == 1, "Only the second (non-raising) call should persist"
