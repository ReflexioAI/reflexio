"""Tests for publish-time shadow comparison dispatch."""

from __future__ import annotations

import queue
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.eval_overview_schema import (
    ShadowComparisonOutput,
    ShadowComparisonVerdict,
)
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.services.shadow_comparison.dispatcher import (
    dispatch_shadow_comparison_judge,
)
from reflexio.server.services.shadow_comparison.worker import (
    ShadowComparisonJob,
    ShadowComparisonWorker,
)


def _request_context() -> SimpleNamespace:
    return SimpleNamespace(
        configurator=SimpleNamespace(
            get_config=lambda: Config(storage_config=StorageConfigSQLite())
        ),
        prompt_manager=MagicMock(),
    )


def _interaction(
    *,
    interaction_id: int,
    role: str,
    content: str,
    shadow_content: str = "",
) -> Interaction:
    return Interaction(
        interaction_id=interaction_id,
        user_id="u1",
        request_id="r1",
        created_at=1_700_000_000 + interaction_id,
        role=role,
        content=content,
        shadow_content=shadow_content,
    )


def _verdict(interaction: Interaction) -> ShadowComparisonVerdict:
    return ShadowComparisonVerdict(
        verdict_id=0,
        interaction_id=str(interaction.interaction_id),
        session_id="s1",
        agent_version="v1",
        reflexio_is_request_1=True,
        output=ShadowComparisonOutput(
            better_request="1",
            is_significantly_better=True,
            comparison_reason="Request 1 wins.",
        ),
        judge_prompt_version="v1.1.0",
        created_at=datetime.now(UTC),
    )


def test_dispatcher_skips_unsupported_storage_without_judge_call(
    monkeypatch,
) -> None:
    judge_cls = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.dispatcher.ShadowComparisonJudge",
        judge_cls,
    )

    dispatch_shadow_comparison_judge(
        storage=object(),
        interactions=[
            _interaction(
                interaction_id=1,
                role="assistant",
                content="regular",
                shadow_content="shadow",
            )
        ],
        session_id="s1",
        agent_version="v1",
        request_context=_request_context(),  # type: ignore[arg-type]
        llm_client=MagicMock(),
    )

    judge_cls.assert_not_called()


def test_dispatcher_judges_shadow_assistant_turns_with_request_context(
    monkeypatch,
) -> None:
    storage = MagicMock()
    storage.supports_shadow_comparison_verdicts.return_value = True
    contexts: list[str] = []

    def fake_judge_turn(self, *, interaction, conversation_context, **_kwargs):
        contexts.append(conversation_context)
        return _verdict(interaction)

    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge."
        "ShadowComparisonJudge.judge_turn",
        fake_judge_turn,
    )

    dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=[
            _interaction(interaction_id=1, role="user", content="Please compare A"),
            _interaction(
                interaction_id=2,
                role="assistant",
                content="regular answer",
                shadow_content="shadow answer",
            ),
            _interaction(
                interaction_id=3,
                role="assistant",
                content="regular no-shadow",
            ),
        ],
        session_id="s1",
        agent_version="v1",
        request_context=_request_context(),  # type: ignore[arg-type]
        llm_client=MagicMock(),
    )

    storage.save_shadow_comparison_verdict.assert_called_once()
    assert contexts == ["User:\nPlease compare A"]


def test_dispatcher_continues_after_individual_failures(monkeypatch) -> None:
    storage = MagicMock()
    storage.supports_shadow_comparison_verdicts.return_value = True
    calls: list[int] = []

    def fake_judge_turn(self, *, interaction, **_kwargs):
        calls.append(interaction.interaction_id)
        if len(calls) == 1:
            raise RuntimeError("judge failed")
        return _verdict(interaction)

    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.judge."
        "ShadowComparisonJudge.judge_turn",
        fake_judge_turn,
    )

    dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=[
            _interaction(
                interaction_id=1,
                role="assistant",
                content="regular one",
                shadow_content="shadow one",
            ),
            _interaction(
                interaction_id=2,
                role="assistant",
                content="regular two",
                shadow_content="shadow two",
            ),
        ],
        session_id="s1",
        agent_version="v1",
        request_context=_request_context(),  # type: ignore[arg-type]
        llm_client=MagicMock(),
    )

    assert calls == [1, 2]
    storage.save_shadow_comparison_verdict.assert_called_once()


def _job(interactions: list[Interaction]) -> ShadowComparisonJob:
    return ShadowComparisonJob(
        org_id="org_test",
        interactions=interactions,
        session_id="s1",
        agent_version="v1",
    )


def test_shadow_worker_drops_when_queue_full(monkeypatch) -> None:
    worker = ShadowComparisonWorker(worker_count=1, queue_size=1)
    monkeypatch.setattr(worker._queue, "put_nowait", MagicMock(side_effect=queue.Full))
    dropped = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.worker.record_usage_event",
        dropped,
    )

    accepted = worker.enqueue(_job([]))

    assert accepted is False
    dropped.assert_called_once()
    assert dropped.call_args.kwargs["event_name"] == "shadow_comparison_dropped"


def test_shadow_worker_loop_resolves_reflexio_and_dispatches(monkeypatch) -> None:
    """The worker loop must re-resolve get_reflexio(org_id) and dispatch the job."""
    reflexio = SimpleNamespace(
        request_context=SimpleNamespace(storage=MagicMock()),
        llm_client=MagicMock(),
    )
    # get_reflexio is imported lazily inside _worker_loop, so patch the source.
    monkeypatch.setattr(
        "reflexio.server.cache.reflexio_cache.get_reflexio",
        lambda *, org_id: reflexio,  # noqa: ARG005 - matches get_reflexio kwarg signature
    )
    dispatched: list[dict] = []
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.worker.dispatch_shadow_comparison_judge",
        lambda **kwargs: dispatched.append(kwargs),
    )

    worker = ShadowComparisonWorker(worker_count=1, queue_size=4)
    worker.enqueue(
        _job(
            [
                _interaction(
                    interaction_id=1,
                    role="assistant",
                    content="regular",
                    shadow_content="shadow",
                )
            ]
        )
    )
    worker._queue.join()

    assert len(dispatched) == 1
    assert dispatched[0]["storage"] is reflexio.request_context.storage
    assert dispatched[0]["llm_client"] is reflexio.llm_client
    assert dispatched[0]["session_id"] == "s1"


def test_dispatcher_skips_non_assistant_shadow_turn(monkeypatch) -> None:
    """A shadow-bearing non-assistant turn must not be judged."""
    storage = MagicMock()
    storage.supports_shadow_comparison_verdicts.return_value = True
    judge_cls = MagicMock()
    monkeypatch.setattr(
        "reflexio.server.services.shadow_comparison.dispatcher.ShadowComparisonJudge",
        judge_cls,
    )

    dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=[
            _interaction(
                interaction_id=1,
                role="user",
                content="regular",
                shadow_content="shadow",
            )
        ],
        session_id="s1",
        agent_version="v1",
        request_context=_request_context(),  # type: ignore[arg-type]
        llm_client=MagicMock(),
    )

    judge_cls.assert_not_called()
    storage.save_shadow_comparison_verdict.assert_not_called()
