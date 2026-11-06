"""Refusal paths of the LiteLLM session-compaction adapter.

Each guard must refuse before a provider call or before an unusable result can
be checkpointed: unverified context budgets, model/route drift inside the
request, malformed provider output, colliding source identities and tampered
cache receipts.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import litellm
import pytest

from reflexio.server.llm import ProviderRequestGuardError
from reflexio.server.services.offline_tuner.evidence.compaction import (
    CompactionError,
    CompactionPoint,
    SessionCompaction,
)
from reflexio.server.services.offline_tuner.open_world import compaction
from reflexio.server.services.offline_tuner.open_world.compaction import (
    SessionCompactor,
    _apply,
    _SelectedCompaction,
    _source,
    verified_compaction_context,
)
from tests.server.services.offline_tuner.open_world.test_compaction import (
    client,
    document,
)


def _compact(compactor: SessionCompactor, **kwargs: Any):
    values: dict[str, Any] = {
        "through_request_id": "request",
        "governance_epochs": (),
        "before_call": lambda: None,
    }
    values.update(kwargs)
    return compactor.compact(document(), **values)


def _receipt_from_first_run() -> tuple[dict[str, Any], SessionCompactor, Any]:
    fake = client()
    compactor = SessionCompactor(fake, project_id="project")
    receipts: list[dict[str, Any]] = []
    _compact(
        compactor,
        checkpoint=lambda identity, batches, value: receipts.append(
            {
                "identity": identity,
                "batches": batches,
                "complete": batches == -1,
                "compaction": value.model_dump(mode="json"),
            }
        ),
    )
    fake.generate_chat_response.reset_mock()
    return receipts[-1], compactor, fake


def test_pinned_route_context_is_verified_without_a_catalogue_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookup = Mock(side_effect=AssertionError("catalogue consulted"))
    monkeypatch.setattr(litellm, "get_model_info", lookup)
    assert verified_compaction_context("minimax/MiniMax-M2.7") == 204_800
    lookup.assert_not_called()


def test_unknown_route_context_is_refused_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unknown(model: str) -> dict[str, Any]:
        raise ValueError(f"model {model} not mapped")

    monkeypatch.setattr(litellm, "get_model_info", unknown)
    with pytest.raises(CompactionError, match="context is not verified") as caught:
        verified_compaction_context("custom/unknown-route")
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({"max_input_tokens": 128_000, "max_tokens": 8_192}, 128_000),
        ({"max_input_tokens": None, "max_tokens": 70_000}, 70_000),
        ({"max_input_tokens": 65_536}, 65_536),
    ],
)
def test_catalogue_context_prefers_input_limit(
    monkeypatch: pytest.MonkeyPatch, info: dict[str, Any], expected: int
) -> None:
    monkeypatch.setattr(litellm, "get_model_info", lambda _m: info)
    assert verified_compaction_context("openai/some-model") == expected


@pytest.mark.parametrize(
    "info",
    [{"max_input_tokens": 65_535}, {"max_tokens": 8_192}, {}],
    ids=("just-under", "output-limit-only", "empty"),
)
def test_context_below_the_usable_floor_is_refused(
    monkeypatch: pytest.MonkeyPatch, info: dict[str, Any]
) -> None:
    monkeypatch.setattr(litellm, "get_model_info", lambda _m: info)
    with pytest.raises(CompactionError, match="no verified usable context budget"):
        verified_compaction_context("openai/small-model")


def test_compactor_construction_refuses_an_unverified_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "get_model_info", lambda _m: {})
    fake = SimpleNamespace(config=SimpleNamespace(model="openai/tiny"))
    with pytest.raises(CompactionError, match="usable context budget"):
        SessionCompactor(fake, project_id="project")  # type: ignore[arg-type]


def _guard_of(fake: Any) -> Any:
    return fake.generate_chat_response.call_args.kwargs["provider_request_guard"]


def test_request_guard_refuses_a_changed_model() -> None:
    fake = client()
    _compact(SessionCompactor(fake, project_id="project"))
    with pytest.raises(ProviderRequestGuardError, match="model changed"):
        _guard_of(fake)({"model": "openai/gpt-4o", "messages": []})


def test_request_guard_refuses_a_request_over_the_context_budget() -> None:
    fake = client()
    compactor = SessionCompactor(fake, project_id="project")
    _compact(compactor)
    guard = _guard_of(fake)
    oversized = [{"role": "user", "content": "x" * compactor.context_tokens}]
    with pytest.raises(ProviderRequestGuardError, match="exceeds model context"):
        guard({"model": fake.config.model, "messages": oversized})
    # A request that fits the reservation is admitted.
    guard({"model": fake.config.model, "messages": [{"role": "user", "content": "x"}]})


def test_output_validator_rejects_a_foreign_response_type() -> None:
    def check(kwargs: dict[str, Any]) -> None:
        validate = kwargs["structured_output_validator"]
        assert validate({"p01": None}) == ("response: expected session compaction",)
        assert validate(None) == ("response: expected session compaction",)

    _compact(SessionCompactor(client(check), project_id="project"))


def test_output_validator_reports_unknown_evidence_ids() -> None:
    def check(kwargs: dict[str, Any]) -> None:
        proposal = _SelectedCompaction.model_validate(
            {
                f"p{index:02d}": {
                    "category": "goal",
                    "detail": "Invented",
                    "evidence_id": "e99",
                }
                if index == 1
                else None
                for index in range(1, 25)
            }
        )
        errors = kwargs["structured_output_validator"](proposal)
        assert errors == (
            "points.0.evidence_id: copy an exact ID from source_records or previous points",
        )

    _compact(SessionCompactor(client(check), project_id="project"))


def test_output_validator_routes_semantic_apply_failures_to_repair() -> None:
    def check(kwargs: dict[str, Any]) -> None:
        proposal = _SelectedCompaction.model_validate(
            {
                f"p{index:02d}": {
                    "category": "goal",
                    "detail": "   ",
                    "evidence_id": "e1",
                }
                if index == 1
                else None
                for index in range(1, 25)
            }
        )
        assert kwargs["structured_output_validator"](proposal) == (
            "compaction contains empty evidence",
        )

    _compact(SessionCompactor(client(check), project_id="project"))


@pytest.mark.parametrize("returned", [None, {"p01": None}, "text"])
def test_provider_without_structured_compaction_is_refused(returned: object) -> None:
    fake = SimpleNamespace(
        config=SimpleNamespace(model="minimax/MiniMax-M2.7"),
        generate_chat_response=Mock(return_value=returned),
    )
    checkpoints: list[object] = []
    with pytest.raises(CompactionError, match="no structured session compaction"):
        _compact(
            SessionCompactor(fake, project_id="project"),  # type: ignore[arg-type]
            checkpoint=lambda *args: checkpoints.append(args),
        )
    assert checkpoints == []


def test_document_without_spans_is_refused_before_authority_or_provider() -> None:
    fake = client()
    authority = Mock()
    empty = replace(document(), spans=())
    with pytest.raises(CompactionError, match="no source evidence"):
        SessionCompactor(fake, project_id="project").compact(
            empty,
            through_request_id=None,
            governance_epochs=(),
            before_call=authority,
        )
    authority.assert_not_called()
    fake.generate_chat_response.assert_not_called()


def test_colliding_source_identities_are_refused_before_provider() -> None:
    original = document()
    span = original.spans[0]
    twin = replace(span, span_id="span-twin")
    doubled = replace(original, spans=(span, twin))
    fake = client()
    with pytest.raises(CompactionError, match="source selector collision"):
        SessionCompactor(fake, project_id="project").compact(
            doubled,
            through_request_id=None,
            governance_epochs=(),
            before_call=lambda: None,
        )
    fake.generate_chat_response.assert_not_called()


def test_apply_refuses_a_document_over_the_frozen_render_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = document()
    value = SessionCompaction(
        points=[
            CompactionPoint(
                category="constraint",
                detail="Keep local time",
                source_id=_source(original.spans[0]).source_id,
                exact_quote="Do not use UTC.",
            )
        ]
    )
    assert _apply(original, value).compaction_summary is not None
    monkeypatch.setattr(compaction, "MAX_COMPACTION_RENDER_TOKENS", 100)
    with pytest.raises(CompactionError, match="frozen payload reservation"):
        _apply(original, value)


def test_complete_receipt_replay_checkpoints_the_final_value() -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    replayed: list[tuple[str, int, SessionCompaction]] = []

    _compact(
        compactor,
        cached=receipt,
        checkpoint=lambda identity, batches, value: replayed.append(
            (identity, batches, value)
        ),
    )

    fake.generate_chat_response.assert_not_called()
    assert replayed == [
        (
            receipt["identity"],
            -1,
            SessionCompaction.model_validate(receipt["compaction"]),
        )
    ]


def test_cache_lookup_callable_receives_the_identity_digest() -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    asked: list[str] = []

    def lookup(digest: str) -> dict[str, Any]:
        asked.append(digest)
        return receipt

    _compact(compactor, cached=lookup)

    assert asked == [receipt["identity"]]
    fake.generate_chat_response.assert_not_called()


def test_cache_lookup_returning_nothing_compacts_from_scratch() -> None:
    _receipt, compactor, fake = _receipt_from_first_run()
    _compact(compactor, cached=lambda _digest: None)
    assert fake.generate_chat_response.call_count == 1


@pytest.mark.parametrize("complete", [True, False])
def test_tampered_receipt_is_refused_not_silently_recompacted(complete: bool) -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    tampered = {
        **receipt,
        "complete": complete,
        "batches": -1 if complete else 1,
        "compaction": {"points": []},
    }
    with pytest.raises(CompactionError, match="invalid cached session compaction"):
        _compact(compactor, cached=tampered)
    fake.generate_chat_response.assert_not_called()


def test_complete_receipt_with_a_forged_quote_is_refused() -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    points = [dict(p) for p in receipt["compaction"]["points"]]
    points[0]["exact_quote"] = "never said"
    forged = {**receipt, "compaction": {"points": points}}
    with pytest.raises(CompactionError, match="does not resolve to original"):
        _compact(compactor, cached=forged)
    fake.generate_chat_response.assert_not_called()


def test_partial_receipt_resumes_from_its_cursor_without_reprocessing() -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    partial = {**receipt, "complete": False, "batches": 1}
    checkpoints: list[int] = []

    result = _compact(
        compactor,
        cached=partial,
        checkpoint=lambda _identity, batches, _value: checkpoints.append(batches),
    )

    # The single batch was already compacted: no provider call, final checkpoint.
    fake.generate_chat_response.assert_not_called()
    assert checkpoints == [-1]
    assert result.compaction_summary is not None


def test_receipt_for_another_identity_is_ignored() -> None:
    receipt, compactor, fake = _receipt_from_first_run()
    foreign = {**receipt, "identity": "0" * 64}
    _compact(compactor, cached=foreign)
    assert fake.generate_chat_response.call_count == 1
