"""Tests for the golden-set search eval harness (mocked, CI-covered).

Seeds real temp SQLite stores and runs the REAL classic unified search in
FTS mode (deterministic without embeddings or API keys; the global litellm
mock covers any stray LLM call). The judge is exercised only through the
stubbed ``search_judge`` fixture — real-LLM scoring lives in
``tests/e2e_tests/test_search_eval_real_llm.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from reflexio.server.api_endpoints.request_context import RequestContext
from tests.eval.conftest import _load
from tests.eval.search.providers import (
    ProviderRun,
    make_classic_search_provider,
    seed_case_entities,
)
from tests.eval.search.runner import (
    CaseOutcome,
    compute_mechanical,
    run_eval,
    score_case,
)

pytestmark = pytest.mark.integration

_CATEGORIES = {
    "recall",
    "preference",
    "supersession",
    "temporal_current",
    "temporal_window",
}


def _cases() -> list[dict[str, Any]]:
    return _load("search")


@pytest.fixture
def classic_provider(tmp_path):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    return make_classic_search_provider(
        storage_base_dir=str(tmp_path),
        llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
    )


# ---------------------------------------------------------------------------
# Golden case hygiene.
# ---------------------------------------------------------------------------


def test_every_case_has_category_and_seeds():
    for case in _cases():
        assert case.get("category") in _CATEGORIES, case["id"]
        seeded = (
            case.get("seeded_profiles", [])
            + case.get("seeded_user_playbooks", [])
            + case.get("seeded_agent_playbooks", [])
        )
        assert seeded, f"{case['id']} seeds no entities"
        keys = [spec["key"] for spec in seeded]
        assert len(keys) == len(set(keys)), f"{case['id']} has duplicate keys"
        for spec in seeded:
            assert "age_days" in spec, f"{case['id']}:{spec['key']} missing age_days"
        labeled = case.get("expected_top_candidates", []) + case.get(
            "must_NOT_rank_first", []
        )
        for key in labeled:
            assert key in keys, f"{case['id']} labels unseeded key {key}"


def test_temporal_categories_are_represented():
    categories = {case.get("category") for case in _cases()}
    assert {"temporal_current", "temporal_window", "supersession"} <= categories


def test_expected_time_window_uses_reformulation_field_names():
    """``expected_time_window`` keys must be ``ReformulationResult`` fields —
    the real-LLM eval asserts them via ``getattr`` on the reformulation."""
    valid_bounds = {"start_days_ago", "end_days_ago"}
    for case in _cases():
        window = case.get("expected_time_window") or {}
        unknown = set(window) - valid_bounds
        assert not unknown, f"{case['id']}: unknown time-window keys {unknown}"


# ---------------------------------------------------------------------------
# Seeding round-trips controlled timestamps.
# ---------------------------------------------------------------------------


def test_seeding_round_trips_timestamps(tmp_path):
    ctx = RequestContext(org_id="eval-seed-rt", storage_base_dir=str(tmp_path))
    storage = ctx.storage
    assert storage is not None
    now = int(datetime.now(UTC).timestamp())
    case = {
        "id": "seed-rt",
        "seeded_profiles": [
            {"key": "p1", "user_id": "u1", "content": "alpha", "age_days": 90}
        ],
        "seeded_user_playbooks": [
            {
                "key": "b1",
                "user_id": "u1",
                "trigger": "t",
                "content": "beta",
                "age_days": 30,
            }
        ],
    }
    key_to_id = seed_case_entities(storage, case, now)

    profiles = storage.get_profiles_by_ids("u1", ["p1"])
    assert len(profiles) == 1
    assert profiles[0].last_modified_timestamp == now - 90 * 86_400

    assert key_to_id["b1"].isdigit() and int(key_to_id["b1"]) > 0


# ---------------------------------------------------------------------------
# Mechanical metrics unit coverage (synthetic response, no storage).
# ---------------------------------------------------------------------------


def _fake_response(profile_ids: list[str]) -> Any:
    profiles = [
        SimpleNamespace(profile_id=pid, content=f"c-{pid}", last_modified_timestamp=1)
        for pid in profile_ids
    ]
    return SimpleNamespace(profiles=profiles, user_playbooks=[], agent_playbooks=[])


_SYNTHETIC_CASE = {
    "id": "synthetic",
    "category": "recall",
    "query": "q",
    "seeded_profiles": [
        {"key": "p_good", "content": "g", "age_days": 1},
        {"key": "p_bad", "content": "b", "age_days": 9},
    ],
    "expected_top_candidates": ["p_good"],
    "must_NOT_rank_first": ["p_bad"],
}
_SYNTHETIC_MAP = {"p_good": "p_good", "p_bad": "p_bad"}


def test_mechanical_metrics_hit_at_rank_two():
    metrics = compute_mechanical(
        _SYNTHETIC_CASE, _fake_response(["p_bad", "p_good"]), _SYNTHETIC_MAP
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == pytest.approx(0.5)
    assert metrics.must_not_first_violated is True


def test_mechanical_metrics_perfect_ranking():
    metrics = compute_mechanical(
        _SYNTHETIC_CASE, _fake_response(["p_good", "p_bad"]), _SYNTHETIC_MAP
    )
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.must_not_first_violated is False


def test_mechanical_metrics_total_miss():
    metrics = compute_mechanical(_SYNTHETIC_CASE, _fake_response([]), _SYNTHETIC_MAP)
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == 0.0
    assert metrics.must_not_first_violated is False


# ---------------------------------------------------------------------------
# Classic provider over the real golden set (mechanical-only).
# ---------------------------------------------------------------------------


def test_classic_provider_runs_all_golden_cases(classic_provider):
    cases = _cases()
    results = run_eval(cases=cases, provider=classic_provider, backend="classic")

    assert results.n == len(cases)
    for outcome in results.outcomes:
        assert 0.0 <= outcome.recall_at_k <= 1.0
        assert 0.0 <= outcome.mrr <= 1.0
        assert outcome.latency_ms >= 0.0
        assert outcome.backend == "classic"

    summary = results.summary()
    assert "Search golden-set eval summary" in summary
    for category in sorted({str(c.get("category", "")) for c in cases}):
        assert f"[{category}]" in summary


def test_classic_passes_direct_recall(classic_provider):
    case = next(c for c in _cases() if c["id"] == "direct_recall")
    run = classic_provider(case)
    outcome = score_case(case=case, run=run, backend="classic")
    assert outcome.recall_at_k == 1.0, "FTS should find the direct lexical match"
    assert outcome.must_not_first_violated is False


# ---------------------------------------------------------------------------
# Temporal-classic provider structural check under the global litellm mock.
# ---------------------------------------------------------------------------


def test_reformulation_provider_structural_under_mock(tmp_path):
    """The reformulation-enabled provider runs end-to-end under the global
    mock: the canned ReformulationResult (no temporal signals) flows through
    the pipeline and a response assembles. Result QUALITY is meaningless
    here (mocked LLM) — quality is measured in the real-LLM e2e eval.
    """
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from tests.eval.search.providers import make_classic_search_provider

    provider = make_classic_search_provider(
        storage_base_dir=str(tmp_path),
        llm_client=LiteLLMClient(LiteLLMConfig(model="claude-sonnet-4-6")),
        enable_reformulation=True,
    )
    case = next(c for c in _cases() if c["id"] == "direct_recall")
    run = provider(case)

    assert run.response.success is True
    outcome = score_case(case=case, run=run, backend="temporal-classic")
    assert outcome.backend == "temporal-classic"


# ---------------------------------------------------------------------------
# Judged path through the stubbed judge fixture (parametrized per case).
# ---------------------------------------------------------------------------


def test_score_golden_case_with_judge(search_case, search_judge):
    run = ProviderRun(response=_fake_response([]), key_to_id={}, latency_ms=1.0)
    outcome = score_case(
        case=search_case, run=run, backend="classic", judge=search_judge
    )
    assert isinstance(outcome, CaseOutcome)
    assert outcome.case_id == search_case["id"]
    assert outcome.category == search_case["category"]
    assert outcome.answer_correctness is not None
    assert outcome.rationale
