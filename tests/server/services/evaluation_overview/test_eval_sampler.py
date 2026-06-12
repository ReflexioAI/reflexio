"""Unit tests for the pure F3 stratified sampler."""

import random

import pytest

from reflexio.server.services.evaluation_overview.eval_sampler import (
    SampleCandidate,
    sample_candidates,
    sample_per_stratum,
    stratify_by_day_and_source,
)


def _cand(sid: str, ts: int, first_request_source: str) -> SampleCandidate:
    return SampleCandidate(
        session_id=sid,
        user_id="u1",
        agent_version="v1",
        source=None,
        created_at=ts,
        first_request_source=first_request_source,
    )


# --- stratify_by_day_and_source -------------------------------------------


def test_stratify_groups_by_day_and_first_request_source():
    base = 1_700_000_000
    day_two = base + 86_400
    cands = [
        _cand("s1", base, "candidate"),
        _cand("s2", base, "candidate"),
        _cand("s3", base, "baseline"),
        _cand("s4", day_two, "candidate"),
        _cand("s5", base, ""),
    ]
    strata = stratify_by_day_and_source(cands)
    assert len(strata) == 4
    base_day = (base // 86_400) * 86_400
    day_two_day = (day_two // 86_400) * 86_400
    assert (base_day, "candidate") in strata
    assert (base_day, "baseline") in strata
    assert (base_day, "") in strata
    assert (day_two_day, "candidate") in strata
    assert {c.session_id for c in strata[(base_day, "candidate")]} == {
        "s1",
        "s2",
    }


def test_stratify_empty_input():
    assert stratify_by_day_and_source([]) == {}


# --- sample_per_stratum ---------------------------------------------------


def test_sample_per_stratum_caps_at_n():
    rng = random.Random(0)  # noqa: S311
    base = 1_700_000_000
    cands = [_cand(f"s{i}", base, "candidate") for i in range(50)]
    sampled = sample_per_stratum(cands, n=10, rng=rng)
    assert len(sampled) == 10
    assert {c.session_id for c in sampled}.issubset({f"s{i}" for i in range(50)})


def test_sample_per_stratum_returns_all_when_below_n():
    rng = random.Random(0)  # noqa: S311
    base = 1_700_000_000
    cands = [_cand(f"s{i}", base, "candidate") for i in range(5)]
    sampled = sample_per_stratum(cands, n=10, rng=rng)
    assert len(sampled) == 5
    assert {c.session_id for c in sampled} == {f"s{i}" for i in range(5)}


def test_sample_per_stratum_is_seeded_for_reproducibility():
    base = 1_700_000_000
    cands = [_cand(f"s{i}", base, "candidate") for i in range(50)]
    a = sample_per_stratum(cands, n=10, rng=random.Random(42))  # noqa: S311
    b = sample_per_stratum(cands, n=10, rng=random.Random(42))  # noqa: S311
    assert [c.session_id for c in a] == [c.session_id for c in b]


def test_sample_per_stratum_rejects_non_positive_n():
    with pytest.raises(ValueError):
        sample_per_stratum([], n=0, rng=random.Random(0))  # noqa: S311
    with pytest.raises(ValueError):
        sample_per_stratum([], n=-1, rng=random.Random(0))  # noqa: S311


# --- sample_candidates ----------------------------------------------------


def test_sample_candidates_strata_independent():
    """Sampling caps per stratum; not in aggregate."""
    rng = random.Random(0)  # noqa: S311
    base = 1_700_000_000
    cands = [_cand(f"t{i}", base, "candidate") for i in range(20)] + [
        _cand(f"c{i}", base, "baseline") for i in range(20)
    ]
    sampled = sample_candidates(cands, n_per_stratum=5, rng=rng)
    assert len(sampled) == 10  # 5 from each of 2 strata


def test_sample_candidates_empty_input_returns_empty():
    rng = random.Random(0)  # noqa: S311
    assert sample_candidates([], n_per_stratum=10, rng=rng) == []
