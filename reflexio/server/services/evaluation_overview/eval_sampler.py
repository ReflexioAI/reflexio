"""Pure stratified sampler for the F3 eval-regen pipeline.

DB-free and LLM-free. Operates on SampleCandidate tuples emitted by the
regen worker's candidate-discovery step; returns the sampled subset.

Stratification key: (day_bucket, first_request_source). Day buckets are
aligned to 86400-second boundaries. The source value is sticky per session
and comes from the session's earliest request.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

# Day bucket width in seconds. Strata are aligned to (ts // _DAY) * _DAY.
_DAY = 86_400


@dataclass(frozen=True)
class SampleCandidate:
    """One candidate for the regen pipeline's sampling step.

    Carries everything the regen worker needs to dispatch a per-session
    evaluation, plus the first-request source used for stratification.

    Attributes:
        session_id: The session to evaluate.
        user_id: Owner of the session's requests.
        agent_version: Agent version to filter on.
        source: Optional source tag (passed through to run_group_evaluation).
        created_at: Unix epoch seconds of the session's most relevant
            timestamp; used for day-bucket alignment.
        first_request_source: Source on the session's first request, read once
            during candidate discovery. Empty string is a real stratum.
    """

    session_id: str
    user_id: str
    agent_version: str
    source: str | None
    created_at: int
    first_request_source: str


def stratify_by_day_and_source(
    candidates: Iterable[SampleCandidate],
) -> dict[tuple[int, str], list[SampleCandidate]]:
    """Bucket candidates by (day_bucket, first_request_source).

    Args:
        candidates: Candidate tuples emitted by the regen candidate-discovery
            step.

    Returns:
        Dict from (day_bucket_start, source) to the candidate list for that
        stratum. Empty strata are absent; the dict has only populated keys.
    """
    strata: dict[tuple[int, str], list[SampleCandidate]] = defaultdict(list)
    for c in candidates:
        day_bucket = (c.created_at // _DAY) * _DAY
        strata[(day_bucket, c.first_request_source)].append(c)
    return dict(strata)


def sample_per_stratum(
    candidates: list[SampleCandidate],
    n: int,
    rng: random.Random,
) -> list[SampleCandidate]:
    """Sample up to ``n`` candidates from a single stratum.

    Args:
        candidates: All candidates in one stratum.
        n: Cap. Strata with fewer than ``n`` items are returned whole.
        rng: Seeded random for reproducibility.

    Returns:
        Sampled subset. Order is ``random.Random.sample``'s implementation
        detail — callers should not depend on it.

    Raises:
        ValueError: When ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if len(candidates) <= n:
        return list(candidates)
    return rng.sample(candidates, n)


def sample_candidates(
    candidates: Iterable[SampleCandidate],
    n_per_stratum: int,
    rng: random.Random,
) -> list[SampleCandidate]:
    """Stratify by (day x source), then sample per stratum, then flatten.

    Predictable cost across runs: at most ``n_per_stratum * num_strata``
    items returned, regardless of traffic volume.

    Args:
        candidates: All candidates in the regen window.
        n_per_stratum: Per-stratum cap (typically
            ``Config.eval_sample_n_per_stratum``).
        rng: Seeded random for reproducibility.

    Returns:
        Flattened sampled list.
    """
    strata = stratify_by_day_and_source(candidates)
    sampled: list[SampleCandidate] = []
    for stratum_candidates in strata.values():
        sampled.extend(sample_per_stratum(stratum_candidates, n_per_stratum, rng))
    return sampled
