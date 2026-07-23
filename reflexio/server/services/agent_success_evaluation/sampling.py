"""Per-family sampling for group evaluation.

A scheduled group evaluation runs two independent judge families over the same
session: **agent success** (one session-level verdict) and **retrieved learning**
(one relevance/impact verdict per served learning). They are sampled separately,
because their consumers need different coverage: the retrieved-learning verdicts
feed the offline playbook tuner, which needs enough sessions per playbook to
clear its evidence floors, while the session-success judge is a product signal
that is fine at a low rate.

This module is imported by exactly ONE caller: the publish scheduler gate in
`generation_service`. It samples both families once, admits the session if either
says yes, and passes the two booleans to the runner. The runner never samples —
it is TOLD. That is stronger than sharing a helper: with a single computation
site, the two families cannot disagree because there is nothing to disagree with.

Direct callers of the runner (regen jobs, the on-demand grade route) leave both
flags at their default and run both families, as before. Sampling is a scheduling
decision, not a runner one.
"""

from __future__ import annotations

import hashlib
from typing import Any


def stable_group_sampling_fraction(org_id: str, user_id: str, session_id: str) -> float:
    """Return a deterministic [0, 1) sample value for one session."""
    key = f"{org_id}\0{user_id}\0{session_id}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _samples(rate: float, fraction: float) -> bool:
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return fraction < rate


def samples_agent_success(
    agent_success_config: Any | None,
    *,
    org_id: str,
    user_id: str,
    session_id: str,
    evaluation_only: bool = False,
) -> bool:
    """Whether this session is sampled for the session-success judge."""
    if agent_success_config is None:
        return False
    rate = agent_success_config.sampling_rate
    if (
        evaluation_only
        and agent_success_config.evaluation_only_sampling_rate is not None
    ):
        rate = agent_success_config.evaluation_only_sampling_rate
    return _samples(
        float(rate),
        stable_group_sampling_fraction(org_id, user_id, session_id),
    )


def samples_retrieved_learning(
    agent_success_config: Any | None,
    *,
    org_id: str,
    user_id: str,
    session_id: str,
) -> bool:
    """Whether this session is sampled for the retrieved-learning judge.

    ``retrieved_learning_sampling_rate=None`` inherits ``sampling_rate``, so an
    org that has not opted in keeps exactly its previous behavior.
    """
    if agent_success_config is None:
        return False
    rate = getattr(agent_success_config, "retrieved_learning_sampling_rate", None)
    if rate is None:
        rate = agent_success_config.sampling_rate
    return _samples(
        float(rate),
        stable_group_sampling_fraction(org_id, user_id, session_id),
    )
