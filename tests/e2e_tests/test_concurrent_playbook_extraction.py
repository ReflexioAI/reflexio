"""Concurrent users must retain every admitted extraction batch.

Each user owns a durable stream and an independent org/user lease. Shared worker
capacity may delay a user, but cannot overwrite or discard that user's backlog.
Force barriers make these deliberately small batches eligible immediately.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import InteractionData
from reflexio.models.config_schema import SINGLETON_USER_PLAYBOOK_NAME
from tests.server.test_utils import skip_in_precommit, skip_low_priority

pytestmark = pytest.mark.e2e


# Three deliberately-distinct user conversations so each batch produces
# different raw playbooks (no cross-user noise). Each batch has enough
# corrective signal that the playbook extractor would happily emit at
# least one playbook on a clean run.
_BATCHES: list[list[dict]] = [
    [
        {
            "role": "User",
            "content": "I really need you to stop using bullet points -- prose only.",
        },
        {"role": "Agent", "content": "Sure, I'll switch to prose."},
        {
            "role": "User",
            "content": "Good. And don't say 'sure' constantly, it sounds robotic.",
        },
        {"role": "Agent", "content": "Understood, I'll vary my acknowledgments."},
        {
            "role": "User",
            "content": "Last thing -- always cite sources for technical claims.",
        },
        {"role": "Agent", "content": "Noted: prose, no 'sure', cite sources."},
    ],
    [
        {
            "role": "User",
            "content": "Stop summarizing my code -- just answer the question I asked.",
        },
        {"role": "Agent", "content": "I'll skip the summary."},
        {
            "role": "User",
            "content": "And give me a one-line answer first, then the explanation if I ask.",
        },
        {"role": "Agent", "content": "Got it: lead with the one-liner."},
        {
            "role": "User",
            "content": "Also, never refactor my code without asking first.",
        },
        {"role": "Agent", "content": "Confirmed: no unsolicited refactors."},
    ],
    [
        {
            "role": "User",
            "content": "When debugging, you keep guessing -- read the actual error first.",
        },
        {"role": "Agent", "content": "I'll read the error before hypothesizing."},
        {
            "role": "User",
            "content": "And don't suggest random library swaps without checking the package.",
        },
        {"role": "Agent", "content": "I'll inspect installed packages first."},
        {"role": "User", "content": "Show me the diff before applying it -- always."},
        {"role": "Agent", "content": "Acknowledged: diff first, apply after approval."},
    ],
]


def _publish_for_user(
    reflexio: Reflexio, user_id: str, agent_version: str, batch: list[dict]
) -> str:
    """Publish one user's batch and return the user_id on completion."""
    interactions = [InteractionData(**turn) for turn in batch]
    response = reflexio.publish_interaction(
        {
            "user_id": user_id,
            # Per-user session so concurrent users don't share session state —
            # this test isolates concurrent user streams.
            "session_id": f"e2e_test_session_{user_id}",
            "interaction_data_list": interactions,
            "source": "concurrent_test",
            "force_extraction": True,
            "agent_version": agent_version,
        }
    )
    assert response.success is True, f"Publish failed for {user_id}: {response.message}"
    return user_id


@skip_in_precommit
@skip_low_priority
def test_concurrent_publishes_distinct_users_all_produce_playbooks(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],  # noqa: ARG001
):
    """Three concurrent publishes for distinct users should each produce
    at least one raw playbook.

    Each user's durable admission survives any wait for shared worker capacity.
    """
    agent_version = "v_concurrent_test"
    user_ids = ["concurrent_user_a", "concurrent_user_b", "concurrent_user_c"]

    # Stagger by ~50ms so they overlap on the same lock window without
    # being literal milliseconds apart (matches the test-backend-pipeline
    # observed timing of ~2s spacing being lost).
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for uid, batch in zip(user_ids, _BATCHES, strict=True):
            futures.append(
                executor.submit(
                    _publish_for_user,
                    reflexio_instance_playbook_only,
                    uid,
                    agent_version,
                    batch,
                )
            )
            time.sleep(0.05)
        completed = [f.result(timeout=120) for f in as_completed(futures)]

    assert sorted(completed) == sorted(user_ids), (
        f"All three publishes should report success, got: {completed}"
    )

    # Per-user playbook count: each user should have produced at least
    # one raw playbook for their distinct corrective signal.
    storage = reflexio_instance_playbook_only.request_context.storage
    per_user_counts: dict[str, int] = {}
    for uid in user_ids:
        playbooks = storage.get_user_playbooks(  # type: ignore[reportOptionalMemberAccess]
            # Raw playbooks are written under the singleton name, NOT the
            # config's ``extractor_name`` ("test_playbook") — querying by the
            # extractor name matched nothing and read as "the R2 bug is back".
            user_id=uid,
            playbook_name=SINGLETON_USER_PLAYBOOK_NAME,
        )
        per_user_counts[uid] = len(playbooks)

    missing = [uid for uid, n in per_user_counts.items() if n == 0]
    assert not missing, (
        f"All three users should have >=1 raw playbook, "
        f"but {missing} have zero. "
        f"Counts: {per_user_counts}. "
        f"This is the R2 bug -- see module docstring."
    )
