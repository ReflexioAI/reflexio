"""Concurrent playbook extraction (R2 — fixed).

Three publishes for distinct user_ids land within ~2s of each other on the
shared per-org ``playbook_generation`` lock. Pre-fix:

  - The first acquires the lock.
  - The second and third lose the race; each writes its request_id into
    ``pending_request_id`` (single slot), with the third overwriting the
    second.
  - When the first finishes, ``release_lock`` returns the third request_id,
    but the rerun loop re-uses the FIRST user's request payload — so the
    bookmark advances past users 2/3's interactions and they never get
    extracted.

Post-fix (option (b) from #59): ``pending_request_id`` is replaced by a
FIFO ``pending_request_queue`` whose entries carry the original request
payload. The drain loop pops one at a time and re-runs ``_run_generation``
against THAT request (not the holder's). All three users now see at least
one raw playbook generated for their distinct corrective signal.

The lock remains per-org so cross-user feedback dedup invariants
(see playbook_consolidator) are unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.service_schemas import InteractionData
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
            "interaction_data_list": interactions,
            "source": "concurrent_test",
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

    Post-fix: the pending-request queue preserves each blocked publish's
    payload, so the drain loop reruns extraction against the queued
    user's interactions instead of the holder's. All three users get at
    least one raw playbook.
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
            user_id=uid, playbook_name="test_playbook"
        )
        per_user_counts[uid] = len(playbooks)

    missing = [uid for uid, n in per_user_counts.items() if n == 0]
    assert not missing, (
        f"All three users should have >=1 raw playbook, "
        f"but {missing} have zero. "
        f"Counts: {per_user_counts}. "
        f"This is the R2 bug -- see module docstring."
    )
