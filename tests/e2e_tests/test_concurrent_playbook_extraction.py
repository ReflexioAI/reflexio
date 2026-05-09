"""Concurrent playbook extraction repro (reflexio-enterprise#59 / R2).

This test reproduces the bug observed in the test-backend-pipeline run:

  - Three publishes for distinct user_ids land within ~2s of each other.
  - The first acquires the per-org playbook_generation lock.
  - The second and third lose the race; each writes its request_id into
    pending_request_id, with the third overwriting the second.
  - When the first finishes, release_lock returns the third request_id.
  - The base run() loop re-runs with the original request payload (not
    the third user's) -- so users 2 and 3 never get their interactions
    extracted.
  - Only the first user produces a raw playbook.

Marked xfail until reflexio-enterprise#59 is resolved. The architectural
options being weighed in #59 are roughly:

  (a) Make the playbook lock per-user (matching profiles), trading
      cross-user dedup for parallelism.
  (b) Switch ``pending_request_id`` from "single slot, last-wins" to a
      queue, AND have the rerun loop iterate over distinct queued
      payloads -- not the original request.
  (c) Detach extraction from the publish-time lock entirely and run it
      from a periodic worker that scans for any unprocessed
      interactions.

(a) is the smallest change but breaks the cross-user dedup invariant.
(b) preserves dedup but is the largest refactor. (c) is the cleanest
long-term answer but needs a reliable scheduler.

The test passes once any of those is in place AND all three users see
at least one raw playbook generated for their distinct interactions.
"""

# TODO(reflexio-enterprise#59): remove xfail once the per-user lock or
# pending-queue refactor lands. The decision among (a)/(b)/(c) above
# should be made by an architect-level review, not bolted on as a fix.

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
@pytest.mark.xfail(
    reason=(
        "R2: concurrent publishes for distinct users on the per-org "
        "playbook lock lose extraction for everyone but the first. "
        "Tracked as reflexio-enterprise#59. Options: per-user lock, "
        "pending-id queue, or detached worker. See module docstring."
    ),
    strict=False,
)
def test_concurrent_publishes_distinct_users_all_produce_playbooks(
    reflexio_instance_playbook_only: Reflexio,
    cleanup_playbook_only: Callable[[], None],  # noqa: ARG001
):
    """Three concurrent publishes for distinct users should each produce
    at least one raw playbook.

    The current implementation fails this assertion: typically only the
    first user's batch produces a raw playbook, because the other two
    lose the per-org playbook_generation lock and the rerun-on-pending
    loop re-executes with the FIRST request's payload (different
    user_id), not the queued ones.
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
