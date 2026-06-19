"""End-to-end test for write-time contradiction resolution.

Verifies the load-bearing invariant of the reflection-extraction-polarity
feature at the consolidator boundary:

When an EXISTING positive ``UserPlaybook`` collides with a NEW
failure-path-derived NEGATIVE candidate on the same trigger (a
same-situation contradiction), the consolidator MUST route the pair
through a contradiction-aware decision (``RejectNewDecision`` or
``DifferentiateDecision``) and MUST NOT ``unify`` them into one
self-contradicting skill. Under Option B (consolidator-compose) this
no-self-contradiction judgment is made by the LLM in the consolidation
prompt, not by a mechanical apply-time polarity validator. After the
generation-service apply path runs (``delete_user_playbooks_by_ids``
followed by ``save_user_playbooks``), storage MUST NOT contain two
current rows on the same trigger with opposing polarity — the two rules
must never co-exist.

This e2e test pairs with the integration-level coverage in
``tests/server/services/playbook/test_playbook_consolidator_integration.py``
(E3/E4). The integration tests cover each apply branch with focused
fixtures; this e2e test rides the full apply pipeline against real
``SQLiteStorage`` from a temp directory, so the post-state assertions
ride the same storage round-trips as production.

The full publish→extract→consolidate pipeline is deliberately
short-circuited:

* Phase 1 (extraction stand-in): the existing positive playbook is
  persisted directly; the failure-path candidate is constructed in
  memory. Extractor-side polarity threading is covered exhaustively by
  C3/D6 integration tests.
* Phase 2 (consolidation): the real ``PlaybookConsolidator`` is invoked
  with a scripted LLM response covering the LLM-driven resolutions of a
  same-situation contradiction pair under Option B:
  ``RejectNewDecision`` (the existing positive wins) and
  ``DifferentiateDecision`` (the two rules refine onto disjoint
  triggers). The LLM must NOT ``unify`` a same-situation contradiction
  (that would make the skill self-contradict); the apply layer no longer
  enforces a mechanical polarity validator, so the guarantee is now
  LLM-judged.
* Phase 3 (apply): the generation-service apply flow
  (``delete_user_playbooks_by_ids`` → ``save_user_playbooks``) is
  replicated, then storage is queried directly.

The LLM is mocked per test because e2e tests bypass the global
``litellm.completion`` mock.
"""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.playbook_consolidator import (
    DifferentiateDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
    RejectNewDecision,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.e2e


def _is_avoidance(content: str) -> bool:
    """Local wording check: does this row read as negative/avoidance advice?

    Trivial fixture-wording predicate over the test's controlled (mocked/
    seeded) content — NOT a general polarity classifier. Orientation here is
    purely a wording convention; negative rows are phrased as avoidance.
    """
    return content.lstrip().startswith(("Avoid", "Do not", "Don't", "Never"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_dir():
    """Create an isolated temp directory for SQLite storage."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def request_context(temp_storage_dir, worker_id):
    """Real ``RequestContext`` backed by real SQLite storage in a temp dir.

    Embeddings are patched out so storage doesn't try to call out to an
    embedding model during ``save_user_playbooks``. The prompt manager is
    mocked because the consolidator renders a prompt before calling the
    (mocked) LLM client.

    Args:
        temp_storage_dir (str): Pytest-managed temp directory path.
        worker_id (str): pytest-xdist worker id; ensures per-worker org id
            so parallel runs don't collide.

    Yields:
        RequestContext: A context wired to real SQLite storage, with
        embeddings and the prompt manager mocked out.
    """
    with (
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
        patch.object(
            LiteLLMClient,
            "get_embeddings",
            side_effect=lambda texts, *_args, **_kwargs: [[0.0] * 512 for _ in texts],
        ),
    ):
        ctx = RequestContext(
            org_id=f"contradiction_resolution_e2e_{worker_id}",
            storage_base_dir=temp_storage_dir,
        )
        ctx.prompt_manager = MagicMock()
        ctx.prompt_manager.render_prompt.return_value = "mock prompt"
        yield ctx


@pytest.fixture
def llm_client():
    """Mock LLM client whose ``generate_chat_response`` is scripted per test."""
    return MagicMock(spec=LiteLLMClient)


@pytest.fixture
def consolidator(request_context, llm_client) -> PlaybookConsolidator:
    """Real ``PlaybookConsolidator`` wired to real storage + mock LLM client."""
    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as mock_svm:
        mock_svm.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-test"
        }
        return PlaybookConsolidator(
            request_context=request_context, llm_client=llm_client
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_positive_playbook(
    storage: BaseStorage,
    *,
    user_id: str,
    trigger: str = "when user asks about product X",
) -> UserPlaybook:
    """Persist a positive playbook as if a prior extraction had emitted it.

    Args:
        storage: Real ``BaseStorage`` instance (SQLite in this e2e test).
        user_id (str): User scope for the playbook.
        trigger (str): Trigger string scoping the rule.

    Returns:
        UserPlaybook: The persisted row with its storage-assigned id.
    """
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v1",
        request_id="existing_req",
        playbook_name="default",
        content="Recommend product X — past sessions showed strong interest.",
        trigger=trigger,
        rationale="Earlier session ended in a successful conversion on X.",
        source="api",
        source_interaction_ids=[],
    )
    storage.save_user_playbooks([pb])
    rows = storage.get_user_playbooks(user_id=user_id)
    assert len(rows) == 1, "Setup failure — expected one seeded existing playbook"
    return rows[0]


def _build_failure_path_negative_candidate(
    *,
    user_id: str,
    trigger: str = "when user asks about product X",
) -> UserPlaybook:
    """Construct a NEW failure-path negative candidate on the same trigger.

    Models the output of the failure-path extractor (covered end-to-end by
    D6 / C3 integration tests): a clear user-pushback window yields a
    candidate ``UserPlaybook`` with ``polarity="negative"`` and an
    ``Avoid``-prefixed body, on the SAME trigger as an existing positive
    rule.

    Args:
        user_id (str): User scope for the candidate.
        trigger (str): Trigger string — must match the existing rule so
            the consolidator sees a contradiction.

    Returns:
        UserPlaybook: A fresh, not-yet-persisted candidate.
    """
    return UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v1",
        request_id="failure_req",
        playbook_name="default",
        content="Avoid suggesting product X — the user said no twice today.",
        trigger=trigger,
        rationale="User pushed back: 'Stop suggesting X, I told you no twice.'",
        source="api",
        source_interaction_ids=[],
    )


def _drive_consolidator(
    consolidator: PlaybookConsolidator,
    *,
    candidates: list[UserPlaybook],
    existing_playbooks: list[UserPlaybook],
    decisions: list,
    request_id: str = "consolidation_req",
) -> tuple[list[UserPlaybook], list[int]]:
    """Run ``deduplicate`` with a scripted LLM and pre-fetched existing rows.

    Patches ``_retrieve_existing_playbooks`` so the decision can reference
    ``EXISTING-0`` without depending on the search backend's ranking
    behaviour. Mirrors ``_run_consolidator`` in the integration tests.

    Args:
        consolidator: Real ``PlaybookConsolidator`` instance.
        candidates: Candidate ``UserPlaybook`` rows.
        existing_playbooks: Pre-fetched existing rows (positional
            ``EXISTING-N`` ids).
        decisions: Decisions returned by the scripted LLM.
        request_id (str): Request id forwarded to ``deduplicate``.

    Returns:
        tuple[list[UserPlaybook], list[int]]: ``(rows_to_save,
        ids_to_delete)`` as returned by ``deduplicate``.
    """
    consolidator.client.generate_chat_response.return_value = (  # type: ignore[attr-defined]
        PlaybookConsolidationOutput(decisions=decisions)
    )
    with (
        patch.object(
            consolidator,
            "_retrieve_existing_playbooks",
            return_value=existing_playbooks,
        ),
        patch.dict("os.environ", {"MOCK_LLM_RESPONSE": "false"}),
    ):
        rows, archive_ids, _merge_groups = consolidator.deduplicate(
            results=[candidates],
            request_id=request_id,
            agent_version="v1",
        )
    return rows, archive_ids


def _apply_to_storage(
    storage: BaseStorage,
    rows_to_save: list[UserPlaybook],
    ids_to_delete: list[int],
) -> None:
    """Replicate the generation-service apply: delete then save.

    Args:
        storage: Real ``BaseStorage`` handle (SQLite in this e2e test).
        rows_to_save: Rows produced by the consolidator.
        ids_to_delete: Existing ids the consolidator chose to archive.
    """
    if ids_to_delete:
        storage.delete_user_playbooks_by_ids(ids_to_delete)
    if rows_to_save:
        storage.save_user_playbooks(rows_to_save)


def _assert_no_opposing_polarity_on_same_trigger(
    surviving: list[UserPlaybook],
) -> None:
    """Assert no two surviving rows share a trigger with opposing polarity.

    This is the load-bearing post-condition for write-time contradiction
    resolution: the consolidator's job is to ensure that opposing-polarity
    rules with the same trigger never co-exist in current storage. Rows
    with a ``None`` trigger are skipped — they don't participate in a
    same-trigger collision.

    Args:
        surviving: Current ``UserPlaybook`` rows after apply.

    Raises:
        AssertionError: If any two surviving rows share a non-empty
            trigger but disagree on polarity.
    """
    triggers_seen: dict[str, set[bool]] = {}
    for pb in surviving:
        if pb.trigger is None:
            continue
        orientations = triggers_seen.setdefault(pb.trigger, set())
        orientations.add(_is_avoidance(pb.content))
    for trigger, orientations in triggers_seen.items():
        assert len(orientations) <= 1, (
            f"contradiction-resolution invariant violated: trigger {trigger!r} "
            f"has surviving rows that mix avoidance and non-avoidance wording"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_existing_positive_plus_failure_path_resolves_via_reject_new(
    request_context: RequestContext,
    consolidator: PlaybookConsolidator,
):
    """Existing positive + failure-path negative ⇒ ``reject_new`` keeps positive.

    Legitimate 4-kind resolution where the LLM judges the prior positive
    rule still applies and the new negative observation is one-off noise:

    1. Seed an EXISTING positive playbook on trigger T.
    2. Construct a NEW failure-path-derived negative candidate on the
       same trigger T.
    3. Drive the real consolidator with a scripted LLM that returns a
       ``RejectNewDecision`` naming the existing as superseding.
    4. Apply the consolidator output to storage and assert:

       * Storage is unchanged; the existing positive row remains.
       * The negative candidate did NOT leak in via the safety fallback.
       * No surviving row pairs share a trigger with opposing polarity.
    """
    storage = request_context.storage
    assert storage is not None, "RequestContext must provide SQLite storage"
    user_id = "u_contradiction_reject_new"
    trigger = "when user asks about product X"

    # Phase 1 — extraction stand-in: seed the existing positive playbook.
    existing = _seed_positive_playbook(storage, user_id=user_id, trigger=trigger)
    assert existing.content.lstrip().startswith("Recommend"), (
        "Setup failure — seeded playbook must use positive (Recommend) wording; "
        f"got {existing.content!r}"
    )

    # Phase 1 (cont.) — construct the failure-path negative candidate on
    # the same trigger.
    candidate = _build_failure_path_negative_candidate(user_id=user_id, trigger=trigger)
    assert candidate.content.lstrip().startswith("Avoid"), (
        "Setup failure — failure-path candidate must use avoidance wording; "
        f"got {candidate.content!r}"
    )

    # Phase 2 — drive the real consolidator with a ``RejectNewDecision``.
    # Same trigger, opposite polarity: the existing rule wins; the new
    # candidate is dropped without archiving anything.
    rows, archive_ids = _drive_consolidator(
        consolidator,
        candidates=[candidate],
        existing_playbooks=[existing],
        decisions=[
            RejectNewDecision(
                new_id="NEW-0",
                superseded_by_existing_id=existing.user_playbook_id,
                reason=(
                    "storage-stability tie-break: prior positive rule still applies"
                ),
            )
        ],
    )

    # Consolidator contract: no row emitted, no archive — pure no-op.
    assert archive_ids == [], (
        f"reject_new must NOT archive the existing row; got {archive_ids!r}"
    )
    assert rows == [], (
        f"reject_new must NOT emit any new row; got {[r.content for r in rows]}"
    )

    # Phase 3 — apply to storage and inspect the post-state directly.
    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)

    # Storage post-state: the seeded positive row is still the only row.
    assert len(surviving) == 1, (
        "exactly one row must survive reject_new resolution; got "
        f"{[(r.user_playbook_id, r.content) for r in surviving]}"
    )
    survivor = surviving[0]
    assert survivor.user_playbook_id == existing.user_playbook_id, (
        "surviving row must be the original existing playbook"
    )
    assert survivor.content.lstrip().startswith("Recommend")
    assert survivor.trigger == trigger
    assert survivor.content == existing.content

    # Load-bearing invariant: no two current rows share a trigger with
    # opposing polarity. Asserting it here independent of row counts
    # documents the contract; the assertion above would catch any
    # regression that left both rows current, but this makes the intent
    # explicit.
    _assert_no_opposing_polarity_on_same_trigger(surviving)


def test_existing_positive_plus_failure_path_does_not_self_contradict_via_unify(
    request_context: RequestContext,
    consolidator: PlaybookConsolidator,
):
    """A same-situation contradiction is resolved by the LLM's decision kind, not ``unify``.

    Under Option B (consolidator-compose) the no-self-contradiction guard
    moved from a mechanical apply-time polarity validator to the
    consolidation prompt: the LLM must NOT route a same-trigger
    opposite-advice pair through ``unify`` (that would make the skill
    contradict itself on the same situation) and instead chooses
    ``reject_new`` (or ``differentiate``). This test drives the LLM-reported
    ``reject_new`` resolution and asserts the load-bearing post-state
    invariant: storage never holds two current rows on the same trigger with
    opposing polarity. The mechanical guarantee is now LLM-judged (expected
    per Option B).
    """
    storage = request_context.storage
    assert storage is not None, "RequestContext must provide SQLite storage"
    user_id = "u_contradiction_no_self_contradict"
    trigger = "when user asks about product X"

    existing = _seed_positive_playbook(storage, user_id=user_id, trigger=trigger)
    candidate = _build_failure_path_negative_candidate(user_id=user_id, trigger=trigger)

    # The LLM judges the pair a same-situation contradiction and routes it
    # through ``reject_new`` rather than self-contradicting via ``unify``.
    rows, archive_ids = _drive_consolidator(
        consolidator,
        candidates=[candidate],
        existing_playbooks=[existing],
        decisions=[
            RejectNewDecision(
                new_id="NEW-0",
                superseded_by_existing_id=existing.user_playbook_id,
                reason="same-situation contradiction — prior positive rule wins",
            )
        ],
    )

    # reject_new is a pure no-op: no row emitted, no archive.
    assert rows == [], (
        f"reject_new must NOT produce a row; got {[r.content for r in rows]}"
    )
    assert archive_ids == [], (
        f"reject_new must NOT archive the existing row; got {archive_ids!r}"
    )

    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)

    # Storage post-state: the seeded positive row remains, and the negative
    # candidate did NOT leak in via the safety fallback.
    assert len(surviving) == 1, (
        "exactly one row must survive same-situation contradiction resolution; got "
        f"{[(r.user_playbook_id, r.content) for r in surviving]}"
    )
    assert surviving[0].user_playbook_id == existing.user_playbook_id
    assert surviving[0].content.lstrip().startswith("Recommend")

    _assert_no_opposing_polarity_on_same_trigger(surviving)


def test_existing_positive_plus_failure_path_resolves_via_differentiate(
    request_context: RequestContext,
    consolidator: PlaybookConsolidator,
):
    """Existing positive + failure-path negative ⇒ ``differentiate`` refines both triggers.

    The alternative legitimate resolution path: the LLM decides the two
    rules cover different sub-cases and refines BOTH triggers so they no
    longer collide. The original is archived; two new rows are emitted
    with disjoint refined triggers — so the invariant ("no opposing
    polarities on the same trigger") still holds.
    """
    storage = request_context.storage
    assert storage is not None, "RequestContext must provide SQLite storage"
    user_id = "u_contradiction_differentiate"
    trigger = "when user asks about product X"

    existing = _seed_positive_playbook(storage, user_id=user_id, trigger=trigger)
    candidate = _build_failure_path_negative_candidate(user_id=user_id, trigger=trigger)

    rows, archive_ids = _drive_consolidator(
        consolidator,
        candidates=[candidate],
        existing_playbooks=[existing],
        decisions=[
            DifferentiateDecision(
                new_id="NEW-0",
                existing_id=existing.user_playbook_id,
                refined_new_trigger=(
                    "when user asks about product X AND has declined it earlier today"
                ),
                refined_existing_trigger=(
                    "when user asks about product X AND has not declined it recently"
                ),
            )
        ],
    )

    # The original is archived and two refined rows are emitted.
    assert archive_ids == [existing.user_playbook_id]
    assert len(rows) == 2, (
        f"differentiate must emit two refined rows; got {len(rows)}: "
        f"{[(r.trigger, r.content) for r in rows]}"
    )

    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)
    assert len(surviving) == 2, (
        f"two refined rows must survive; got {len(surviving)}: "
        f"{[(r.trigger, r.content) for r in surviving]}"
    )

    surviving_triggers = {r.trigger for r in surviving}
    assert trigger not in surviving_triggers, (
        f"original colliding trigger {trigger!r} must not survive "
        f"differentiation; got {surviving_triggers}"
    )

    # Load-bearing invariant: no two current rows share a trigger with
    # opposing polarity (vacuously true when triggers are disjoint, but
    # the assertion documents the contract).
    _assert_no_opposing_polarity_on_same_trigger(surviving)
