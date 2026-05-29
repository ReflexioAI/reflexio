"""End-to-end test for write-time contradiction resolution.

Verifies the load-bearing invariant of the reflection-extraction-polarity
feature at the consolidator boundary:

When an EXISTING positive ``UserPlaybook`` collides with a NEW
failure-path-derived NEGATIVE candidate on the same trigger, the
consolidator MUST route the pair through a contradiction-aware decision
(``UnifyDecision`` carrying the polarity the NEW evidence justifies,
``RejectNewDecision``, or ``DifferentiateDecision``). After the
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
  with a scripted LLM response covering each of the three allowed
  resolutions of the contradiction pair under the 4-kind redesign:
  ``RejectNewDecision`` (the existing positive wins),
  ``DifferentiateDecision`` (the two rules refine onto disjoint
  triggers), and a forbidden ``UnifyDecision`` with mismatched polarity
  (rejected by the apply-layer polarity validator). The forbidden case
  is the structural linchpin guard the 4-kind redesign buys.
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
    UnifyDecision,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

pytestmark = pytest.mark.e2e


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
        polarity="positive",
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
        polarity="negative",
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
        return consolidator.deduplicate(
            results=[candidates],
            request_id=request_id,
            agent_version="v1",
        )


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
    triggers_seen: dict[str, set[str]] = {}
    for pb in surviving:
        if pb.trigger is None:
            continue
        polarities = triggers_seen.setdefault(pb.trigger, set())
        polarities.add(pb.polarity)
    for trigger, polarities in triggers_seen.items():
        assert polarities <= {"positive"} or polarities <= {"negative"}, (
            f"contradiction-resolution invariant violated: trigger {trigger!r} "
            f"has surviving rows with polarities {polarities}"
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
    assert existing.polarity == "positive", (
        f"Setup failure — seeded playbook must be positive; got {existing.polarity!r}"
    )

    # Phase 1 (cont.) — construct the failure-path negative candidate on
    # the same trigger.
    candidate = _build_failure_path_negative_candidate(user_id=user_id, trigger=trigger)
    assert candidate.polarity == "negative", (
        "Setup failure — failure-path candidate must be negative; "
        f"got {candidate.polarity!r}"
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
        f"reject_new must NOT emit any new row; got "
        f"{[(r.content, r.polarity) for r in rows]}"
    )

    # Phase 3 — apply to storage and inspect the post-state directly.
    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)

    # Storage post-state: the seeded positive row is still the only row.
    assert len(surviving) == 1, (
        "exactly one row must survive reject_new resolution; got "
        f"{[(r.user_playbook_id, r.content, r.polarity) for r in surviving]}"
    )
    survivor = surviving[0]
    assert survivor.user_playbook_id == existing.user_playbook_id, (
        "surviving row must be the original existing playbook"
    )
    assert survivor.polarity == "positive"
    assert survivor.trigger == trigger
    assert survivor.content == existing.content

    # Load-bearing invariant: no two current rows share a trigger with
    # opposing polarity. Asserting it here independent of row counts
    # documents the contract; the assertion above would catch any
    # regression that left both rows current, but this makes the intent
    # explicit.
    _assert_no_opposing_polarity_on_same_trigger(surviving)


def test_existing_positive_plus_failure_path_rejects_mismatched_polarity_unify(
    request_context: RequestContext,
    consolidator: PlaybookConsolidator,
):
    """A ``unify`` archiving an opposite-polarity EXISTING is rejected by the validator.

    If the LLM mis-emits a ``UnifyDecision`` that archives an EXISTING
    positive row while declaring ``polarity="negative"``, the apply-layer
    polarity validator raises ``ConsolidationContractError`` and the
    per-decision isolation in ``_build_deduplicated_results`` bumps the
    failed counter while suppressing the safety fallback for the orphan
    candidate. Storage is unchanged: the existing positive row remains,
    and the negative candidate is NOT silently inserted as an opposing
    twin. This is the structural linchpin of the 4-kind redesign.
    """
    storage = request_context.storage
    assert storage is not None, "RequestContext must provide SQLite storage"
    user_id = "u_contradiction_unify_reject"
    trigger = "when user asks about product X"

    existing = _seed_positive_playbook(storage, user_id=user_id, trigger=trigger)
    candidate = _build_failure_path_negative_candidate(user_id=user_id, trigger=trigger)

    rows, archive_ids = _drive_consolidator(
        consolidator,
        candidates=[candidate],
        existing_playbooks=[existing],
        decisions=[
            UnifyDecision(
                new_id="NEW-0",
                archive_existing_ids=[0],
                content=candidate.content,
                trigger=trigger,
                rationale="LLM mis-merged opposite-polarity pair",
                polarity="negative",
            )
        ],
    )

    # The polarity validator refused: no row emitted, no archive.
    assert rows == [], (
        f"polarity mismatch must NOT produce a unified row; got "
        f"{[(r.content, r.polarity) for r in rows]}"
    )
    assert archive_ids == [], (
        f"polarity mismatch must NOT archive the existing row; got {archive_ids!r}"
    )

    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)

    # Storage post-state: the seeded positive row remains, and the negative
    # candidate did NOT leak in via the safety fallback (the contract-error
    # path marks the orphan handled).
    assert len(surviving) == 1, (
        "exactly one row must survive polarity-mismatch rejection; got "
        f"{[(r.user_playbook_id, r.content, r.polarity) for r in surviving]}"
    )
    assert surviving[0].user_playbook_id == existing.user_playbook_id
    assert surviving[0].polarity == "positive"

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
        f"{[(r.trigger, r.polarity) for r in rows]}"
    )

    _apply_to_storage(storage, rows, archive_ids)
    surviving = storage.get_user_playbooks(user_id=user_id)
    assert len(surviving) == 2, (
        f"two refined rows must survive; got {len(surviving)}: "
        f"{[(r.trigger, r.polarity) for r in surviving]}"
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
