"""Integration tests for the playbook consolidator apply paths.

These tests drive ``PlaybookConsolidator.deduplicate`` end-to-end with a real
``SQLiteStorage`` instance and a mocked LLM, verifying that each of the four
``ConsolidationDecision`` kinds produces the correct storage transitions:

* ``UnifyDecision`` — 0..N EXISTING archived; one row inserted carrying the
  LLM-supplied final ``content`` / ``trigger`` / ``rationale`` / ``polarity``.
* ``RejectNewDecision`` — storage state unchanged (NEW dropped, EXISTING wins).
* ``DifferentiateDecision`` — existing archived, two refined rows emitted.
* ``IndependentDecision`` — new candidate inserted, no archive.

The mocked LLM returns ``PlaybookConsolidationOutput`` directly, so these
tests focus on the dispatch + apply behaviour. Archive semantics are modelled
by callers (the generation service runs ``delete_user_playbooks_by_ids`` on
the returned id list); the apply path itself returns ``(rows_to_save,
ids_to_delete)`` and these tests verify that contract.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient
from reflexio.server.services.playbook.playbook_consolidator import (
    DifferentiateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
    RejectNewDecision,
    UnifyDecision,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


# ===============================
# Fixtures
# ===============================


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for SQLite isolation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def sqlite_storage(temp_storage_dir, worker_id):
    """Real SQLite storage in a per-test temp dir + per-worker org id."""
    return SQLiteStorage(
        org_id=f"test-consolidator-apply-{worker_id}",
        db_path=os.path.join(temp_storage_dir, "consolidator_apply.db"),
    )


@pytest.fixture
def request_context(sqlite_storage, temp_storage_dir, worker_id):
    """RequestContext wired to the real SQLite storage with a mocked prompt manager."""
    context = RequestContext(
        org_id=f"test-consolidator-apply-{worker_id}",
        storage_base_dir=temp_storage_dir,
    )
    context.storage = sqlite_storage
    context.prompt_manager = MagicMock()
    context.prompt_manager.render_prompt.return_value = "mock prompt"
    return context


@pytest.fixture
def mock_llm_client():
    """Mock LiteLLM client. ``generate_chat_response`` is set per-test."""
    return MagicMock(spec=LiteLLMClient)


@pytest.fixture
def consolidator(request_context, mock_llm_client):
    """``PlaybookConsolidator`` wired with real storage + mock LLM client."""
    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as mock_svm:
        mock_svm.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-test"
        }
        return PlaybookConsolidator(
            request_context=request_context, llm_client=mock_llm_client
        )


# ===============================
# Helpers
# ===============================


def _make_existing_playbook(
    storage: SQLiteStorage,
    *,
    user_id: str = "u1",
    playbook_name: str = "default",
    content: str | None = None,
    trigger: str = "when Y",
    polarity: str = "positive",
) -> UserPlaybook:
    """Insert one existing UserPlaybook into storage and return the persisted row.

    Args:
        storage: Real SQLite storage handle.
        user_id: User id scoping the playbook.
        playbook_name: Playbook name field.
        content: Optional content. Defaults to "Avoid X." for negative and
            "Recommend X." for positive.
        trigger: Trigger string.
        polarity: ``"positive"`` or ``"negative"``.

    Returns:
        The persisted ``UserPlaybook`` with its assigned ``user_playbook_id``.
    """
    if content is None:
        content = "Avoid X." if polarity == "negative" else "Recommend X."
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v0",
        request_id="r0",
        playbook_name=playbook_name,
        content=content,
        trigger=trigger,
        rationale="r",
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
        polarity=polarity,  # type: ignore[arg-type]
    )
    storage.save_user_playbooks([pb])
    saved = storage.get_user_playbooks(user_id=user_id)
    assert len(saved) == 1, "Seed setup failed — expected one persisted playbook"
    return saved[0]


def _make_candidate(
    *,
    user_id: str = "u1",
    content: str = "Recommend X.",
    trigger: str = "when Y",
    polarity: str = "positive",
    request_id: str = "r1",
) -> UserPlaybook:
    """Build a NEW candidate UserPlaybook (not persisted).

    Args:
        user_id: User id scoping the candidate.
        content: Candidate content body.
        trigger: Trigger string.
        polarity: ``"positive"`` or ``"negative"``.
        request_id: Request id for the candidate.

    Returns:
        A fresh ``UserPlaybook`` ready to flow through ``deduplicate``.
    """
    return UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v0",
        request_id=request_id,
        playbook_name="default",
        content=content,
        trigger=trigger,
        rationale="r",
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
        polarity=polarity,  # type: ignore[arg-type]
    )


def _run_consolidator(
    consolidator: PlaybookConsolidator,
    *,
    candidates: list[UserPlaybook],
    existing_playbooks: list[UserPlaybook],
    decisions: list,
    request_id: str = "req_test",
) -> tuple[list[UserPlaybook], list[int]]:
    """Drive ``deduplicate`` with a scripted LLM response and pre-fetched existing rows.

    Patches ``_retrieve_existing_playbooks`` so the LLM-mock decisions can
    reference EXISTING-N ids by position without depending on the search
    backend's ranking.

    Args:
        consolidator: Configured consolidator instance.
        candidates: Flat list of candidate ``UserPlaybook`` rows.
        existing_playbooks: Pre-fetched existing rows (positional ids
            ``EXISTING-0``, ``EXISTING-1``, …).
        decisions: Decisions returned by the mocked LLM.
        request_id: Request id forwarded to ``deduplicate``.

    Returns:
        Tuple of (rows_to_save, ids_to_delete) returned by ``deduplicate``.
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
            agent_version="v0",
        )


def _apply_to_storage(
    storage: SQLiteStorage,
    rows_to_save: list[UserPlaybook],
    ids_to_delete: list[int],
) -> None:
    """Replicate the generation-service apply: delete then save.

    Args:
        storage: Real SQLite storage handle.
        rows_to_save: Rows produced by the consolidator.
        ids_to_delete: Existing ids the consolidator chose to archive.
    """
    if ids_to_delete:
        storage.delete_user_playbooks_by_ids(ids_to_delete)
    if rows_to_save:
        storage.save_user_playbooks(rows_to_save)


# ===============================
# Tests — one class per decision kind
# ===============================


class TestUnify:
    """``UnifyDecision`` — archive 0..N EXISTING; insert one LLM-supplied row."""

    def test_pair_replacement_archives_existing_and_inserts_unified(
        self, sqlite_storage, request_context, consolidator
    ):
        """Pair replacement (was ``prefer_new``): NEW negative supersedes EXISTING negative.

        ``unify`` with one archived EXISTING and one NEW produces a single
        unified row carrying the LLM-supplied content and polarity. Polarity
        validator requires the archived EXISTING's polarity to match the
        decision's polarity, so this scenario uses a same-polarity pair.
        """
        existing = _make_existing_playbook(sqlite_storage, polarity="negative")
        candidate = _make_candidate(content="Avoid X (always).", polarity="negative")

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                UnifyDecision(
                    new_id="NEW-0",
                    archive_existing_ids=[0],
                    content="Avoid X (always).",
                    trigger="when Y",
                    rationale="merged",
                    polarity="negative",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        assert rows[0].content == "Avoid X (always)."
        assert rows[0].polarity == "negative"

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        # SQLite delete is a hard remove; only the unified row remains.
        assert len(surviving) == 1
        assert surviving[0].polarity == "negative"
        assert surviving[0].content == "Avoid X (always)."

    def test_n_way_merge_archives_all_existing_members_and_inserts_one(
        self, sqlite_storage, request_context, consolidator
    ):
        """N-way merge (was ``duplicate``): one NEW + multiple EXISTING archived.

        Verifies that the apply path archives every referenced EXISTING id and
        combines source_interaction_ids across all members.
        """
        existing_a = _make_existing_playbook(
            sqlite_storage,
            user_id="u_nway",
            playbook_name="a",
            content="Recommend X (variant a).",
            polarity="positive",
        )
        # Second existing row on the same user_id; bypass _make_existing_playbook's
        # single-row assertion by saving directly.
        pb_b = UserPlaybook(
            user_playbook_id=0,
            user_id="u_nway",
            agent_version="v0",
            request_id="r0",
            playbook_name="b",
            content="Recommend X (variant b).",
            trigger="when Y",
            rationale="r",
            polarity="positive",
            source="chat",
            source_interaction_ids=[],
        )
        sqlite_storage.save_user_playbooks([pb_b])
        all_existing = sqlite_storage.get_user_playbooks(user_id="u_nway")
        assert len(all_existing) == 2
        existing_b = next(p for p in all_existing if p.user_playbook_id != existing_a.user_playbook_id)

        candidate = _make_candidate(
            user_id="u_nway",
            content="Recommend X (canonical).",
            polarity="positive",
        )
        candidate.source_interaction_ids = [10]
        existing_a.source_interaction_ids = [1]
        existing_b.source_interaction_ids = [2]

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing_a, existing_b],
            decisions=[
                UnifyDecision(
                    new_id="NEW-0",
                    archive_existing_ids=[0, 1],
                    content="Recommend X (canonical).",
                    trigger="when Y",
                    rationale="merged",
                    polarity="positive",
                )
            ],
        )

        assert set(archive_ids) == {
            existing_a.user_playbook_id,
            existing_b.user_playbook_id,
        }
        assert len(rows) == 1
        assert set(rows[0].source_interaction_ids) == {1, 2, 10}

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u_nway")
        assert len(surviving) == 1
        assert surviving[0].content == "Recommend X (canonical)."

    def test_insert_without_archive(self, sqlite_storage, request_context, consolidator):
        """``unify`` with empty ``archive_existing_ids`` inserts NEW without archiving.

        This shape is conceptually ``independent`` at the storage layer; the
        prompt should steer the LLM toward ``independent`` in this case, but
        the apply path supports the degenerate ``unify`` shape.
        """
        candidate = _make_candidate(content="Recommend Z.", polarity="positive")

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[],
            decisions=[
                UnifyDecision(
                    new_id="NEW-0",
                    archive_existing_ids=[],
                    content="Recommend Z.",
                    trigger="when Y",
                    rationale="r",
                    polarity="positive",
                )
            ],
        )

        assert archive_ids == []
        assert len(rows) == 1
        assert rows[0].content == "Recommend Z."

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1
        assert surviving[0].content == "Recommend Z."


class TestRejectNew:
    """``RejectNewDecision`` — NEW dropped; EXISTING wins; no storage change."""

    def test_storage_unchanged(self, sqlite_storage, request_context, consolidator):
        """Existing supersedes candidate ⇒ no rows produced, archive list empty."""
        existing = _make_existing_playbook(sqlite_storage, polarity="positive")
        candidate = _make_candidate(content="Recommend X.", polarity="positive")

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                RejectNewDecision(
                    new_id="NEW-0",
                    superseded_by_existing_id=existing.user_playbook_id,
                )
            ],
        )

        assert rows == []
        assert archive_ids == []

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1
        assert surviving[0].user_playbook_id == existing.user_playbook_id
        assert surviving[0].content == "Recommend X."


class TestDifferentiate:
    """``DifferentiateDecision`` — archive existing, insert two refined rows."""

    def test_archives_both_and_inserts_two_refined(
        self, sqlite_storage, request_context, consolidator
    ):
        """Refined triggers produce two new rows; original existing row archived."""
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Recommend X (premium).",
            trigger="when Y",
            polarity="positive",
        )

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                DifferentiateDecision(
                    new_id="NEW-0",
                    existing_id=existing.user_playbook_id,
                    refined_new_trigger="when Y AND user is premium",
                    refined_existing_trigger="when Y AND user is free tier",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 2

        triggers = {r.trigger for r in rows}
        contents = {r.content for r in rows}
        assert "when Y AND user is premium" in triggers
        assert "when Y AND user is free tier" in triggers
        assert "Recommend X (premium)." in contents
        assert "Recommend X." in contents
        # Refined rows must NOT reuse the original primary key.
        assert all(r.user_playbook_id == 0 for r in rows)

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 2
        surviving_triggers = {r.trigger for r in surviving}
        assert surviving_triggers == {
            "when Y AND user is premium",
            "when Y AND user is free tier",
        }


class TestIndependent:
    """``IndependentDecision`` — insert new only; no archive."""

    def test_inserts_new_only(self, sqlite_storage, request_context, consolidator):
        """Unrelated candidate ⇒ stored as a fresh row; no existing archived."""
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Recommend Z.", trigger="when W", polarity="positive"
        )

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[IndependentDecision(new_id="NEW-0")],
        )

        assert archive_ids == []
        assert len(rows) == 1
        assert rows[0].content == "Recommend Z."

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 2
        contents = {r.content for r in surviving}
        assert contents == {"Recommend X.", "Recommend Z."}


class TestContradictionResolutionContract:
    """Linchpin contract: opposing-polarity same-trigger pairs MUST route through
    a contradiction kind (``unify`` with matching polarity, ``reject_new``, or
    ``differentiate``) and MUST NEVER be silently merged via a mixed-polarity
    ``unify`` or accepted as ``independent``.

    Under the 4-kind redesign the apply layer enforces this via the ``unify``
    polarity validator: a ``UnifyDecision`` that archives an EXISTING row with
    a different polarity raises ``ConsolidationContractError`` and the
    per-decision isolation in ``_build_deduplicated_results`` bumps the
    ``failed_count`` and suppresses the safety fallback for the NEW members,
    so the orphan candidate is not silently re-inserted as an opposing twin.
    """

    def test_opposing_polarity_unify_is_rejected_by_validator(
        self, sqlite_storage, request_context, consolidator, caplog
    ):
        """A ``unify`` archiving an opposite-polarity EXISTING is rejected.

        If the LLM returns a ``UnifyDecision`` that archives a positive
        EXISTING row but declares ``polarity="negative"`` (matching the NEW
        candidate), the apply layer raises ``ConsolidationContractError`` and
        the per-decision isolation in ``_build_deduplicated_results`` bumps
        the failed counter. Crucially, the safety fallback must NOT silently
        re-insert the orphan candidate — that would still leave both opposing
        rules in current storage, breaking the contract.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Avoid X.",
            trigger="when Y",
            polarity="negative",
        )

        with caplog.at_level("WARNING"):
            rows, archive_ids = _run_consolidator(
                consolidator,
                candidates=[candidate],
                existing_playbooks=[existing],
                decisions=[
                    UnifyDecision(
                        new_id="NEW-0",
                        archive_existing_ids=[0],
                        content="Avoid X.",
                        trigger="when Y",
                        rationale="conflict — LLM mis-merged opposite polarities",
                        polarity="negative",
                    )
                ],
            )

        # Apply layer rejected the bad decision: no row produced, no archive.
        assert rows == [], (
            "contract violation must NOT produce a unified row — got "
            f"{[(r.content, r.polarity) for r in rows]}"
        )
        assert archive_ids == [], (
            f"contract violation must NOT archive the existing row — got {archive_ids}"
        )

        # The per-decision isolation logged the contract violation.
        assert any(
            "consolidation_contract_violation" in record.message
            for record in caplog.records
        ), (
            "expected a consolidation_contract_violation warning; got: "
            f"{[r.message for r in caplog.records]}"
        )

        # Storage state: the existing positive row remains untouched, and the
        # negative candidate was NOT silently inserted by the safety fallback.
        # Opposing-polarity rules with the same trigger must NEVER both occupy
        # current state simultaneously.
        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1, (
            "exactly one row must survive — got "
            f"{[(r.content, r.polarity) for r in surviving]}"
        )
        assert surviving[0].user_playbook_id == existing.user_playbook_id
        assert surviving[0].polarity == "positive"
        assert surviving[0].content == "Recommend X."

    def test_opposing_polarity_resolves_via_reject_new(
        self, sqlite_storage, request_context, consolidator
    ):
        """Legitimate path: ``RejectNewDecision`` keeps EXISTING, drops NEW.

        Same trigger, opposite polarity — the LLM correctly routes the pair
        through ``reject_new`` (the EXISTING positive rule still applies; the
        new negative observation is treated as noise). Storage is unchanged
        and the candidate does not leak in via the safety fallback.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Avoid X.",
            trigger="when Y",
            polarity="negative",
        )

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                RejectNewDecision(
                    new_id="NEW-0",
                    superseded_by_existing_id=existing.user_playbook_id,
                    reason="storage-stability tie-break on opposite-polarity pair",
                )
            ],
        )

        assert rows == []
        assert archive_ids == []

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1
        assert surviving[0].user_playbook_id == existing.user_playbook_id
        assert surviving[0].polarity == "positive"
        assert surviving[0].content == "Recommend X."

    def test_opposing_polarity_resolves_via_differentiate(
        self, sqlite_storage, request_context, consolidator
    ):
        """Legitimate path: ``DifferentiateDecision`` refines both triggers cleanly.

        Same trigger, opposite polarity — the LLM correctly refines the two
        rules onto disjoint triggers so they no longer collide. The existing
        positive row is archived; two new rows emerge with disjoint refined
        triggers and opposite polarities — and the linchpin invariant
        ("no opposing polarities on the same trigger") still holds because the
        triggers are disjoint.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Avoid X.",
            trigger="when Y",
            polarity="negative",
        )

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                DifferentiateDecision(
                    new_id="NEW-0",
                    existing_id=existing.user_playbook_id,
                    refined_new_trigger="when Y AND has declined X recently",
                    refined_existing_trigger="when Y AND has not declined X recently",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 2

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 2
        surviving_triggers = {r.trigger for r in surviving}
        assert "when Y" not in surviving_triggers
        # Each refined trigger appears with exactly one polarity.
        polarity_by_trigger = {r.trigger: r.polarity for r in surviving}
        assert (
            polarity_by_trigger["when Y AND has declined X recently"] == "negative"
        )
        assert (
            polarity_by_trigger["when Y AND has not declined X recently"]
            == "positive"
        )

    def test_independent_over_contradiction_pair_is_forbidden_post_hoc(
        self, sqlite_storage, request_context, consolidator
    ):
        """Linchpin contract: ``independent`` MUST NOT be chosen for the contradiction pair.

        The 4-kind redesign does not add a runtime guard against ``independent``
        over a same-trigger opposite-polarity pair (that responsibility lives
        in the prompt's hard-rules section). This test pins the contract via a
        post-hoc assertion: if the LLM mis-emits ``independent``, the storage
        state would end up with two opposite-polarity rows on the same trigger
        — which the assertion catches and flags as a contract violation.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Avoid X.",
            trigger="when Y",
            polarity="negative",
        )

        # Forbidden LLM response: ``independent`` over the contradiction pair.
        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[IndependentDecision(new_id="NEW-0")],
        )

        # The apply layer DOES execute the independent decision (the
        # consolidator does not look across decisions to detect this).
        # The contract is structural: storage post-state would carry two
        # opposite-polarity rows on the same trigger, which the assertion
        # below flags as a violation. The prompt is responsible for never
        # emitting this shape.
        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")

        polarities_per_trigger: dict[str, set[str]] = {}
        for pb in surviving:
            if pb.trigger is None:
                continue
            polarities_per_trigger.setdefault(pb.trigger, set()).add(pb.polarity)

        violations = [
            (trigger, polarities)
            for trigger, polarities in polarities_per_trigger.items()
            if "positive" in polarities and "negative" in polarities
        ]
        assert violations, (
            "expected the forbidden 'independent over contradiction pair' to leave "
            "the post-state with opposing-polarity rows on the same trigger; got "
            f"{polarities_per_trigger!r}"
        )
        # The assertion above pins the contract: if the apply layer ever grows
        # a runtime guard, this test will fail and should be updated to assert
        # that the violation is rejected at apply time instead.
