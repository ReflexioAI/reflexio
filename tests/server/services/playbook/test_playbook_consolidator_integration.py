"""Integration tests for the playbook consolidator apply paths.

These tests drive ``PlaybookConsolidator.deduplicate`` end-to-end with a real
``SQLiteStorage`` instance and a mocked LLM, verifying that each of the four
``ConsolidationDecision`` kinds produces the correct storage transitions:

* ``UnifyDecision`` — 0..N EXISTING archived; one row inserted carrying the
  LLM-supplied final ``content`` / ``trigger`` / ``rationale``. Under Option B
  a unified skill may hold mixed-polarity rules and the apply path performs no
  mechanical polarity check (the no-self-contradiction judgment is the LLM's).
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
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.service_schemas import UserPlaybook
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
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
    # Orientation is a wording convention: a negative row uses avoidance
    # framing ("Avoid …") and carries a failure signal in its rationale to be
    # coherent, mirroring what the extractor actually writes.
    rationale = "user pushback observed" if polarity == "negative" else "r"
    pb = UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v0",
        request_id="r0",
        playbook_name=playbook_name,
        content=content,
        trigger=trigger,
        rationale=rationale,
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
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
    # Orientation is a wording convention: a negative candidate uses avoidance
    # framing ("Avoid …") and carries a failure signal in its rationale,
    # mirroring what the extractor actually writes.
    rationale = "user pushback observed" if polarity == "negative" else "r"
    return UserPlaybook(
        user_playbook_id=0,
        user_id=user_id,
        agent_version="v0",
        request_id=request_id,
        playbook_name="default",
        content=content,
        trigger=trigger,
        rationale=rationale,
        blocking_issue=None,
        status=None,
        source="chat",
        source_interaction_ids=[],
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

    ``deduplicate`` returns a 3-tuple ``(rows, archive_ids, merge_groups)``; the
    apply-path tests in this file assert only on ``(rows, archive_ids)``, so the
    merge-group element is dropped here. Merge-group routing through
    ``merge_records`` is covered by
    ``test_consolidation_lineage_integration.py``.

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
        rows, archive_ids, _merge_groups = consolidator.deduplicate(
            results=[candidates],
            request_id=request_id,
            agent_version="v0",
        )
    return rows, archive_ids


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
        unified row carrying the LLM-supplied content. Under Option B there is
        no apply-time polarity check; this same-orientation pair exercises the
        plain dedup/supersede path.
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
                    rationale="merged after user pushback observed",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        assert rows[0].content == "Avoid X (always)."
        assert rows[0].content.lstrip().startswith("Avoid")

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        # SQLite delete is a hard remove; only the unified row remains.
        assert len(surviving) == 1
        assert surviving[0].content.lstrip().startswith("Avoid")
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
            source="chat",
            source_interaction_ids=[],
        )
        sqlite_storage.save_user_playbooks([pb_b])
        all_existing = sqlite_storage.get_user_playbooks(user_id="u_nway")
        assert len(all_existing) == 2
        existing_b = next(
            p for p in all_existing if p.user_playbook_id != existing_a.user_playbook_id
        )

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

    def test_insert_without_archive(
        self, sqlite_storage, request_context, consolidator
    ):
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
    """Option B contract: a same-SITUATION contradiction (same trigger, opposite
    advice) MUST route through ``differentiate`` or ``reject_new`` — never
    ``unify`` (which would let a skill contradict itself) and never
    ``independent``. The no-self-contradiction judgment is now made by the LLM
    in the consolidation prompt; the apply layer no longer enforces a mechanical
    same-polarity guard.

    Conversely, a mixed-polarity ``unify`` across DIFFERENT sub-aspects (a
    do-rule + an avoid-rule for distinct situations) is now LEGITIMATE and
    composes a multi-rule skill — the case the old mechanical validator wrongly
    blocked.
    """

    def test_mixed_polarity_unify_composes_multi_rule_skill(
        self, sqlite_storage, request_context, consolidator
    ):
        """Mixed-polarity ``unify`` on DIFFERENT sub-aspects now SUCCEEDS.

        A NEW avoid-rule on a distinct sub-aspect ("avoid Friday deploys")
        composes with an EXISTING do-rule ("announce in the channel") into one
        multi-rule skill. Under Option B the apply layer no longer derives a
        whole-content polarity nor rejects the merge — the LLM is responsible
        for only composing coherent, non-self-contradicting rules. The merge
        must apply: the existing row is archived and the unified row carries
        both rules.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Do: announce the deploy in the team channel.",
            trigger="deploying a service",
        )
        candidate = _make_candidate(
            content="Avoid Friday-afternoon deploys.",
            trigger="deploying a service",
            polarity="negative",
        )

        unified_content = (
            "Do: announce the deploy in the team channel. "
            "Avoid: Friday-afternoon deploys."
        )
        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                UnifyDecision(
                    new_id="NEW-0",
                    archive_existing_ids=[0],
                    content=unified_content,
                    trigger="deploying a service",
                    rationale=(
                        "composed multi-rule deploy skill: announce (do) and "
                        "avoid Friday deploys (avoid) cover different sub-aspects"
                    ),
                )
            ],
        )

        # The merge applied: the existing row is archived and one unified row
        # carrying BOTH rules is produced. No ConsolidationContractError.
        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        assert rows[0].content == unified_content
        # Both the do-rule and the avoid-rule survived the merge.
        assert "announce" in rows[0].content.lower()
        assert "friday" in rows[0].content.lower()

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1, (
            "exactly one composed skill must survive — got "
            f"{[r.content for r in surviving]}"
        )
        assert surviving[0].content == unified_content

    def test_same_situation_contradiction_does_not_unify(
        self, sqlite_storage, request_context, consolidator
    ):
        """Same-situation contradiction routes through ``differentiate``, NOT ``unify``.

        Same trigger, opposite advice on the SAME sub-aspect ("use -F" vs
        "avoid -F"). Under Option B this is the forbidden self-contradiction
        case, and the decision is driven by the LLM: the mocked LLM returns a
        ``DifferentiateDecision`` (refine the triggers so each rule owns a
        disjoint situation) rather than a ``unify``. The apply layer no longer
        has a mechanical guard — it simply executes the LLM's decision. Assert
        the pair is NOT merged into one self-contradicting skill.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Use -F when pushing.",
            trigger="git push",
        )
        candidate = _make_candidate(
            content="Avoid -F when pushing.",
            trigger="git push",
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
                    refined_new_trigger="git push to a shared branch",
                    refined_existing_trigger="git push to your own feature branch",
                )
            ],
        )

        # NOT unified into one row: differentiate archives the existing and
        # emits two refined rows on disjoint triggers.
        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 2
        assert all(
            r.content != "Use -F when pushing. Avoid -F when pushing." for r in rows
        )

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 2
        surviving_triggers = {r.trigger for r in surviving}
        assert "git push" not in surviving_triggers
        assert surviving_triggers == {
            "git push to a shared branch",
            "git push to your own feature branch",
        }

    def test_same_situation_contradiction_resolves_via_reject_new(
        self, sqlite_storage, request_context, consolidator
    ):
        """Same-situation contradiction can also resolve via ``reject_new``.

        The other LLM-driven resolution: the existing rule wins and the new
        contradicting candidate is dropped. Storage is unchanged and the
        candidate does not leak in via the safety fallback.
        """
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Use -F when pushing.",
            trigger="git push",
        )
        candidate = _make_candidate(
            content="Avoid -F when pushing.",
            trigger="git push",
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
                    reason="storage-stability tie-break on same-situation contradiction",
                )
            ],
        )

        assert rows == []
        assert archive_ids == []

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1
        assert surviving[0].user_playbook_id == existing.user_playbook_id
        assert surviving[0].content == "Use -F when pushing."

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
        assert surviving[0].content.lstrip().startswith("Recommend")
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
        # Each refined trigger carries exactly the expected wording: the
        # declined-recently branch keeps the negative (avoidance) candidate;
        # the not-declined branch keeps the original positive rule.
        content_by_trigger = {r.trigger: r.content for r in surviving}
        assert (
            content_by_trigger["when Y AND has declined X recently"]
            .lstrip()
            .startswith("Avoid")
        )
        assert (
            content_by_trigger["when Y AND has not declined X recently"]
            .lstrip()
            .startswith("Recommend")
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

        # Orientation is a wording convention: avoidance ("Avoid …") vs. not.
        orientation_per_trigger: dict[str, set[bool]] = {}
        for pb in surviving:
            if pb.trigger is None:
                continue
            orientation_per_trigger.setdefault(pb.trigger, set()).add(
                pb.content.lstrip().startswith("Avoid")
            )

        violations = [
            (trigger, orientations)
            for trigger, orientations in orientation_per_trigger.items()
            if len(orientations) > 1
        ]
        assert violations, (
            "expected the forbidden 'independent over contradiction pair' to leave "
            "the post-state with mixed-orientation rows on the same trigger; got "
            f"{ {t: [pb.content for pb in surviving if pb.trigger == t] for t in orientation_per_trigger}!r}"
        )
        # The assertion above pins the contract: if the apply layer ever grows
        # a runtime guard, this test will fail and should be updated to assert
        # that the violation is rejected at apply time instead.


# ===============================
# End-to-end: native litellm retry + fallback flows through the consolidator
# ===============================


def _make_completion_response(content: str) -> MagicMock:
    """Build a minimal mock ``litellm.completion`` response.

    Mirrors the shape consumed by :class:`LiteLLMClient._make_request`: a
    single choice with ``message.content`` plus a token-usage object whose
    cache-detail fields are present but empty so the logging path does not
    raise. Matches the helper in ``tests/server/llm/test_litellm_client_unit.py``
    so the behaviour observed here is identical to that suite.

    Args:
        content (str): Raw text body the consolidator will parse as JSON into
            ``PlaybookConsolidationOutput``.

    Returns:
        MagicMock: Object compatible with ``response.choices[0].message.content``.
    """
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    resp.usage.prompt_tokens_details = None
    resp.usage.cache_creation_input_tokens = None
    resp.usage.cache_read_input_tokens = None
    # ``_emit_fallback_observability`` reads ``_hidden_params`` and falls back
    # to ``response.model``; leave both unset so it short-circuits without
    # tripping the Sentry path.
    resp._hidden_params = {}
    resp.model = None
    return resp


def _build_real_client_consolidator(
    request_context: RequestContext,
    *,
    config: LiteLLMConfig,
) -> PlaybookConsolidator:
    """Build a ``PlaybookConsolidator`` wired to a real ``LiteLLMClient``.

    The shared ``consolidator`` fixture in this file wires the consolidator to
    a ``MagicMock(spec=LiteLLMClient)`` — useful for testing the apply path
    but it never reaches ``litellm.completion``, so the native fallback
    delegation cannot be observed through it. These end-to-end tests need a
    real ``LiteLLMClient`` instance so the request actually flows into
    ``_make_request`` -> ``litellm.completion``.

    Args:
        request_context (RequestContext): Real ``RequestContext`` (storage +
            mocked prompt manager + real configurator).
        config (LiteLLMConfig): Fully-formed config — built fresh per-test so
            the ``fallback_models`` default factory reads the test's env state
            at construction time.

    Returns:
        PlaybookConsolidator: Consolidator whose ``client`` is a real
            ``LiteLLMClient``; ``model_name`` is fixed to ``"gpt-test"`` via
            the same ``SiteVarManager`` patch used by the file's shared
            fixture so the apply path is comparable.
    """
    client = LiteLLMClient(config)
    with patch(
        "reflexio.server.services.deduplication_utils.SiteVarManager"
    ) as mock_svm:
        mock_svm.return_value.get_site_var.return_value = {
            "default_generation_model_name": "gpt-test"
        }
        return PlaybookConsolidator(request_context=request_context, llm_client=client)


class TestConsolidatorNativeFallbackEndToEnd:
    """End-to-end: ``_consolidation_decisions`` forwards the fallback list into ``litellm.completion``.

    The consolidator was the original production incident site (structured
    output parse path on top of native fallback; same-model retry of a hung
    primary is disabled — PYTHON-FASTAPI-62), so this is the highest-fidelity
    exercise of the plumbing. Pinning both the "env var on => fallback
    configured" and "env var unset => no fallback" branches at this level
    prevents regressions where the plumbing works in isolation but breaks once
    a structured-output wrapper sits on top.
    """

    def test_consolidator_calls_litellm_with_fallback_configured(
        self, request_context, monkeypatch
    ):
        """Production-style: ``REFLEXIO_LLM_FALLBACK_MODELS`` set globally.

        Asserts the consolidator's call into ``litellm.completion`` carries
        ``num_retries=0`` (forced on the completion path so a hung primary
        can't be same-model-retried before the fallback — PYTHON-FASTAPI-62)
        and ``fallbacks=["gpt-5.4-mini"]`` end-to-end — proving native fallback
        delegation survives the structured-output parse wrapper.
        """
        monkeypatch.setenv("REFLEXIO_LLM_FALLBACK_MODELS", "gpt-5.4-mini")
        # LiteLLMConfig.fallback_models is a default_factory that reads
        # os.environ at construction — build the config AFTER setenv so the
        # new value flows in. The shared ``consolidator`` fixture builds its
        # mock client lazily but caches it for the test's lifetime; this
        # parallel builder sidesteps that cache entirely.
        consolidator = _build_real_client_consolidator(
            request_context,
            config=LiteLLMConfig(model="minimax/MiniMax-M3"),
        )

        captured: dict[str, Any] = {}

        def _fake(**params: Any) -> MagicMock:
            captured.update(params)
            return _make_completion_response('{"decisions": []}')

        monkeypatch.setattr("litellm.completion", _fake)

        result = consolidator._consolidation_decisions(
            new_playbooks=[], existing_playbooks=[]
        )

        # End-to-end shape: parsed cleanly into the structured-output model.
        assert isinstance(result, PlaybookConsolidationOutput)
        assert result.decisions == []

        # Linchpin: fallbacks forwarded; num_retries forced to 0 so a hung
        # primary can't be same-model-retried before reaching the fallback
        # (PYTHON-FASTAPI-62).
        assert captured.get("num_retries") == 0
        assert captured.get("fallbacks") == ["gpt-5.4-mini"]

    def test_consolidator_uses_no_fallback_when_env_unset(
        self, request_context, monkeypatch
    ):
        """Local / OSS safety contract: no env var => no fallback configured.

        With ``REFLEXIO_LLM_FALLBACK_MODELS`` unset and no explicit
        construction-arg, ``litellm.completion`` MUST NOT receive a
        ``fallbacks`` kwarg. This preserves the "never silently route to an
        unintended provider" guarantee for local reflexio and the
        claude-smart integration documented in ``LiteLLMConfig``.
        """
        monkeypatch.delenv("REFLEXIO_LLM_FALLBACK_MODELS", raising=False)
        consolidator = _build_real_client_consolidator(
            request_context,
            config=LiteLLMConfig(model="claude-code/claude-sonnet-4-6"),
        )

        captured: dict[str, Any] = {}

        def _fake(**params: Any) -> MagicMock:
            captured.update(params)
            return _make_completion_response('{"decisions": []}')

        monkeypatch.setattr("litellm.completion", _fake)

        result = consolidator._consolidation_decisions(
            new_playbooks=[], existing_playbooks=[]
        )

        assert isinstance(result, PlaybookConsolidationOutput)
        assert result.decisions == []
        # ``num_retries`` is forced to 0 on the completion path; ``fallbacks``
        # must be absent so litellm has no fallback chain to traverse.
        assert captured.get("num_retries") == 0
        assert "fallbacks" not in captured
