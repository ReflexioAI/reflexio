"""Integration tests for the playbook consolidator apply paths.

These tests drive ``PlaybookConsolidator.deduplicate`` end-to-end with a real
``SQLiteStorage`` instance and a mocked LLM, verifying that each of the five
``ConsolidationDecision`` kinds produces the correct storage transitions:

* ``PreferNewDecision``  — existing row archived, new candidate inserted.
* ``PreferExistingDecision`` — storage state unchanged.
* ``DifferentiateDecision`` — existing archived, two refined rows emitted.
* ``IndependentDecision`` — new candidate inserted, no archive.
* ``DuplicateDecision`` — EXISTING members archived, one merged row inserted
  with ``merged_polarity`` threaded through.

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
    DuplicateDecision,
    IndependentDecision,
    PlaybookConsolidationOutput,
    PlaybookConsolidator,
    PreferExistingDecision,
    PreferNewDecision,
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


class TestPreferNew:
    """``PreferNewDecision`` — archive existing, insert new as-is."""

    def test_archives_existing_and_inserts_new(
        self, sqlite_storage, request_context, consolidator
    ):
        """Seeded positive existing + negative candidate ⇒ existing archived, new inserted."""
        existing = _make_existing_playbook(sqlite_storage, polarity="positive")
        candidate = _make_candidate(content="Avoid X.", polarity="negative")

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                PreferNewDecision(new_id="NEW-0", existing_id=existing.user_playbook_id)
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        assert rows[0].content == "Avoid X."
        assert rows[0].polarity == "negative"

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        # SQLite delete is a hard remove; only the candidate row remains.
        assert len(surviving) == 1
        assert surviving[0].polarity == "negative"
        assert surviving[0].content == "Avoid X."


class TestPreferExisting:
    """``PreferExistingDecision`` — storage state unchanged."""

    def test_storage_unchanged(self, sqlite_storage, request_context, consolidator):
        """Existing wins ⇒ candidate dropped, archive list empty, existing untouched."""
        existing = _make_existing_playbook(sqlite_storage, polarity="positive")
        candidate = _make_candidate(content="Recommend X.", polarity="positive")

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                PreferExistingDecision(
                    new_id="NEW-0", existing_id=existing.user_playbook_id
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


class TestDuplicate:
    """``DuplicateDecision`` — archive EXISTING members, insert one merged row."""

    def test_archives_existing_members_and_inserts_merged(
        self, sqlite_storage, request_context, consolidator
    ):
        """Merged row inherits ``merged_polarity`` and combines source ids."""
        existing = _make_existing_playbook(
            sqlite_storage,
            polarity="positive",
            content="Recommend X.",
            trigger="when Y",
        )
        candidate = _make_candidate(
            content="Recommend X (eagerly).",
            trigger="when Y",
            polarity="positive",
        )
        candidate.source_interaction_ids = [10, 20]
        # Existing row was seeded with empty source_interaction_ids; bump it so
        # we can verify that the merge combines NEW + EXISTING source ids.
        existing.source_interaction_ids = [99]

        rows, archive_ids = _run_consolidator(
            consolidator,
            candidates=[candidate],
            existing_playbooks=[existing],
            decisions=[
                DuplicateDecision(
                    item_ids=["NEW-0", "EXISTING-0"],
                    merged_content="Recommend X.",
                    merged_trigger="when Y",
                    merged_rationale="merged rationale",
                    merged_polarity="negative",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        merged = rows[0]
        assert merged.content == "Recommend X."
        assert merged.trigger == "when Y"
        assert merged.rationale == "merged rationale"
        # ``merged_polarity`` from the decision must win over the template's polarity.
        assert merged.polarity == "negative"
        # Source ids should combine NEW + EXISTING members.
        assert set(merged.source_interaction_ids) == {10, 20, 99}

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        assert len(surviving) == 1
        assert surviving[0].polarity == "negative"
        assert surviving[0].content == "Recommend X."


class TestContradictionResolutionContract:
    """Linchpin contract: opposing-polarity same-trigger pairs MUST route through
    a contradiction kind (``prefer_new`` / ``prefer_existing`` / ``differentiate``)
    and MUST NEVER be silently merged as a ``DuplicateDecision`` or accepted as
    ``IndependentDecision``.

    Section E4 of the reflection-extraction-polarity plan asserts this as a
    structural invariant of the apply layer — the LLM prompt encodes it as
    soft guidance, but the apply path treats a mixed-polarity
    ``DuplicateDecision`` as a runtime contract violation and refuses to
    materialise either rule, preventing the two opposite recommendations from
    co-existing in current storage.
    """

    def test_opposing_polarity_same_trigger_resolves_to_prefer_or_differentiate(
        self, sqlite_storage, request_context, consolidator, caplog
    ):
        """A bad ``DuplicateDecision`` over opposing polarities is rejected.

        If the LLM returns a ``DuplicateDecision`` that groups a positive
        existing row with a negative candidate sharing the same trigger, the
        apply layer raises ``ConsolidationContractViolation`` and the
        per-decision isolation in ``_build_deduplicated_results`` bumps the
        failed counter. Crucially, the safety fallback must NOT silently
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
                    DuplicateDecision(
                        item_ids=["NEW-0", "EXISTING-0"],
                        merged_content="Avoid X.",
                        merged_trigger="when Y",
                        merged_rationale="conflict — LLM mis-merged opposite polarities",
                        merged_polarity="negative",
                    )
                ],
            )

        # Apply layer rejected the bad decision: no row produced, no archive.
        assert rows == [], (
            "contract violation must NOT produce a merged row — got "
            f"{[(r.content, r.polarity) for r in rows]}"
        )
        assert archive_ids == [], (
            "contract violation must NOT archive the existing row — got "
            f"{archive_ids}"
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

    def test_opposing_polarity_same_trigger_with_prefer_new_archives_existing(
        self, sqlite_storage, request_context, consolidator
    ):
        """The legitimate path: ``PreferNewDecision`` flips polarity cleanly.

        Same trigger, opposite polarity — the LLM correctly routes the pair
        through ``prefer_new`` (the new negative evidence supersedes the old
        positive recommendation). The existing positive row is archived and
        the negative candidate becomes the sole current row, so the two
        opposite rules never co-exist.
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
                PreferNewDecision(
                    new_id="NEW-0",
                    existing_id=existing.user_playbook_id,
                    reason="negative evidence supersedes prior positive recommendation",
                )
            ],
        )

        assert archive_ids == [existing.user_playbook_id]
        assert len(rows) == 1
        assert rows[0].polarity == "negative"
        assert rows[0].content == "Avoid X."

        _apply_to_storage(sqlite_storage, rows, archive_ids)
        surviving = sqlite_storage.get_user_playbooks(user_id="u1")
        # SQLite delete is a hard remove in the integration setup; only the
        # negative successor remains. The legacy positive row is gone, so
        # there's no opposing-polarity co-existence.
        assert len(surviving) == 1
        assert surviving[0].polarity == "negative"
        assert surviving[0].content == "Avoid X."
        assert surviving[0].trigger == "when Y"
        # And explicitly: no surviving row with polarity="positive" and the
        # same trigger remains.
        assert not any(
            r.polarity == "positive" and r.trigger == "when Y" for r in surviving
        ), (
            "no positive-polarity row with trigger 'when Y' may survive — got "
            f"{[(r.content, r.polarity, r.trigger) for r in surviving]}"
        )
