"""Unit tests for the sliding-window ReflectionService.

Stubs the LLM call but uses a real SQLite storage to exercise the
bookmark / window / archive paths end-to-end.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import (
    Citation,
    Interaction,
    Request,
    UserPlaybook,
    UserProfile,
)
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.operation_state_utils import OperationStateManager
from reflexio.server.services.reflection.reflection_service import ReflectionService
from reflexio.server.services.reflection.reflection_service_utils import (
    REFLECTION_OPERATION_NAME,
    ReflectionDecision,
    ReflectionOutput,
    ReflectionServiceRequest,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_storage_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def request_context(temp_storage_dir):
    # Patch _get_embedding so storage doesn't try to call out for embeddings.
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    with (
        patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
        patch.object(
            LiteLLMClient,
            "get_embeddings",
            side_effect=lambda texts, *_args, **_kwargs: [[0.0] * 512 for _ in texts],
        ),
    ):
        ctx = RequestContext(org_id="test_org", storage_base_dir=temp_storage_dir)
        yield ctx


@pytest.fixture
def llm_client():
    client = MagicMock()
    # Default: no decisions. Tests override per-case.
    client.generate_chat_response.return_value = ReflectionOutput(decisions=[])
    return client


@pytest.fixture
def service(request_context, llm_client):
    return ReflectionService(request_context=request_context, llm_client=llm_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_profile(
    storage, user_id: str, profile_id: str, content: str = "old content"
) -> UserProfile:
    p = UserProfile(
        profile_id=profile_id,
        user_id=user_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="seed_req",
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        custom_features={"k": "v"},
        source="seed",
    )
    storage.add_user_profile(user_id, [p])
    return p


def _seed_playbook(
    storage,
    user_playbook_id: int,
    user_id: str,
    playbook_name: str = "fb",
    content: str = "old rule",
    source_interaction_ids: list[int] | None = None,
) -> UserPlaybook:
    pb = UserPlaybook(
        user_playbook_id=user_playbook_id,
        user_id=user_id,
        agent_version="v1",
        request_id=f"seed_{user_playbook_id}",
        playbook_name=playbook_name,
        content=content,
        trigger="when X",
        rationale="because Y",
        source="seed",
        source_interaction_ids=source_interaction_ids or [],
    )
    storage.save_user_playbooks([pb])
    return pb


def _seed_request_with_interactions(
    storage,
    user_id: str,
    request_id: str,
    interactions: list[Interaction],
) -> None:
    storage.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            source="cli",
            agent_version="v1",
        )
    )
    storage.add_user_interactions_bulk(user_id=user_id, interactions=interactions)


def _make_interaction(
    user_id: str,
    request_id: str,
    role: str,
    content: str,
    citations: list[Citation] | None = None,
) -> Interaction:
    return Interaction(
        user_id=user_id,
        request_id=request_id,
        role=role,
        content=content,
        created_at=int(datetime.now(UTC).timestamp()),
        citations=citations or [],
    )


def _set_config(request_context, **overrides):
    """Patch the configurator to return a Config with overrides."""
    from reflexio.models.config_schema import Config, ReflectionConfig

    cfg = Config.model_validate(
        {
            "storage_config": {"db_path": None},
            "window_size": 5,
            "stride_size": 2,
            "reflection_config": ReflectionConfig().model_dump(),
            **overrides,
        }
    )
    request_context.configurator = MagicMock()
    request_context.configurator.get_config.return_value = cfg


def _bookmark_state(storage, org_id: str, user_id: str) -> dict | None:
    """Read the reflection bookmark directly from storage."""
    mgr = OperationStateManager(storage, org_id, REFLECTION_OPERATION_NAME)
    state_key = mgr._bookmark_key(REFLECTION_OPERATION_NAME, scope_id=user_id)
    return storage.get_operation_state(state_key)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGate:
    def test_disabled_short_circuits(self, request_context, service, llm_client):
        from reflexio.models.config_schema import Config, ReflectionConfig

        cfg = Config.model_validate(
            {
                "storage_config": {"db_path": None},
                "reflection_config": ReflectionConfig(enabled=False).model_dump(),
            }
        )
        request_context.configurator = MagicMock()
        request_context.configurator.get_config.return_value = cfg

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is False
        assert result.gate_open is False
        llm_client.generate_chat_response.assert_not_called()

    def test_gate_closed_when_below_stride_size(
        self, request_context, service, llm_client
    ):
        _set_config(request_context, window_size=10, stride_size=5)
        # Seed only 2 interactions; stride_size=5.
        _seed_request_with_interactions(
            request_context.storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello"),
            ],
        )
        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.gate_open is False
        assert result.ran is False
        llm_client.generate_chat_response.assert_not_called()
        assert _bookmark_state(request_context.storage, "test_org", "u1") is None


class TestNoCitations:
    def test_window_without_citations_advances_bookmark(
        self, request_context, service, llm_client
    ):
        _set_config(request_context, window_size=5, stride_size=2)
        _seed_request_with_interactions(
            request_context.storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello"),
            ],
        )
        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.gate_open is True
        assert result.cited_count == 0
        assert result.ran is False
        llm_client.generate_chat_response.assert_not_called()
        assert _bookmark_state(request_context.storage, "test_org", "u1") is not None


class TestCitedRowsAlreadyArchived:
    def test_skips_when_all_cited_rows_already_archived(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage

        # Seed a profile and immediately archive it (simulates the
        # deduplicator having already archived a cited row earlier in
        # the publish flow).
        _seed_profile(storage, "u1", "p1")
        storage.archive_profile_by_id("u1", "p1")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.gate_open is True
        assert result.cited_count == 1
        assert result.considered_count == 0
        assert result.ran is False
        llm_client.generate_chat_response.assert_not_called()
        # Window was examined → bookmark advances.
        assert _bookmark_state(request_context.storage, "test_org", "u1") is not None


class TestReplaceProfile:
    def test_replace_archives_cited_and_inserts_new_current(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1", content="old content")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_content="new content",
                    new_profile_time_to_live=ProfileTimeToLive.ONE_QUARTER,
                    reason="user contradicted earlier preference",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is True
        assert result.revised_count == 1
        assert result.no_change_count == 0

        current = storage.get_user_profile("u1", status_filter=[None])
        archived = storage.get_user_profile("u1", status_filter=[Status.ARCHIVED])
        assert len(current) == 1
        assert len(archived) == 1
        assert current[0].profile_id != "p1"
        assert current[0].content == "new content"
        assert current[0].profile_time_to_live == ProfileTimeToLive.ONE_QUARTER
        # Carried-over identity / metadata.
        assert current[0].user_id == "u1"
        assert current[0].custom_features == {"k": "v"}
        assert current[0].source == "seed"
        assert archived[0].profile_id == "p1"


class TestReplacePlaybook:
    def test_replace_archives_cited_and_inserts_new_current(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="new rule",
                    # A content rewrite must carry new_rationale (the prompt
                    # sets it on every playbook content revision); new_trigger
                    # omitted → falls back to archived.
                    new_rationale="rewritten because Y was wrong",
                    reason="rule was wrong",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is True
        assert result.revised_count == 1

        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        archived = storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert len(current) == 1
        assert len(archived) == 1
        assert current[0].user_playbook_id != 1
        assert current[0].content == "new rule"
        # Trigger falls back to archived row's value; rationale is the supplied one.
        assert current[0].trigger == "when X"
        assert current[0].rationale == "rewritten because Y was wrong"
        assert current[0].user_id == "u1"
        assert current[0].agent_version == "v1"
        assert current[0].playbook_name == "fb"

    def test_replace_preserves_source_interaction_ids(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", source_interaction_ids=[10, 11])

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="new rule",
                    new_rationale="rewritten because Y was wrong",
                    reason="rule was wrong",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert len(current) == 1
        assert current[0].source_interaction_ids == [10, 11]


class TestNoChange:
    def test_no_change_does_not_mutate_storage(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    reason="still correct",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is True
        assert result.no_change_count == 1
        assert result.revised_count == 0
        # Original profile is still current.
        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 1
        assert current[0].profile_id == "p1"


class TestLLMFailureBookmark:
    def test_llm_raises_does_not_advance_bookmark(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.side_effect = RuntimeError("network")

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is False
        assert result.gate_open is True
        # Profile untouched.
        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 1
        # Bookmark NOT advanced.
        assert _bookmark_state(request_context.storage, "test_org", "u1") is None


class TestPerDecisionMalformed:
    def test_unparsable_playbook_id_does_not_block_other_decisions(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1")
        _seed_playbook(storage, 1, "u1")

        cites = [
            Citation(kind="profile", real_id="p1"),
            Citation(kind="playbook", real_id="1"),
        ]
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "Assistant", "hello", citations=cites),
                _make_interaction("u1", "r1", "User", "ok"),
            ],
        )

        # LLM returns a malformed playbook target_id and a valid profile replace.
        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="not-an-int",
                    new_content="garbled",
                    reason="malformed",
                ),
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_content="new content",
                    reason="needed update",
                ),
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))
        assert result.ran is True
        assert result.revised_count == 1
        assert result.failed_count >= 1
        # Profile updated, playbook untouched.
        archived_profiles = storage.get_user_profile(
            "u1", status_filter=[Status.ARCHIVED]
        )
        assert len(archived_profiles) == 1
        archived_playbooks = storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert archived_playbooks == []


class TestArchiveAfterInsertFailure:
    """Post-insert archive failures must keep the new row and log at ERROR.

    The reflection service explicitly accepts a transient duplicate
    (cited row stays current, new row inserted) over silently dropping
    user data when the archive step fails. These tests pin that
    behavior down.
    """

    def test_replace_profile_archive_raises_keeps_new_row(
        self, request_context, service, llm_client, caplog
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1", content="old content")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_content="new content",
                    reason="needed update",
                )
            ]
        )

        with (
            patch.object(
                storage,
                "archive_profile_by_id",
                side_effect=RuntimeError("disk full"),
            ),
            caplog.at_level(
                "ERROR", logger="reflexio.server.services.reflection.reflection_service"
            ),
        ):
            result = service.run(ReflectionServiceRequest(user_id="u1"))

        # Replacement still counted as successful — new row is durable.
        assert result.ran is True
        assert result.revised_count == 1

        # New current row exists; cited row is also still current (transient duplicate).
        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 2
        assert {p.content for p in current} == {"old content", "new content"}

        # ERROR log includes both ids so an operator can reconcile.
        assert "reflection_archive_after_insert_failed" in caplog.text
        assert "kind=profile" in caplog.text
        assert "cited_id=p1" in caplog.text

    def test_replace_profile_archive_returns_false_logs_noop(
        self, request_context, service, llm_client, caplog
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_content="new content",
                    reason="needed update",
                )
            ]
        )

        # archive_profile_by_id returns False (e.g. row was already
        # archived between resolve and apply by a concurrent writer).
        with (
            patch.object(storage, "archive_profile_by_id", return_value=False),
            caplog.at_level(
                "ERROR", logger="reflexio.server.services.reflection.reflection_service"
            ),
        ):
            result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert "reflection_archive_after_insert_noop" in caplog.text

    def test_replace_playbook_archive_raises_keeps_new_row(
        self, request_context, service, llm_client, caplog
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="new rule",
                    new_rationale="rewritten after the rule misfired",
                    reason="needed update",
                )
            ]
        )

        with (
            patch.object(
                storage,
                "archive_user_playbook_by_id",
                side_effect=RuntimeError("disk full"),
            ),
            caplog.at_level(
                "ERROR", logger="reflexio.server.services.reflection.reflection_service"
            ),
        ):
            result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.ran is True
        assert result.revised_count == 1

        # Both rows still current — transient duplicate.
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert len(current) == 2
        assert {p.content for p in current} == {"old rule", "new rule"}

        assert "reflection_archive_after_insert_failed" in caplog.text
        assert "kind=playbook" in caplog.text
        assert "cited_id=1" in caplog.text


class TestLLMReportedFlip:
    def test_llm_reported_flip_applies_and_counts_as_revision(
        self, request_context, service, llm_client
    ):
        """LLM-reported flip: cited rule rewritten in the opposite orientation.

        A flip is no longer derived from wording — it is LLM-reported via a
        ``new_content`` rewrite carrying a ``new_rationale`` that names the
        motivating failure (per the memory_reflection prompt). It applies and
        counts as an ordinary revision; there is no separate flip counter.

        Verifies:
        - The cited playbook is archived and the rewritten row is current.
        - result.revised_count == 1 and result.content_revised_count == 1.
        - result.no_change_count == 0.
        - ReflectionResult has no flipped_count attribute (counter retired).
        """
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="Do X when Y.")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="Avoid X when Y.",
                    new_rationale="user pushed back when X was recommended",
                    reason="evidence of failure",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.ran is True
        assert result.revised_count == 1
        assert result.content_revised_count == 1
        assert result.no_change_count == 0
        # flipped_count was retired — no flip-specific counter remains.
        assert not hasattr(result, "flipped_count")

        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        archived = storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert len(current) == 1
        assert len(archived) == 1
        assert current[0].content == "Avoid X when Y."
        assert current[0].rationale == "user pushed back when X was recommended"
        assert archived[0].user_playbook_id == 1

    def test_content_revision_without_rationale_counts_as_failed(
        self, request_context, service, llm_client
    ):
        """A playbook content rewrite missing new_rationale is rejected.

        Flip-requires-rationale, expressed without polarity derivation: the
        prompt sets ``new_rationale`` on every playbook content rewrite
        (flips and substance rewrites alike), so ``_validate_decision``
        rejects any ``new_content`` revision that omits it. The rejected
        decision is counted as failed and the cited row stays current.
        """
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="Do X when Y.")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="Avoid X when Y.",
                    # new_rationale intentionally omitted — should fail validation
                    reason="evidence of failure",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.ran is True
        assert result.failed_count == 1
        assert result.revised_count == 0
        # Cited playbook must remain current — no archive happened.
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert len(current) == 1
        assert current[0].user_playbook_id == 1

    def test_trigger_only_revision_does_not_require_rationale(
        self, request_context, service, llm_client
    ):
        """A trigger-only revision (no new_content) is exempt from the rule.

        The new_rationale invariant gates only ``new_content`` rewrites;
        narrowing/widening a trigger without touching content applies
        normally and counts as a revision.
        """
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="Do X when Y.")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_trigger="when Y and not Z",
                    reason="sharpened trigger",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.ran is True
        assert result.revised_count == 1
        assert result.trigger_revised_count == 1
        assert result.failed_count == 0
        current = storage.get_user_playbooks(user_id="u1", status_filter=[None])
        assert len(current) == 1
        assert current[0].content == "Do X when Y."
        assert current[0].trigger == "when Y and not Z"


class TestPerPassCap:
    def test_cap_stops_applying_after_n_and_increments_capped_count(
        self, request_context, service, llm_client
    ):
        """With max_revisions_per_pass=2, only 2 of 3 revisions apply."""
        from reflexio.models.config_schema import Config, ReflectionConfig

        cfg = Config.model_validate(
            {
                "storage_config": {"db_path": None},
                "window_size": 5,
                "stride_size": 2,
                "reflection_config": ReflectionConfig(
                    max_revisions_per_pass=2
                ).model_dump(),
            }
        )
        request_context.configurator = MagicMock()
        request_context.configurator.get_config.return_value = cfg

        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="rule one")
        _seed_playbook(storage, 2, "u1", content="rule two")
        _seed_playbook(storage, 3, "u1", content="rule three")

        cites = [
            Citation(kind="playbook", real_id="1"),
            Citation(kind="playbook", real_id="2"),
            Citation(kind="playbook", real_id="3"),
        ]
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "Assistant", "hello", citations=cites),
                _make_interaction("u1", "r1", "User", "ok"),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id=str(i),
                    new_content=f"revised {i}",
                    new_rationale=f"rewritten rule {i}",
                    reason="needed update",
                )
                for i in (1, 2, 3)
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.ran is True
        assert result.revised_count == 2
        assert result.capped_count == 1
        # Exactly two archives happened.
        archived = storage.get_user_playbooks(
            user_id="u1", status_filter=[Status.ARCHIVED]
        )
        assert len(archived) == 2

    def test_no_change_decisions_do_not_count_against_cap(
        self, request_context, service, llm_client
    ):
        """no_change decisions don't consume cap budget or bump capped_count."""
        from reflexio.models.config_schema import Config, ReflectionConfig

        cfg = Config.model_validate(
            {
                "storage_config": {"db_path": None},
                "window_size": 5,
                "stride_size": 2,
                "reflection_config": ReflectionConfig(
                    max_revisions_per_pass=1
                ).model_dump(),
            }
        )
        request_context.configurator = MagicMock()
        request_context.configurator.get_config.return_value = cfg

        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="rule one")
        _seed_playbook(storage, 2, "u1", content="rule two")

        cites = [
            Citation(kind="playbook", real_id="1"),
            Citation(kind="playbook", real_id="2"),
        ]
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "Assistant", "hello", citations=cites),
                _make_interaction("u1", "r1", "User", "ok"),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    reason="still correct",  # no_change
                ),
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="2",
                    new_content="revised two",  # revision, within cap
                    new_rationale="rewritten rule two",
                    reason="needed update",
                ),
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.no_change_count == 1
        assert result.revised_count == 1
        assert result.capped_count == 0


class TestFieldDerivableCounters:
    def test_trigger_revision_bumps_trigger_revised_count(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_trigger="when Z instead",
                    reason="trigger was too broad",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert result.trigger_revised_count == 1
        assert result.content_revised_count == 0
        assert result.ttl_changed_count == 0

    def test_content_revision_bumps_content_revised_count(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="sharper rule",
                    new_rationale="rewritten after content proved wrong",
                    reason="content was wrong",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert result.content_revised_count == 1
        assert result.trigger_revised_count == 0
        assert result.ttl_changed_count == 0

    def test_ttl_change_bumps_ttl_changed_count(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_profile(storage, "u1", "p1")

        cite = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_profile_time_to_live=ProfileTimeToLive.ONE_QUARTER,
                    reason="preference is temporary",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert result.ttl_changed_count == 1
        assert result.content_revised_count == 0
        assert result.trigger_revised_count == 0

    def test_combined_revision_bumps_multiple_counters(
        self, request_context, service, llm_client
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="sharper rule",
                    new_trigger="when Z instead",
                    new_rationale="rewritten after both content and trigger misfired",
                    reason="both wrong",
                )
            ]
        )

        result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert result.content_revised_count == 1
        assert result.trigger_revised_count == 1
        assert result.ttl_changed_count == 0

    def test_edit_magnitude_logged_for_applied_revision(
        self, request_context, service, llm_client, caplog
    ):
        _set_config(request_context)
        storage = request_context.storage
        _seed_playbook(storage, 1, "u1", content="old rule")

        cite = Citation(kind="playbook", real_id="1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite]),
            ],
        )

        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="playbook",
                    target_id="1",
                    new_content="a much longer replacement rule",
                    new_rationale="rewritten with a longer rule",
                    reason="needed update",
                )
            ]
        )

        with caplog.at_level(
            "INFO",
            logger="reflexio.server.services.reflection.reflection_service",
        ):
            result = service.run(ReflectionServiceRequest(user_id="u1"))

        assert result.revised_count == 1
        assert "event=reflection_edit_magnitude" in caplog.text
        assert "target_kind=playbook" in caplog.text
        assert "target_id=1" in caplog.text
        # old "old rule" = 8 chars, new = 30 chars, delta = 22.
        assert "old_len=8" in caplog.text
        assert "new_len=30" in caplog.text
        assert "delta=22" in caplog.text
