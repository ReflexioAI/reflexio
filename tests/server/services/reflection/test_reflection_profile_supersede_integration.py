"""Integration tests: reflection profile edit routes through supersede_record.

After a reflection pass that revises a cited profile, the cited (old) profile
must become status=SUPERSEDED with a ``superseded_by`` pointer, and a ``revise``
lineage event must exist for the successor.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.entities import Citation, UserProfile
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.reflection.reflection_service_utils import (
    ReflectionDecision,
    ReflectionOutput,
    ReflectionServiceRequest,
)
from reflexio.server.services.reflection.service import ReflectionService

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures — reused from test_reflection_service.py pattern
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


def _make_interaction(
    user_id: str,
    request_id: str,
    role: str,
    content: str,
    citations: list | None = None,
):
    from reflexio.models.api_schema.domain.entities import Interaction

    return Interaction(
        user_id=user_id,
        request_id=request_id,
        role=role,
        content=content,
        created_at=int(datetime.now(UTC).timestamp()),
        citations=citations or [],
    )


def _seed_request_with_interactions(storage, user_id, request_id, interactions):
    from reflexio.models.api_schema.domain.entities import Request

    storage.add_request(
        Request(
            request_id=request_id,
            user_id=user_id,
            session_id="test_session",
            source="cli",
            agent_version="v1",
        )
    )
    storage.add_user_interactions_bulk(user_id=user_id, interactions=interactions)


def _set_config(request_context, **overrides):
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReflectionProfileSupersede:
    """Reflection profile replacement uses supersede_record, not archive."""

    @pytest.fixture
    def request_context(self, tmp_path):
        from reflexio.server.llm.litellm_client import LiteLLMClient
        from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

        with (
            patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
            patch.object(
                LiteLLMClient,
                "get_embeddings",
                side_effect=lambda texts, *_args, **_kwargs: [
                    [0.0] * 512 for _ in texts
                ],
            ),
        ):
            ctx = RequestContext(org_id="test_org", storage_base_dir=str(tmp_path))
            yield ctx

    @pytest.fixture
    def llm_client(self):
        client = MagicMock()
        client.generate_chat_response.return_value = ReflectionOutput(decisions=[])
        return client

    @pytest.fixture
    def service(self, request_context, llm_client):
        return ReflectionService(request_context=request_context, llm_client=llm_client)

    def test_cited_profile_becomes_superseded_with_pointer_and_revise_event(
        self, request_context, service, llm_client
    ):
        """After reflection revises a profile:
        (a) cited profile is SUPERSEDED with superseded_by == new profile id
        (b) a 'revise' lineage event exists for the successor profile
        """
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
                    reason="user contradicted earlier preference",
                )
            ]
        )

        result = service.run(
            ReflectionServiceRequest(user_id="u1", request_id="req-reflect-1")
        )
        assert result.ran is True
        assert result.revised_count == 1

        # (a) The old profile must be SUPERSEDED — not ARCHIVED — with a pointer.
        superseded = storage.get_user_profile("u1", status_filter=[Status.SUPERSEDED])
        assert len(superseded) == 1
        old = superseded[0]
        assert old.profile_id == "p1"
        assert old.status == Status.SUPERSEDED

        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 1
        new_id = current[0].profile_id
        assert new_id != "p1"
        assert current[0].content == "new content"

        # superseded_by pointer references the new current profile.
        assert old.superseded_by == new_id

        # (b) A 'revise' lineage event must exist for the successor.
        events = storage.get_lineage_events(entity_type="profile", entity_id=new_id)
        revise_events = [e for e in events if e.op == "revise"]
        assert len(revise_events) >= 1
        revise_evt = revise_events[0]
        assert revise_evt.actor == "reflection"
        assert "p1" in revise_evt.source_ids
        # Pin that the reflection pass's request_id flows end-to-end into the revise event
        # (relied on by B3's request-id reconstruction).
        assert revise_evt.request_id == "req-reflect-1"

        # Old profile must NOT appear as ARCHIVED (no bleed from old code path).
        archived = storage.get_user_profile("u1", status_filter=[Status.ARCHIVED])
        assert archived == []

    def test_no_supersede_when_no_change(self, request_context, service, llm_client):
        """When LLM returns no_change, no supersede event is written."""
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
        assert result.no_change_count == 1
        assert result.revised_count == 0

        # p1 remains CURRENT, no lineage events for it.
        current = storage.get_user_profile("u1", status_filter=[None])
        assert len(current) == 1
        assert current[0].profile_id == "p1"
        events = storage.get_lineage_events(entity_type="profile", entity_id="p1")
        assert events == []

    def test_two_passes_no_explicit_request_id_produce_distinct_revise_events(
        self, request_context, llm_client
    ):
        """Two reflection passes with no explicit request_id produce two distinct revise events.

        Before F006 fix: request_id defaulted to "", so both passes produced the same
        5-col idempotency key (org, profile, "revise", "") and the second event was
        silently dropped by INSERT OR IGNORE.  After the fix: each ReflectionServiceRequest
        constructed without a request_id gets a fresh uuid4().hex, so both events persist.
        """
        # Use post_horizon_size=0 to disable the horizon filter so both passes can
        # proceed regardless of how many follow-up interactions exist.
        _set_config(request_context, reflection_config={"post_horizon_size": 0})
        storage = request_context.storage

        # --- Pass 1: revise p1 → successor p2 ---
        _seed_profile(storage, "u1", "p1", content="original content")
        cite1 = Citation(kind="profile", real_id="p1")
        _seed_request_with_interactions(
            storage,
            "u1",
            "r1",
            [
                _make_interaction("u1", "r1", "User", "hi"),
                _make_interaction("u1", "r1", "Assistant", "hello", citations=[cite1]),
            ],
        )
        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id="p1",
                    new_content="revised content pass-1",
                    reason="pass 1 revision",
                )
            ]
        )
        svc1 = ReflectionService(request_context=request_context, llm_client=llm_client)
        result1 = svc1.run(ReflectionServiceRequest(user_id="u1"))
        assert result1.revised_count == 1

        # Fetch the successor profile written by pass 1.
        current_after_1 = storage.get_user_profile("u1", status_filter=[None])
        assert len(current_after_1) == 1
        p2_id = current_after_1[0].profile_id
        assert p2_id != "p1"

        # --- Pass 2: revise p2 → successor p3 (stride gate reset by new interactions) ---
        cite2 = Citation(kind="profile", real_id=p2_id)
        _seed_request_with_interactions(
            storage,
            "u1",
            "r2",
            [
                _make_interaction("u1", "r2", "User", "more input"),
                _make_interaction("u1", "r2", "Assistant", "got it", citations=[cite2]),
            ],
        )
        llm_client.generate_chat_response.return_value = ReflectionOutput(
            decisions=[
                ReflectionDecision(
                    target_kind="profile",
                    target_id=p2_id,
                    new_content="revised content pass-2",
                    reason="pass 2 revision",
                )
            ]
        )
        svc2 = ReflectionService(request_context=request_context, llm_client=llm_client)
        result2 = svc2.run(ReflectionServiceRequest(user_id="u1"))
        assert result2.revised_count == 1

        current_after_2 = storage.get_user_profile("u1", status_filter=[None])
        assert len(current_after_2) == 1
        p3_id = current_after_2[0].profile_id
        assert p3_id != p2_id

        # Both successors must have exactly one revise event and the two
        # request_ids must be distinct (no collapsed idempotency key).
        events_p2 = storage.get_lineage_events(entity_type="profile", entity_id=p2_id)
        events_p3 = storage.get_lineage_events(entity_type="profile", entity_id=p3_id)
        revise_p2 = [e for e in events_p2 if e.op == "revise"]
        revise_p3 = [e for e in events_p3 if e.op == "revise"]
        assert len(revise_p2) == 1, "pass-1 revise event must be present"
        assert len(revise_p3) == 1, "pass-2 revise event must be present"
        assert revise_p2[0].request_id != revise_p3[0].request_id, (
            "two passes with no explicit request_id must produce distinct lineage request_ids"
        )
