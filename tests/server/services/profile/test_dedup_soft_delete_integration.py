"""Integration tests for dedup soft-delete + set-based lineage (B3-pre T1).

Tests the new supersede_profiles_by_ids storage method and the branched
_finalize_extracted_items dedup path behind is_dedup_soft_delete_enabled.

Test tiers:
- Storage-level: SQLiteStorage.supersede_profiles_by_ids behavior
- Service-level: _finalize_extracted_items branch (flag ON vs OFF) using mocked storage
- generated_from_request_id verification on the add side
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import (
    ProfileTimeToLive,
    UserProfile,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    user_id: str,
    profile_id: str,
    content: str = "some content",
    status: Status | None = None,
    generated_from_request_id: str = "req_0",
) -> UserProfile:
    return UserProfile(
        user_id=user_id,
        profile_id=profile_id,
        content=content,
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id=generated_from_request_id,
        profile_time_to_live=ProfileTimeToLive.INFINITY,
        source="test",
        status=status,
    )


# ---------------------------------------------------------------------------
# Storage-level tests
# ---------------------------------------------------------------------------


class TestSuperposeProfilesByIds:
    @pytest.fixture
    def db(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512),
        ):
            yield SQLiteStorage(org_id="test_org", db_path=f"{tmp}/test.db")

    def test_basic_soft_delete(self, db: SQLiteStorage) -> None:
        """M removed profiles get status=SUPERSEDED, content intact, row survives."""
        db.add_user_profile(
            "u1",
            [
                _make_profile("u1", "p1", content="profile one"),
                _make_profile("u1", "p2", content="profile two"),
            ],
        )

        count = db.supersede_profiles_by_ids("u1", ["p1", "p2"], request_id="req_abc")

        assert count == 2
        # Row survived with content intact
        p1 = db.get_profile_by_id("p1", include_tombstones=True)
        assert p1 is not None
        assert p1.content == "profile one"
        assert p1.status == Status.SUPERSEDED

        p2 = db.get_profile_by_id("p2", include_tombstones=True)
        assert p2 is not None
        assert p2.content == "profile two"
        assert p2.status == Status.SUPERSEDED

    def test_default_reads_exclude_tombstones(self, db: SQLiteStorage) -> None:
        """get_user_profile (default) and get_profile_by_id exclude SUPERSEDED rows."""
        db.add_user_profile("u1", [_make_profile("u1", "p1")])
        db.supersede_profiles_by_ids("u1", ["p1"], request_id="req_x")

        # Default get excludes tombstones
        assert db.get_user_profile("u1") == []
        assert db.get_profile_by_id("p1") is None

    def test_lineage_event_share_request_id(self, db: SQLiteStorage) -> None:
        """All emitted status_change events share the caller-supplied request_id."""
        db.add_user_profile(
            "u1",
            [
                _make_profile("u1", "pa"),
                _make_profile("u1", "pb"),
                _make_profile("u1", "pc"),
            ],
        )
        req = "shared_req_42"

        db.supersede_profiles_by_ids("u1", ["pa", "pb", "pc"], request_id=req)

        rows = db.conn.execute(
            "SELECT entity_id, op, request_id, to_status FROM lineage_event WHERE op='status_change' AND request_id=?",
            (req,),
        ).fetchall()
        assert len(rows) == 3
        entity_ids = {r["entity_id"] for r in rows}
        assert entity_ids == {"pa", "pb", "pc"}
        for r in rows:
            assert r["to_status"] == "superseded"
            assert r["request_id"] == req

    def test_from_status_derived_not_hardcoded(self, db: SQLiteStorage) -> None:
        """Soft-deleting a PENDING profile records from_status='pending', not None."""
        db.add_user_profile(
            "u1", [_make_profile("u1", "p_pend", status=Status.PENDING)]
        )
        db.supersede_profiles_by_ids("u1", ["p_pend"], request_id="req_pend")

        row = db.conn.execute(
            "SELECT from_status FROM lineage_event WHERE entity_id='p_pend' AND op='status_change'",
        ).fetchone()
        assert row is not None
        assert row["from_status"] == "pending"

    def test_already_superseded_skipped(self, db: SQLiteStorage) -> None:
        """Already-superseded profiles are skipped; rowcount reflects only actual updates."""
        db.add_user_profile("u1", [_make_profile("u1", "p1")])
        db.supersede_profiles_by_ids("u1", ["p1"], request_id="req1")

        # Second call: same id already superseded
        count2 = db.supersede_profiles_by_ids("u1", ["p1"], request_id="req2")
        assert count2 == 0

    def test_nonexistent_id_skipped(self, db: SQLiteStorage) -> None:
        """Non-existent profile ids are silently skipped."""
        count = db.supersede_profiles_by_ids("u1", ["ghost_id"], request_id="req_x")
        assert count == 0

    def test_empty_list_returns_zero(self, db: SQLiteStorage) -> None:
        """Empty profile_ids list returns 0 immediately."""
        count = db.supersede_profiles_by_ids("u1", [], request_id="req_x")
        assert count == 0

    def test_user_id_scoped(self, db: SQLiteStorage) -> None:
        """Soft-delete for user u1 does NOT affect user u2's profile."""
        # profile_id is the primary key in SQLite storage — each user needs distinct ids.
        db.add_user_profile("u1", [_make_profile("u1", "pid_u1")])
        db.add_user_profile("u2", [_make_profile("u2", "pid_u2")])

        count = db.supersede_profiles_by_ids("u1", ["pid_u1"], request_id="req_scope")

        assert count == 1
        # u2's profile is untouched
        u2_profiles = db.get_user_profile("u2")
        assert len(u2_profiles) == 1
        assert u2_profiles[0].status is None  # still CURRENT


# ---------------------------------------------------------------------------
# Service-level tests (_finalize_extracted_items branch)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def request_context(mock_storage):
    ctx = MagicMock()
    ctx.storage = mock_storage
    ctx.org_id = "svc_org"
    ctx.prompt_manager = MagicMock()
    ctx.configurator = MagicMock()
    return ctx


@pytest.fixture
def service(request_context):
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
    from reflexio.server.services.profile.profile_generation_service import (
        ProfileGenerationService,
        ProfileGenerationServiceConfig,
    )

    llm = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    svc = ProfileGenerationService(llm_client=llm, request_context=request_context)
    svc.service_config = ProfileGenerationServiceConfig(
        user_id="u1",
        request_id="run_req_1",
        source="test",
    )
    yield svc


def _make_new_profile(idx: int, request_id: str = "run_req_1") -> UserProfile:
    return _make_profile("u1", f"new_{idx}", generated_from_request_id=request_id)


class TestFinalizeExtractedItemsFlagOff:
    """Flag OFF (default): existing hard-delete path is byte-for-byte unchanged."""

    def test_flag_off_uses_delete_user_profile(self, service, mock_storage) -> None:
        """When flag is OFF, delete_user_profile is called for each superseded id."""
        removed_ids = ["old_1", "old_2"]
        superseded = [_make_profile("u1", i) for i in removed_ids]

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_dedup_soft_delete_enabled",
                return_value=False,
            ),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
            ) as mock_dedup_cls,
        ):
            mock_dedup = MagicMock()
            mock_dedup.deduplicate.return_value = (
                [_make_new_profile(0)],
                removed_ids,
                superseded,
            )
            mock_dedup_cls.return_value = mock_dedup
            service._finalize_extracted_items([_make_new_profile(0)])

        assert mock_storage.delete_user_profile.call_count == 2
        mock_storage.supersede_profiles_by_ids.assert_not_called()

    def test_flag_off_emits_no_set_based_lineage(self, service, mock_storage) -> None:
        """When flag is OFF, supersede_profiles_by_ids is never touched."""
        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_dedup_soft_delete_enabled",
                return_value=False,
            ),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
            ) as mock_dedup_cls,
        ):
            mock_dedup = MagicMock()
            mock_dedup.deduplicate.return_value = (
                [_make_new_profile(0)],
                ["old_a"],
                [_make_profile("u1", "old_a")],
            )
            mock_dedup_cls.return_value = mock_dedup
            service._finalize_extracted_items([_make_new_profile(0)])

        mock_storage.supersede_profiles_by_ids.assert_not_called()


class TestFinalizeExtractedItemsFlagOn:
    """Flag ON: dedup removes via supersede_profiles_by_ids, not delete_user_profile."""

    def test_flag_on_uses_supersede_not_delete(self, service, mock_storage) -> None:
        """When flag ON, supersede_profiles_by_ids is called, delete_user_profile is not."""
        removed_ids = ["old_1", "old_2", "old_3"]
        superseded = [_make_profile("u1", i) for i in removed_ids]

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_dedup_soft_delete_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
            ) as mock_dedup_cls,
        ):
            mock_dedup = MagicMock()
            mock_dedup.deduplicate.return_value = (
                [_make_new_profile(0), _make_new_profile(1)],
                removed_ids,
                superseded,
            )
            mock_dedup_cls.return_value = mock_dedup
            service._finalize_extracted_items(
                [_make_new_profile(0), _make_new_profile(1)]
            )

        mock_storage.supersede_profiles_by_ids.assert_called_once_with(
            user_id="u1",
            profile_ids=removed_ids,
            request_id="run_req_1",
        )
        mock_storage.delete_user_profile.assert_not_called()

    def test_empty_request_id_falls_back_to_hard_delete(
        self, service, mock_storage
    ) -> None:
        """Flag ON + empty request_id falls back to hard-delete (no set-based lineage under '')."""
        from reflexio.server.services.profile.profile_generation_service import (
            ProfileGenerationServiceConfig,
        )

        service.service_config = ProfileGenerationServiceConfig(
            user_id="u1",
            request_id="",  # empty
            source="test",
        )
        removed_ids = ["old_1"]
        superseded = [_make_profile("u1", "old_1")]

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_dedup_soft_delete_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
            ) as mock_dedup_cls,
        ):
            mock_dedup = MagicMock()
            mock_dedup.deduplicate.return_value = (
                [_make_new_profile(0, request_id="")],
                removed_ids,
                superseded,
            )
            mock_dedup_cls.return_value = mock_dedup
            service._finalize_extracted_items([_make_new_profile(0, request_id="")])

        # Falls back to hard-delete, NOT set-based lineage under ""
        mock_storage.supersede_profiles_by_ids.assert_not_called()
        mock_storage.delete_user_profile.assert_called_once()

    def test_added_profiles_carry_generated_from_request_id(
        self, service, mock_storage
    ) -> None:
        """Added profiles from the deduplicator carry generated_from_request_id == request_id."""
        req_id = "run_req_1"
        # Simulate deduplicator output: new profiles have generated_from_request_id set
        new_p = _make_new_profile(0, request_id=req_id)
        assert new_p.generated_from_request_id == req_id, (
            "STOP: deduplicator does NOT set generated_from_request_id on added profiles. "
            "Reconstruction depends on this column — do not proceed without fixing T2."
        )

        with (
            patch(
                "reflexio.server.site_var.feature_flags.is_dedup_soft_delete_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.site_var.feature_flags.is_deduplicator_enabled",
                return_value=True,
            ),
            patch(
                "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
            ) as mock_dedup_cls,
        ):
            mock_dedup = MagicMock()
            mock_dedup.deduplicate.return_value = ([new_p], [], [])
            mock_dedup_cls.return_value = mock_dedup
            service._finalize_extracted_items([new_p])

        saved_profiles = mock_storage.add_user_profile.call_args[0][1]
        assert all(p.generated_from_request_id == req_id for p in saved_profiles)
