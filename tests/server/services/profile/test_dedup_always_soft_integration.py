"""Integration tests: dedup removals are ALWAYS soft-superseded (never hard-deleted).

These tests guard the central invariant of the "always soft" dedup fix:

    A dedup removal either soft-supersedes (tombstone + ONE status_change/
    superseded lineage event under the run's request_id), OR nothing happens —
    never a hard_delete for a dedup removal. The change log is then rebuilt from
    those lineage events (reconstruct_profile_change_log); the legacy
    ``profile_change_logs`` table is no longer written.

They replace the deleted ``test_dedup_soft_delete_integration.py`` (which was
built around the now-removed ``is_dedup_soft_delete_enabled`` flag and the
hard-delete fallback).

Test groups:
- A: happy-path with REAL SQLiteStorage → soft tombstone + one superseded event,
     reconstruction reflects the removal, and the legacy table stays unwritten.
- B: failure-path (mocked storage) — supersede raises → no hard-delete, no legacy
     write, run does not raise.
- C: empty request_id (mocked storage) — fail-loud anomaly, NO removal at all.
- D: regression guard — the dedup path never calls delete_user_profile.

The service-level branch logic is backend-independent (cross-backend
supersede_profiles_by_ids coverage lives in the storage contract tier), so
these tests are sqlite-only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.enums import ProfileTimeToLive, Status
from reflexio.models.api_schema.service_schemas import UserProfile
from reflexio.server.services.profile.profile_generation_service import (
    ProfileGenerationService,
    ProfileGenerationServiceConfig,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_DEDUP_FLAG = "reflexio.server.site_var.feature_flags.is_deduplicator_enabled"
_DEDUP_CLS = "reflexio.server.services.profile.profile_deduplicator.ProfileDeduplicator"
_CAPTURE_ANOMALY = (
    "reflexio.server.services.profile.profile_generation_service.capture_anomaly"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_profile(
    user_id: str,
    profile_id: str,
    content: str = "some content",
    status: Status | None = None,
    generated_from_request_id: str = "",
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


def _build_service(
    storage,
    *,
    org_id: str,
    user_id: str,
    request_id: str,
) -> ProfileGenerationService:
    """Construct a ProfileGenerationService over the given storage.

    The constructor only reads ``storage``/``org_id``/``configurator`` off the
    request_context; the deduplicator and the LLM are mocked in each test, so a
    MagicMock request_context wrapping the (real or mock) storage is sufficient.
    """
    from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

    ctx = MagicMock()
    ctx.storage = storage
    ctx.org_id = org_id
    ctx.configurator = MagicMock()
    ctx.prompt_manager = MagicMock()

    llm = LiteLLMClient(LiteLLMConfig(model="gpt-4o-mini"))
    svc = ProfileGenerationService(llm_client=llm, request_context=ctx)
    svc.service_config = ProfileGenerationServiceConfig(
        user_id=user_id,
        request_id=request_id,
        source="test",
    )
    return svc


def _patch_dedup(*, all_new, existing_ids, superseded):
    """Patch is_deduplicator_enabled=True and the ProfileDeduplicator class.

    Returns a context-manager-yielding helper used as ``with _patch_dedup(...):``.
    """
    mock_dedup = MagicMock()
    mock_dedup.deduplicate.return_value = (all_new, existing_ids, superseded)
    mock_dedup_cls = patch(_DEDUP_CLS)
    flag = patch(_DEDUP_FLAG, return_value=True)
    return mock_dedup, mock_dedup_cls, flag


# ===========================================================================
# A. Happy-path: REAL sqlite storage + reconstruction MATCH (core invariant)
# ===========================================================================


def test_dedup_removal_soft_supersedes_and_reconstructs(tmp_path) -> None:
    """A committed dedup removal leaves a tombstone + one superseded event, the
    legacy table stays unwritten, and reconstruction reflects the removal."""
    org_id = "always-soft-org-A"
    user_id = "u_A"
    request_id = "manual_softA1"

    storage = SQLiteStorage(org_id=org_id, db_path=str(tmp_path / "a.db"))
    storage.migrate()

    p_old = _make_profile(
        user_id, "p_old_A", content="old facts", generated_from_request_id="seed"
    )
    storage.add_user_profile(user_id, [p_old])

    p_new = _make_profile(
        user_id,
        "p_new_A",
        content="new facts",
        generated_from_request_id=request_id,
    )

    svc = _build_service(storage, org_id=org_id, user_id=user_id, request_id=request_id)
    mock_dedup, mock_dedup_cls, flag = _patch_dedup(
        all_new=[p_new],
        existing_ids=[p_old.profile_id],
        superseded=[p_old],
    )
    with mock_dedup_cls as cls, flag:
        cls.return_value = mock_dedup
        svc._finalize_extracted_items([p_new])

    # (i) P_old's row is SUPERSEDED, not absent (content preserved as tombstone).
    tomb = storage.get_profile_by_id("p_old_A", include_tombstones=True)
    assert tomb is not None
    assert tomb.status == Status.SUPERSEDED
    assert tomb.content == "old facts"
    # And hidden from default reads.
    assert storage.get_profile_by_id("p_old_A") is None

    # (ii) Exactly ONE status_change/superseded event for P_old under THIS request_id.
    events = storage.get_lineage_events(entity_id="p_old_A")
    superseded_events = [
        e
        for e in events
        if e.op == "status_change"
        and e.to_status == "superseded"
        and e.request_id == request_id
    ]
    assert len(superseded_events) == 1

    # (iii) ZERO hard_delete events for P_old.
    hard_deletes = [e for e in events if e.op == "hard_delete"]
    assert hard_deletes == []

    # (iv) The legacy profile_change_logs table is NO LONGER written.
    assert storage.get_profile_change_logs() == []

    # (v) Reconstruction (served by the endpoint) reflects the removal + addition
    # for this request_id, rebuilt purely from lineage events.
    recon = reconstruct_profile_change_log(storage)
    assert recon.success
    recon_log = next(
        log for log in recon.profile_change_logs if log.request_id == request_id
    )
    assert [p.profile_id for p in recon_log.removed_profiles] == ["p_old_A"]
    assert [p.profile_id for p in recon_log.added_profiles] == ["p_new_A"]


# ===========================================================================
# B. Failure-path (mock storage): atomicity guard — no phantom removal
# ===========================================================================


def test_supersede_raises_does_not_hard_delete_or_write_legacy() -> None:
    """If supersede raises, the run does not raise, does not fall back to a
    hard-delete, and never writes the (frozen) legacy change-log table."""
    mock_storage = MagicMock()
    mock_storage.supersede_profiles_by_ids.side_effect = RuntimeError("boom")

    p_old = _make_profile("u_B", "old_B")
    p_new = _make_profile("u_B", "new_B", generated_from_request_id="run_B")

    svc = _build_service(
        mock_storage, org_id="org_B", user_id="u_B", request_id="run_B"
    )
    mock_dedup, mock_dedup_cls, flag = _patch_dedup(
        all_new=[p_new],
        existing_ids=["old_B"],
        superseded=[p_old],
    )
    with mock_dedup_cls as cls, flag:
        cls.return_value = mock_dedup
        # Must NOT raise — the exception is swallowed and logged.
        svc._finalize_extracted_items([p_new])

    mock_storage.supersede_profiles_by_ids.assert_called_once()
    mock_storage.delete_user_profile.assert_not_called()
    mock_storage.add_profile_change_log.assert_not_called()


def test_supersede_called_with_full_intent_and_no_legacy_write() -> None:
    """The service forwards the deduplicator's full removal intent to
    supersede_profiles_by_ids (the committed-subset semantics now live in storage
    + reconstruction, not the service) and never writes the legacy table."""
    mock_storage = MagicMock()
    mock_storage.supersede_profiles_by_ids.return_value = ["old_1"]

    superseded = [_make_profile("u_B2", "old_1"), _make_profile("u_B2", "old_2")]
    p_new = _make_profile("u_B2", "new_B2", generated_from_request_id="run_B2")

    svc = _build_service(
        mock_storage, org_id="org_B2", user_id="u_B2", request_id="run_B2"
    )
    mock_dedup, mock_dedup_cls, flag = _patch_dedup(
        all_new=[p_new],
        existing_ids=["old_1", "old_2"],
        superseded=superseded,
    )
    with mock_dedup_cls as cls, flag:
        cls.return_value = mock_dedup
        svc._finalize_extracted_items([p_new])

    mock_storage.supersede_profiles_by_ids.assert_called_once_with(
        user_id="u_B2",
        profile_ids=["old_1", "old_2"],
        request_id="run_B2",
    )
    mock_storage.add_profile_change_log.assert_not_called()


# ===========================================================================
# C. Empty request_id (mock storage): fail-loud anomaly, NO removal
# ===========================================================================


def test_empty_request_id_skips_removal_and_fires_anomaly() -> None:
    """Empty request_id: supersede NOT called, delete_user_profile NOT called,
    capture_anomaly fires, and the legacy log records NO removal.

    Replaces the old test_empty_request_id_falls_back_to_hard_delete — an empty
    request_id must NEVER trigger a destructive hard-delete.
    """
    mock_storage = MagicMock()

    p_old = _make_profile("u_C", "old_C")
    p_new = _make_profile("u_C", "new_C")

    svc = _build_service(mock_storage, org_id="org_C", user_id="u_C", request_id="")
    mock_dedup, mock_dedup_cls, flag = _patch_dedup(
        all_new=[p_new],
        existing_ids=["old_C"],
        superseded=[p_old],
    )
    with mock_dedup_cls as cls, flag, patch(_CAPTURE_ANOMALY) as mock_anomaly:
        cls.return_value = mock_dedup
        svc._finalize_extracted_items([p_new])

    mock_storage.supersede_profiles_by_ids.assert_not_called()
    mock_storage.delete_user_profile.assert_not_called()

    mock_anomaly.assert_called_once()
    assert mock_anomaly.call_args[0][0] == "lineage.dedup.missing_request_id"

    # No legacy change-log row written (the table is frozen) and no removal at all.
    mock_storage.add_profile_change_log.assert_not_called()


# ===========================================================================
# D. Regression guard: dedup path never hard-deletes
# ===========================================================================


def test_success_path_supersedes_and_never_hard_deletes() -> None:
    """The dedup removal path calls supersede_profiles_by_ids and NEVER calls
    delete_user_profile (the legitimate hard-delete callers are elsewhere)."""
    mock_storage = MagicMock()
    mock_storage.supersede_profiles_by_ids.return_value = ["old_D"]

    p_old = _make_profile("u_D", "old_D")
    p_new = _make_profile("u_D", "new_D", generated_from_request_id="run_D")

    svc = _build_service(
        mock_storage, org_id="org_D", user_id="u_D", request_id="run_D"
    )
    mock_dedup, mock_dedup_cls, flag = _patch_dedup(
        all_new=[p_new],
        existing_ids=["old_D"],
        superseded=[p_old],
    )
    with mock_dedup_cls as cls, flag:
        cls.return_value = mock_dedup
        svc._finalize_extracted_items([p_new])

    mock_storage.supersede_profiles_by_ids.assert_called_once_with(
        user_id="u_D",
        profile_ids=["old_D"],
        request_id="run_D",
    )
    mock_storage.delete_user_profile.assert_not_called()
