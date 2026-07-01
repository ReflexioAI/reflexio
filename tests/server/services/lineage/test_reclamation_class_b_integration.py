"""Integration tests: Class B direct-delete sweep in _gc_tick (Task 2.1).

Key invariant: Class B (share-link + pending-tool-call reclamation) runs even
when ``lineage_gc.enabled=False``.  Class A (profile expiry sweep + tombstone GC)
must NOT run when ``lineage_gc.enabled=False``.

Both are gated independently:
  - Class A runs under ``if cfg.lineage_gc.enabled``
  - Class B runs under ``if cfg.expiry_reclamation.enabled``
  - The scheduler STARTS when EITHER flag is True.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.config_schema import ExpiryReclamationConfig, LineageGCConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    maybe_start_lineage_gc,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_ORG_ID = "class_b_gc_test_org"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def org_id() -> str:
    return _ORG_ID


def _make_ctx_factory(
    tmp_path,
    *,
    lineage_gc_enabled: bool,
    expiry_reclamation_enabled: bool,
) -> tuple[SQLiteStorage, object]:
    """Build (storage, factory) pair for the given flag combination."""
    storage = SQLiteStorage(org_id=_ORG_ID, db_path=str(tmp_path / "gc_b_test.db"))
    cfg = SimpleNamespace(
        lineage_gc=LineageGCConfig(enabled=lineage_gc_enabled, tombstone_grace_window_days=1),
        expiry_reclamation=ExpiryReclamationConfig(enabled=expiry_reclamation_enabled),
    )

    def factory(org_id: str) -> RequestContext:
        ctx = RequestContext.__new__(RequestContext)
        ctx.org_id = org_id
        ctx.storage = storage
        ctx.storage_base_dir = None
        configurator = MagicMock()
        configurator.get_config.return_value = cfg
        ctx.configurator = configurator
        return ctx

    return storage, factory


def _seed_expired_share_link(storage: SQLiteStorage, expires_at: int) -> None:
    """Create a share link with the given expiration epoch."""
    storage.create_share_link(
        token="shr_class_b_test",
        resource_type="profile",
        resource_id="r_class_b",
        expires_at=expires_at,
        created_by_email=None,
    )


def _share_link_count(storage: SQLiteStorage) -> int:
    """Return the number of share links in storage."""
    return len(storage.get_share_links())


def _seed_active_profile(storage: SQLiteStorage) -> UserProfile:
    """Seed an active profile with a far-past expiration (eligible for expiry sweep)."""
    p = UserProfile(
        profile_id="p_class_b_active",
        user_id="u_class_b",
        content="test profile",
        last_modified_timestamp=1,
        generated_from_request_id="r_class_b_profile",
        expiration_timestamp=100,  # far in the past
    )
    storage.add_user_profile("u_class_b", [p])
    return p


# ---------------------------------------------------------------------------
# Test 1: Class B reclaims even when lineage_gc is DISABLED
# ---------------------------------------------------------------------------


def test_class_b_runs_when_lineage_gc_disabled(tmp_path, org_id):
    """Class B sweep deletes expired share links when lineage_gc.enabled=False.

    Also asserts that Class A (profile expiry sweep) does NOT run, confirming
    that the two guards are independent.
    """
    storage, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=True,
    )

    # Seed a long-expired share link.
    _seed_expired_share_link(storage, expires_at=1)
    assert _share_link_count(storage) == 1

    # Seed an active profile with a past expiration — if Class A ran it would
    # be tombstoned.
    profile = _seed_active_profile(storage)

    sched = LineageGCScheduler(request_context_factory=factory, bootstrap_org_id=org_id)
    sched._gc_tick([org_id])

    # Class B: share link must be reclaimed.
    assert _share_link_count(storage) == 0, (
        "Class B must delete the expired share link even when lineage_gc is disabled"
    )

    # Class A must NOT have run: the active profile must still be active (not tombstoned).
    row = storage.get_profile_by_id(profile.profile_id, include_tombstones=True)
    assert row is not None, "Profile must still exist — Class A must not have run"
    assert row.status is None, (
        "Profile status must remain None (CURRENT); expire_active_profiles must not "
        "have been called (lineage_gc.enabled=False gates Class A)"
    )


# ---------------------------------------------------------------------------
# Test 2: Both disabled → scheduler does not start
# ---------------------------------------------------------------------------


def test_scheduler_does_not_start_when_both_disabled(tmp_path, org_id):
    """maybe_start_lineage_gc returns None when both flags are False."""
    _, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=False,
    )
    result = maybe_start_lineage_gc(factory, bootstrap_org_id=org_id)  # type: ignore[arg-type]
    assert result is None, (
        "Scheduler must not start when lineage_gc.enabled=False and "
        "expiry_reclamation.enabled=False"
    )


# ---------------------------------------------------------------------------
# Test 3: Scheduler starts when only expiry_reclamation is enabled
# ---------------------------------------------------------------------------


def test_scheduler_starts_when_only_expiry_reclamation_enabled(tmp_path, org_id):
    """maybe_start_lineage_gc starts when expiry_reclamation.enabled=True even
    with lineage_gc.enabled=False."""
    _, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=True,
    )
    sched = maybe_start_lineage_gc(factory, bootstrap_org_id=org_id)  # type: ignore[arg-type]
    assert sched is not None, (
        "Scheduler must start when expiry_reclamation.enabled=True "
        "(even with lineage_gc.enabled=False)"
    )
    sched.stop(timeout_seconds=1.0)


# ---------------------------------------------------------------------------
# Test 4: Class B tick is a no-op when expiry_reclamation is disabled
# ---------------------------------------------------------------------------


def test_class_b_does_not_run_when_disabled(tmp_path, org_id):
    """When expiry_reclamation.enabled=False the share link is NOT deleted."""
    storage, factory = _make_ctx_factory(
        tmp_path,
        lineage_gc_enabled=False,
        expiry_reclamation_enabled=False,
    )
    _seed_expired_share_link(storage, expires_at=1)
    assert _share_link_count(storage) == 1

    sched = LineageGCScheduler(request_context_factory=factory, bootstrap_org_id=org_id)
    sched._gc_tick([org_id])

    assert _share_link_count(storage) == 1, (
        "Class B must not run when expiry_reclamation.enabled=False"
    )
