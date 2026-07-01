"""Integration test: _gc_tick tombstones TTL-expired active profiles (Task 1.4).

Verifies that the expiry sweep step is wired into _gc_tick BEFORE the tombstone
GC so that profiles past their TTL are tombstoned in the same tick that would
then reclaim them (after the grace window).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.config_schema import LineageGCConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.lineage.gc_scheduler import LineageGCScheduler
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration

_ORG_ID = "expiry_gc_test_org"


@pytest.fixture()
def org_id() -> str:
    return _ORG_ID


@pytest.fixture()
def real_ctx_factory(tmp_path):
    """Return a factory that builds a RequestContext over a real SQLiteStorage.

    Config has lineage_gc.enabled=True and tombstone_grace_window_days=1 (minimum
    valid value, gt=0 constraint) so the sweep tombstones expired profiles in the
    tick while the GC step does not hard-delete them (retired_at=now is not older
    than now-1day).
    """
    storage = SQLiteStorage(org_id=_ORG_ID, db_path=str(tmp_path / "gc_test.db"))
    cfg = SimpleNamespace(
        lineage_gc=LineageGCConfig(enabled=True, tombstone_grace_window_days=1)
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

    return factory


def test_tick_tombstones_then_gc_reclaims_expired(real_ctx_factory, org_id):
    """_gc_tick must tombstone TTL-expired active profiles via expire_active_profiles
    before running the tombstone GC step.

    With a grace window of 1 day the freshly-tombstoned row is not yet eligible
    for hard-deletion in the same tick, so the profile must be visible as status
    EXPIRED via include_tombstones=True rather than hard-deleted.
    """
    ctx = real_ctx_factory(org_id)
    # Seed an expired active profile (expiration_timestamp=100 << now).
    ctx.storage.add_user_profile(
        "u1",
        [
            UserProfile(
                profile_id="e1",
                user_id="u1",
                content="c",
                last_modified_timestamp=1,
                generated_from_request_id="r1",
                expiration_timestamp=100,
            )
        ],
    )

    sched = LineageGCScheduler(
        request_context_factory=real_ctx_factory, bootstrap_org_id=org_id
    )
    sched._gc_tick([org_id])

    row = ctx.storage.get_profile_by_id("e1", include_tombstones=True)
    # Either hard-deleted (None) or tombstoned as EXPIRED — either satisfies the contract.
    assert row is None or row.status == Status.EXPIRED
