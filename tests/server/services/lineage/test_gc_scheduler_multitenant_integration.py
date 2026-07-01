"""Integration tests: managed-tenant reclamation via org_id_provider (Task 3.4).

In managed multi-tenant mode ``storage.list_org_ids()`` raises
``NotImplementedError`` on Supabase, so ``_discover_org_ids`` degrades to the
bootstrap org only and tenant orgs' expired rows leak. Task 3.4 makes org
discovery injectable: when an ``org_id_provider`` is supplied (the enterprise
supplies a tenant-enumerating one), the profile expiry sweep (Class A) and the
plain-row sweeps (Class B: share links + pending tool calls) reach EVERY tenant.

These tests prove:
1. Multi-tenant reclamation: with a provider returning two tenant orgs and a
   per-org SQLite factory, one tick reclaims BOTH orgs' rows (not just bootstrap).
2. The provider is actually consulted (the enterprise enumeration path), and
   ``storage.list_org_ids()`` is NOT used when a provider is present.
3. Per-org failure isolation still holds under the provider path.
4. The default (no provider) path is unchanged: ``storage.list_org_ids()`` is used.
5. ``maybe_start_lineage_gc`` picks up a module-level provider hook.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.config_schema import ExpiryReclamationConfig, LineageGCConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.services.lineage import gc_scheduler as gc_module
from reflexio.server.services.lineage.gc_scheduler import (
    LineageGCScheduler,
    maybe_start_lineage_gc,
    set_org_id_provider,
)
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import (
    PendingToolCallRecord,
    PendingToolCallStatus,
    build_pending_tool_call_dedup_key,
    build_scope_hash,
)

pytestmark = pytest.mark.integration


def _both_enabled_config() -> SimpleNamespace:
    """Config with Class A (lineage_gc) and Class B (expiry_reclamation) enabled."""
    return SimpleNamespace(
        lineage_gc=LineageGCConfig(enabled=True, tombstone_grace_window_days=1),
        expiry_reclamation=ExpiryReclamationConfig(enabled=True),
    )


def _seed_org(storage: SQLiteStorage, org_id: str) -> str:
    """Seed one org with a TTL-expired profile, expired share link, expired call.

    Returns:
        str: The profile_id seeded (for tombstone assertions).
    """
    profile_id = f"p_{org_id}"
    storage.add_user_profile(
        f"u_{org_id}",
        [
            UserProfile(
                profile_id=profile_id,
                user_id=f"u_{org_id}",
                content="tenant profile",
                last_modified_timestamp=1,
                generated_from_request_id=f"r_{org_id}",
                expiration_timestamp=100,  # far in the past → eligible for expiry sweep
            )
        ],
    )
    storage.create_share_link(
        token=f"shr_{org_id}",
        resource_type="profile",
        resource_id=f"res_{org_id}",
        expires_at=1,  # long expired
        created_by_email=None,
    )
    now = datetime.now(UTC)
    scope = {"org_id": org_id, "scope_kind": "org"}
    storage.create_pending_tool_call(
        PendingToolCallRecord(
            id=f"call_{org_id}",
            org_id=org_id,
            scope=scope,
            scope_hash=build_scope_hash(scope),
            tool_name="ask_human",
            dedup_key=build_pending_tool_call_dedup_key(
                tool_name="ask_human", question_text=f"q_{org_id}"
            ),
            status=PendingToolCallStatus("expired"),
            question_text=f"q_{org_id}",
            expires_at=now - timedelta(days=2),  # past the 1-day Class B grace
            cache_until=now - timedelta(days=2),
        )
    )
    return profile_id


def _assert_reclaimed(storage: SQLiteStorage, org_id: str, profile_id: str) -> None:
    """Assert Class A + Class B reclaimed every seeded row for this org."""
    row = storage.get_profile_by_id(profile_id, include_tombstones=True)
    assert row is not None, f"{org_id}: profile row should still exist (tombstoned)"
    assert row.status is not None, (
        f"{org_id}: Class A must tombstone the TTL-expired active profile"
    )
    assert storage.get_share_links() == [], (
        f"{org_id}: Class B must delete the expired share link"
    )
    assert storage.get_pending_tool_call(f"call_{org_id}") is None, (
        f"{org_id}: Class B must delete the expired pending tool call"
    )


@pytest.fixture
def multitenant(tmp_path):
    """Two tenant orgs, each backed by its own SQLite storage + shared config."""
    storages = {
        org: SQLiteStorage(org_id=org, db_path=str(tmp_path / f"{org}.db"))
        for org in ("org_a", "org_b")
    }
    profile_ids = {org: _seed_org(s, org) for org, s in storages.items()}
    cfg = _both_enabled_config()

    def factory(org_id: str) -> RequestContext:
        ctx = RequestContext.__new__(RequestContext)
        ctx.org_id = org_id
        ctx.storage = storages[org_id]
        ctx.storage_base_dir = None
        configurator = MagicMock()
        configurator.get_config.return_value = cfg
        ctx.configurator = configurator
        return ctx

    return storages, profile_ids, factory


def test_provider_sweeps_all_tenant_orgs(multitenant):
    """With a provider returning both tenants, one tick reclaims BOTH orgs' rows."""
    storages, profile_ids, factory = multitenant

    provider_calls: list[int] = []

    def provider() -> list[str]:
        provider_calls.append(1)
        return ["org_a", "org_b"]

    sched = LineageGCScheduler(
        request_context_factory=factory,
        bootstrap_org_id="org_a",
        org_id_provider=provider,
    )

    bootstrap_ctx = factory("org_a")
    org_ids = sched._discover_org_ids(bootstrap_ctx)
    assert provider_calls, "provider must be consulted (enterprise enumeration path)"
    assert set(org_ids) == {"org_a", "org_b"}

    sched._gc_tick(org_ids)

    for org, storage in storages.items():
        _assert_reclaimed(storage, org, profile_ids[org])


def test_provider_short_circuits_storage_list_org_ids(multitenant):
    """When a provider is set, storage.list_org_ids() is NOT consulted."""
    storages, _profile_ids, factory = multitenant
    # Make list_org_ids explode so any accidental use is caught.
    for storage in storages.values():
        storage.list_org_ids = MagicMock(  # type: ignore[method-assign]
            side_effect=AssertionError("list_org_ids must not be called with a provider")
        )

    sched = LineageGCScheduler(
        request_context_factory=factory,
        bootstrap_org_id="org_a",
        org_id_provider=lambda: ["org_a", "org_b"],
    )
    org_ids = sched._discover_org_ids(factory("org_a"))
    assert set(org_ids) == {"org_a", "org_b"}


def test_provider_path_isolates_per_org_failure(tmp_path):
    """One org raising in the tick does not stop the other org's reclamation."""
    good = SQLiteStorage(org_id="org_good", db_path=str(tmp_path / "good.db"))
    good_profile = _seed_org(good, "org_good")
    cfg = _both_enabled_config()

    def factory(org_id: str) -> RequestContext:
        ctx = RequestContext.__new__(RequestContext)
        ctx.org_id = org_id
        if org_id == "org_bad":
            ctx.storage = MagicMock()
            ctx.storage.expire_active_profiles.side_effect = RuntimeError("boom")
        else:
            ctx.storage = good
        ctx.storage_base_dir = None
        configurator = MagicMock()
        configurator.get_config.return_value = cfg
        ctx.configurator = configurator
        return ctx

    sched = LineageGCScheduler(
        request_context_factory=factory,
        bootstrap_org_id="org_bad",
        org_id_provider=lambda: ["org_bad", "org_good"],
    )
    # Must not raise — per-org failure is isolated.
    sched._gc_tick(sched._discover_org_ids(factory("org_bad")))
    _assert_reclaimed(good, "org_good", good_profile)


def test_default_path_uses_storage_list_org_ids():
    """Without a provider, _discover_org_ids consults storage.list_org_ids()."""
    storage = MagicMock()
    storage.list_org_ids.return_value = ["org_bootstrap", "org_tenant"]
    bootstrap_ctx = RequestContext.__new__(RequestContext)
    bootstrap_ctx.org_id = "org_bootstrap"
    bootstrap_ctx.storage = storage

    sched = LineageGCScheduler(
        request_context_factory=lambda _org_id: bootstrap_ctx,
        bootstrap_org_id="org_bootstrap",
    )
    assert sched.org_id_provider is None
    org_ids = sched._discover_org_ids(bootstrap_ctx)
    storage.list_org_ids.assert_called_once()
    assert set(org_ids) == {"org_bootstrap", "org_tenant"}


def test_maybe_start_lineage_gc_reads_module_provider_hook(tmp_path):
    """maybe_start_lineage_gc picks up a provider set via set_org_id_provider."""
    storage = SQLiteStorage(org_id="org_a", db_path=str(tmp_path / "hook.db"))
    cfg = _both_enabled_config()

    def factory(org_id: str) -> RequestContext:
        ctx = RequestContext.__new__(RequestContext)
        ctx.org_id = org_id
        ctx.storage = storage
        ctx.storage_base_dir = None
        configurator = MagicMock()
        configurator.get_config.return_value = SimpleNamespace(
            lineage_gc=SimpleNamespace(
                enabled=True, poll_interval_seconds=3600, tombstone_grace_window_days=1
            ),
            expiry_reclamation=cfg.expiry_reclamation,
        )
        ctx.configurator = configurator
        return ctx

    hook = lambda: ["org_a", "org_b"]  # noqa: E731
    set_org_id_provider(hook)
    try:
        sched = maybe_start_lineage_gc(factory, bootstrap_org_id="org_a")  # type: ignore[arg-type]
        assert sched is not None
        assert sched.org_id_provider is hook
        sched.stop(timeout_seconds=1.0)
    finally:
        set_org_id_provider(None)
    assert gc_module._org_id_provider_hook is None
