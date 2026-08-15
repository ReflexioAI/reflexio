"""Tests for create_app(capabilities=CapabilityRegistry(...)) wiring.

Covers:
- capability routers are mounted and reachable
- on_startup / on_shutdown lifecycle runs
- on_startup raise is fail-loud (propagates out of the lifespan)
- partial-cleanup invariant: already-started caps shut down; never-started caps skipped
- billing-gate double-wire raises ValueError at construction time
- capabilities=None leaves behavior unchanged (smoke)
"""

from collections.abc import Callable

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from reflexio.server import usage_metrics
from reflexio.server.api import create_app
from reflexio.server.extensions import AppContext, Capability, CapabilityRegistry
from reflexio.server.usage_metrics import (
    UsageEventDeliveryError,
    UsageEventDeliveryStatus,
)


class RouterCap(Capability):
    name = "rc"

    def routers(self, role: str) -> list[APIRouter]:
        r = APIRouter()

        @r.get("/api/_cap_probe")
        def probe() -> dict:
            return {"ok": True}

        return [r]


class LifecycleCap(Capability):
    name = "lc"
    started = False
    stopped = False

    async def on_startup(self, ctx: AppContext) -> None:
        LifecycleCap.started = True

    async def on_shutdown(self) -> None:
        LifecycleCap.stopped = True


class BoomCap(Capability):
    name = "boom"

    async def on_startup(self, ctx: AppContext) -> None:
        raise RuntimeError("startup failed")


class NeverStartedCap(Capability):
    """A cap positioned after BoomCap; its on_startup is never reached."""

    name = "never"
    shutdown_called = False

    async def on_shutdown(self) -> None:
        NeverStartedCap.shutdown_called = True


def test_capability_routers_are_mounted() -> None:
    app = create_app(
        capabilities=CapabilityRegistry([RouterCap()]), mount_data_plane=False
    )
    with TestClient(app) as c:
        assert c.get("/api/_cap_probe").json() == {"ok": True}


def test_on_startup_and_shutdown_run() -> None:
    LifecycleCap.started = LifecycleCap.stopped = False
    app = create_app(
        capabilities=CapabilityRegistry([LifecycleCap()]), mount_data_plane=False
    )
    with TestClient(app):
        assert LifecycleCap.started is True
    assert LifecycleCap.stopped is True


def test_on_startup_raise_is_fail_loud() -> None:
    app = create_app(
        capabilities=CapabilityRegistry([BoomCap()]), mount_data_plane=False
    )
    with pytest.raises(RuntimeError, match="startup failed"), TestClient(app):
        pass


def test_capabilities_none_is_unchanged() -> None:
    app = create_app()  # legacy path, no capabilities
    assert app is not None


def test_oss_startup_failure_clears_exempt_usage_recorder(monkeypatch) -> None:
    usage_metrics.configure_usage_event_recorder(None)

    def fail_startup_guard() -> None:
        raise RuntimeError("startup guard failed")

    monkeypatch.setattr(
        "reflexio.server.llm.model_defaults.validate_llm_availability",
        fail_startup_guard,
    )

    app = create_app(capabilities=None, mount_data_plane=True)
    with pytest.raises(RuntimeError, match="startup guard failed"), TestClient(app):
        pass

    with pytest.raises(UsageEventDeliveryError) as exc_info:
        usage_metrics.record_usage_event_strict(
            org_id="7",
            event_name="learnings_generated",
            event_category="learning",
        )
    assert exc_info.value.status is UsageEventDeliveryStatus.UNKNOWN


def test_oss_metering_startup_failure_clears_exempt_usage_recorder(
    monkeypatch,
) -> None:
    usage_metrics.configure_usage_event_recorder(None)

    def fail_start_search_metering_worker() -> None:
        raise RuntimeError("startup guard failed")

    monkeypatch.setattr(
        "reflexio.server.services.search_metering_worker.start_search_metering_worker",
        fail_start_search_metering_worker,
    )

    app = create_app(capabilities=None, mount_data_plane=False)
    with pytest.raises(RuntimeError, match="startup guard failed"), TestClient(app):
        pass

    with pytest.raises(UsageEventDeliveryError) as exc_info:
        usage_metrics.record_usage_event_strict(
            org_id="7",
            event_name="learnings_generated",
            event_category="learning",
        )
    assert exc_info.value.status is UsageEventDeliveryStatus.UNKNOWN


def test_partial_cleanup_invariant() -> None:
    """Already-started caps must be shut down; caps that never started must not be."""
    LifecycleCap.started = LifecycleCap.stopped = False
    NeverStartedCap.shutdown_called = False

    # Order: LifecycleCap starts successfully, BoomCap raises, NeverStartedCap never runs.
    app = create_app(
        capabilities=CapabilityRegistry([LifecycleCap(), BoomCap(), NeverStartedCap()]),
        mount_data_plane=False,
    )
    with pytest.raises(RuntimeError, match="startup failed"), TestClient(app):
        pass

    assert LifecycleCap.stopped is True, (
        "already-started cap must be shut down on partial failure"
    )
    assert NeverStartedCap.shutdown_called is False, (
        "cap whose on_startup never ran must NOT be shut down"
    )


def test_billing_gate_double_wire_raises() -> None:
    """Passing billing_gate via both get_billing_gate and capabilities must raise."""

    def my_gate(line: str) -> Callable[..., None]:
        def dep() -> None:
            pass

        return dep

    cap_registry = CapabilityRegistry([], billing_gate=my_gate)
    with pytest.raises(ValueError, match="billing_gate"):
        create_app(get_billing_gate=my_gate, capabilities=cap_registry)


class RoleCaptureCap(Capability):
    """Records the role passed to routers() so tests can inspect it."""

    name = "role_capture"
    seen_role: str | None = None

    def routers(self, role: str) -> list[APIRouter]:
        RoleCaptureCap.seen_role = role
        return []


def test_capabilities_role_overrides_mount_data_plane_derivation() -> None:
    """When capabilities.role is set, create_app must use it, not mount_data_plane."""
    RoleCaptureCap.seen_role = None
    reg = CapabilityRegistry([RoleCaptureCap()], role="data-plane")
    # mount_data_plane=True would normally yield "all" — but capabilities.role wins.
    create_app(capabilities=reg, mount_data_plane=True)
    assert RoleCaptureCap.seen_role == "data-plane", (
        "capabilities.role must override mount_data_plane derivation"
    )


def test_capabilities_role_none_falls_back_to_mount_data_plane() -> None:
    """When capabilities.role is None, create_app falls back to mount_data_plane derivation."""
    RoleCaptureCap.seen_role = None
    # role=None → fallback: mount_data_plane=True → "all"
    reg = CapabilityRegistry([RoleCaptureCap()])
    create_app(capabilities=reg, mount_data_plane=True)
    assert RoleCaptureCap.seen_role == "all"

    RoleCaptureCap.seen_role = None
    reg2 = CapabilityRegistry([RoleCaptureCap()])
    create_app(capabilities=reg2, mount_data_plane=False)
    assert RoleCaptureCap.seen_role == "control-plane"


def test_router_double_mount_is_boot_error() -> None:
    """A capability router also passed via additional_routers must raise ValueError at boot."""
    shared = APIRouter()

    class _Dup(Capability):
        name = "dup"

        def routers(self, role: str) -> list[APIRouter]:
            return [shared]

    reg = CapabilityRegistry([_Dup()], role="all")
    with pytest.raises(ValueError, match="dup"):
        create_app(capabilities=reg, additional_routers=[shared])


def test_router_double_mount_guard_uses_identity_not_equality() -> None:
    """Guard must use 'is' (identity), not '==' (equality).

    Two distinct empty APIRouter() instances compare equal via __eq__ (same routes list),
    so the guard must NOT fire when a capability returns a fresh router that happens to be
    route-identical to one in additional_routers.
    """

    class _FreshRouter(Capability):
        name = "fresh"

        def routers(self, role: str) -> list[APIRouter]:
            return [APIRouter()]  # distinct object every call

    extra = APIRouter()  # also empty — equal to the one above via __eq__
    reg = CapabilityRegistry([_FreshRouter()], role="all")
    # Must NOT raise; the two routers are equal but not the same object.
    create_app(capabilities=reg, additional_routers=[extra])
