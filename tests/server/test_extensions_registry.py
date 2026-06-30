import asyncio

import pytest
from fastapi import APIRouter

from reflexio.server.extensions import (
    AppContext,
    Capability,
    CapabilityRegistry,
    HookRegistry,
    ServiceKey,
    get_service,
    register_service,
    require_service,
    reset_services,
)

KEY: ServiceKey[str] = ServiceKey("demo")


def setup_function() -> None:
    reset_services()


def test_get_returns_none_when_unset() -> None:
    assert get_service(KEY) is None


def test_register_then_get_and_require() -> None:
    register_service(KEY, "x")
    assert get_service(KEY) == "x"
    assert require_service(KEY) == "x"


def test_require_raises_when_unset() -> None:
    with pytest.raises(LookupError):
        require_service(KEY)


def test_double_register_is_error_unless_override() -> None:
    register_service(KEY, "x")
    with pytest.raises(RuntimeError):
        register_service(KEY, "y")
    register_service(KEY, "y", override=True)
    assert get_service(KEY) == "y"


def test_key_isolation() -> None:
    """Distinct ServiceKey instances are independent — registering under "a" does not affect "b"."""
    key_a: ServiceKey[str] = ServiceKey("a")
    key_b: ServiceKey[str] = ServiceKey("b")
    register_service(key_a, "val_a")
    assert get_service(key_b) is None


# ---------------------------------------------------------------------------
# Task-2 capability / hook / context tests
# ---------------------------------------------------------------------------


class _Cap(Capability):
    name = "demo_cap"

    def routers(self, role: str) -> list[APIRouter]:
        return [APIRouter()]


def test_capability_defaults_are_noops() -> None:
    cap = _Cap()
    cap.install_services()  # no raise
    cap.install_hooks(HookRegistry())  # no raise
    assert isinstance(cap.routers("all"), list)
    asyncio.run(cap.on_startup(AppContext()))  # no raise
    asyncio.run(cap.on_shutdown())  # no raise


def test_registry_dedup_on_duplicate_name() -> None:
    with pytest.raises(RuntimeError):
        CapabilityRegistry([_Cap(), _Cap()])


def test_registry_carries_singular_concerns() -> None:
    reg = CapabilityRegistry(
        [_Cap()],
        configurator_class=object,
        billing_gate=lambda _: lambda: None,
    )
    assert reg.configurator_class is object
    assert reg.billing_gate is not None


def test_appcontext_defaults() -> None:
    ctx = AppContext()
    assert ctx.self_host_org_id is None and ctx.activated is False
