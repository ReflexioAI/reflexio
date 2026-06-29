"""OSS extension primitives: the process-global capability/service registry.

OSS defines these seams; enterprise registers implementations at its composition
root. OSS never imports reflexio_ext. The service registry is process-global (a
module dict), mirroring set_configurator_class/get_configurator_class, so it is
reachable from every RequestContext construction site — HTTP, library
(lib/_base.py), and background workers — not just app-scoped ones.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from fastapi import APIRouter

from reflexio.server.services.unified_search_service import (
    RetrievalCaptureHook,
    configure_retrieval_capture_hook,
)
from reflexio.server.tracing import Tracer, configure_tracer
from reflexio.server.usage_metrics import (
    UsageEventRecorder,
    configure_usage_event_recorder,
)


class ServiceKey[T]:
    """Typed handle into the runtime-service registry.

    Args:
        name (str): Stable identifier, used in error messages and as the dict key.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"ServiceKey({self.name!r})"


_services: dict[str, object] = {}


def register_service[T](
    key: ServiceKey[T], value: T, *, override: bool = False
) -> None:
    """Register a runtime-service provider under ``key`` (composition root only).

    Args:
        key (ServiceKey[T]): The typed key.
        value (T): The provider/value.
        override (bool): Allow replacing an existing registration. Default False.

    Raises:
        RuntimeError: If ``key`` is already registered and ``override`` is False.
    """
    if not override and key.name in _services:
        raise RuntimeError(f"service already registered for key {key.name!r}")
    _services[key.name] = value


def get_service[T](key: ServiceKey[T]) -> T | None:
    """Return the provider for ``key``, or None if unset (optional providers)."""
    return _services.get(key.name)  # type: ignore[return-value]


def require_service[T](key: ServiceKey[T]) -> T:
    """Return the provider for ``key``, raising if unset (required providers).

    Args:
        key (ServiceKey[T]): The typed key whose provider to retrieve.

    Returns:
        T: The registered provider value.

    Raises:
        LookupError: If no provider is registered for ``key``.
    """
    if key.name not in _services:
        raise LookupError(f"required service not registered for key {key.name!r}")
    return _services[key.name]  # type: ignore[return-value]


def reset_services() -> None:
    """Clear all registered services. Test-only."""
    _services.clear()


# ---------------------------------------------------------------------------
# Hook registry, app context, and capability lifecycle primitives
# ---------------------------------------------------------------------------


class HookRegistry:
    """Facade over OSS process-global hook setters; capabilities install through it."""

    def set_tracer(self, tracer: Tracer | None) -> None:
        """Install (or clear) the process-global request tracer.

        Args:
            tracer (Tracer | None): Tracer implementation, or None to disable.
        """
        configure_tracer(tracer)

    def set_usage_recorder(self, recorder: UsageEventRecorder | None) -> None:
        """Install (or clear) the process-global usage-event recorder.

        Args:
            recorder (UsageEventRecorder | None): Recorder callable, or None to disable.
        """
        configure_usage_event_recorder(recorder)

    def set_retrieval_capture(self, hook: RetrievalCaptureHook | None) -> None:
        """Install (or clear) the process-global retrieval-capture hook.

        Args:
            hook (RetrievalCaptureHook | None): Hook callable, or None to disable.
        """
        configure_retrieval_capture_hook(hook)


@dataclass(frozen=True)
class AppContext:
    """Inter-step startup data passed to Capability.on_startup.

    Mirrors the values that thread through reflexio_ext lifespan locals
    (prepare_enterprise_startup output).

    Attributes:
        self_host_org_id (str | None): Org ID for self-hosted deployments, or None for platform.
        activated (bool): Whether the enterprise activation check passed.
    """

    self_host_org_id: str | None = None
    activated: bool = False


class Capability(ABC):
    """An enterprise feature plugged into the app at its composition root.

    All methods are no-op defaults so the registry can drive every capability
    through the same phases without hasattr checks. The provider owns this
    lifecycle shape (ABC); runtime-feature *contracts* are Protocols defined by OSS.
    """

    name: ClassVar[str]

    def install_services(self) -> None:  # noqa: B027
        """Register runtime-service providers (process-global). Default: none."""

    def install_hooks(self, hooks: HookRegistry) -> None:  # noqa: B027
        """Install pipeline hooks (tracer/usage-recorder/retrieval-capture). Default: none.

        Args:
            hooks (HookRegistry): Registry to install hooks through.
        """

    def routers(self, role: str) -> list[APIRouter]:  # noqa: ARG002
        """Return routers to mount for this deployment role. Default: none.

        Args:
            role (str): Deployment role (e.g. "all", "data-plane").

        Returns:
            list[APIRouter]: Routers to mount; empty list by default.
        """
        return []

    async def on_startup(self, ctx: AppContext) -> None:  # noqa: B027
        """Start daemons/schedulers. Raising aborts boot (fail-loud). Default: none.

        Args:
            ctx (AppContext): Startup context produced by prepare_enterprise_startup.
        """

    async def on_shutdown(self) -> None:  # noqa: B027
        """Stop what on_startup started. Default: none."""


class CapabilityRegistry:
    """Holds additive capabilities plus the two singular construction-time concerns.

    Args:
        capabilities (list[Capability]): Capability instances; names must be unique.
        configurator_class (type | None): Single configurator class, or None.
        billing_gate (Callable | None): Billing gate factory, or None.

    Raises:
        RuntimeError: If two capabilities share the same name.
    """

    def __init__(
        self,
        capabilities: list[Capability],
        *,
        configurator_class: type | None = None,
        billing_gate: Callable[[str], Callable[..., None]] | None = None,
    ) -> None:
        names = [c.name for c in capabilities]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise RuntimeError(f"duplicate capability names: {sorted(dupes)}")
        self.capabilities = capabilities
        self.configurator_class = configurator_class
        self.billing_gate = billing_gate

    def router_names(self) -> set[str]:
        """Names of capabilities that contribute at least one router (for dual-path de-dup).

        Returns:
            set[str]: Names of capabilities returning non-empty router lists.
        """
        return {
            c.name
            for c in self.capabilities
            if c.routers("all") or c.routers("data-plane")
        }
