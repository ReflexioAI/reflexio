"""OSS extension primitives: the process-global capability/service registry.

OSS defines these seams; enterprise registers implementations at its composition
root. OSS never imports reflexio_ext. The service registry is process-global (a
module dict), mirroring set_configurator_class/get_configurator_class, so it is
reachable from every RequestContext construction site — HTTP, library
(lib/_base.py), and background workers — not just app-scoped ones.
"""

from __future__ import annotations


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

    Raises:
        LookupError: If no provider is registered for ``key``.
    """
    if key.name not in _services:
        raise LookupError(f"required service not registered for key {key.name!r}")
    return _services[key.name]  # type: ignore[return-value]


def reset_services() -> None:
    """Clear all registered services. Test-only."""
    _services.clear()
