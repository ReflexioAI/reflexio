"""Typed helpers for tests that need static narrowing, not runtime changes.

These helpers exist purely to satisfy the type checker at test call sites:
``require_storage`` narrows the optional ``RequestContext.storage`` attribute,
and ``as_mock`` views a spec'd mock held behind a real-class-typed attribute
as the ``MagicMock`` it actually is at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from reflexio.lib.reflexio_lib import Reflexio
    from reflexio.server.services.storage.storage_base import BaseStorage


def require_storage(instance: Reflexio) -> BaseStorage:
    """
    Return the instance's storage, asserting it is configured.

    ``RequestContext.storage`` is typed ``BaseStorage | None`` because storage
    creation can legitimately fail at runtime. Tests that operate on a fully
    configured instance use this helper to narrow away the ``None`` branch once
    instead of sprinkling ``assert`` statements at every access site.

    Args:
        instance (Reflexio): A Reflexio instance whose storage must be configured.

    Returns:
        BaseStorage: The configured storage backend.

    Raises:
        AssertionError: If storage is not configured on the instance.
    """
    storage = instance.request_context.storage
    if storage is None:
        raise AssertionError("test requires a configured storage backend")
    return storage


def as_mock(obj: object) -> MagicMock:
    """
    View a runtime mock behind a real-class-typed attribute as a MagicMock.

    Tests that build objects with ``Mock(spec=SomeClass)`` collaborators access
    mock-only attributes (``return_value``, ``side_effect``, ``assert_called_*``)
    through attributes the type checker sees as the real class. This cast makes
    the runtime reality visible to the checker without changing behavior.

    Args:
        obj (object): An object that is a Mock/MagicMock at runtime.

    Returns:
        MagicMock: The same object, typed as MagicMock.
    """
    return cast(MagicMock, obj)
