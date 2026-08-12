"""Drop-in replacement for the ``mem0`` package that also learns via Reflexio.

Customers already using mem0's managed platform switch a single import::

    from mem0 import MemoryClient          # before
    from reflexio.mem0 import MemoryClient  # after

The hosted sync and async clients mirror ``add()`` traces to Reflexio. Search
remains exactly mem0 unless ``include_reflexio=True`` is requested. Local
``Memory`` and ``AsyncMemory`` are re-exported unchanged.
"""

try:
    from mem0 import AsyncMemory, Memory
except ModuleNotFoundError as exc:
    if exc.name != "mem0":
        raise
    raise ImportError(
        "reflexio.mem0 requires the optional dependency 'mem0ai'. "
        "Install it with: pip install 'reflexio-ai[mem0]'"
    ) from exc

from reflexio.mem0._facade import (
    AsyncReflexioFacade,
    ReflexioFacade,
    ReflexioNotConfiguredError,
    ReflexioOperationError,
)
from reflexio.mem0._wrapper import (
    AsyncMemoryClient,
    MemoryClient,
    ReflexioNamespaceCollisionError,
)

__all__ = [
    "AsyncMemory",
    "AsyncMemoryClient",
    "AsyncReflexioFacade",
    "Memory",
    "MemoryClient",
    "ReflexioFacade",
    "ReflexioNamespaceCollisionError",
    "ReflexioNotConfiguredError",
    "ReflexioOperationError",
]
