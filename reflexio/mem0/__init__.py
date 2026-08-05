"""Drop-in replacement for the ``mem0`` package that also learns via Reflexio.

Customers already using mem0's managed platform switch a single import::

    from mem0 import MemoryClient          # before
    from reflexio.mem0 import MemoryClient  # after

``MemoryClient`` forwards every call to real mem0, additionally publishes
``add()`` traces to Reflexio, and augments ``search()`` payloads with
Reflexio playbooks and profiles. ``Memory``, ``AsyncMemory``, and
``AsyncMemoryClient`` are re-exported unwrapped so existing imports keep
working.
"""

try:
    from mem0 import AsyncMemory, AsyncMemoryClient, Memory
except ImportError as exc:
    raise ImportError(
        "reflexio.mem0 requires the optional dependency 'mem0ai'. "
        "Install it with: pip install 'reflexio-ai[mem0]'"
    ) from exc

from reflexio.mem0._wrapper import MemoryClient

__all__ = ["AsyncMemory", "AsyncMemoryClient", "Memory", "MemoryClient"]
