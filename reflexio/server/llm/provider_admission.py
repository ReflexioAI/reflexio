"""Shared provider admission for extension-owned HTTP model transports.

The implementation and permits are shared with native model calls. Transports
hold the context until their underlying I/O and retiring workers have finished.
"""

from reflexio.server.llm._provider_concurrency import (
    ProviderCapSaturatedError,
    named_provider_slot,
)

__all__ = ["ProviderCapSaturatedError", "named_provider_slot"]
