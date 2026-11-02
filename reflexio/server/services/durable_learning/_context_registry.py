"""One ordered weak catalog for current identity and live directory ownership."""

from __future__ import annotations

import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reflexio.server.api_endpoints.request_context import RequestContext


@dataclass(frozen=True)
class _Entry:
    org_id: str
    directory: str | None
    context: weakref.ReferenceType[RequestContext]


class WeakContextRegistry:
    """Newest living registration wins; collecting it reveals the next live one.

    Registry bookkeeping never holds a context strongly or performs storage I/O.
    Consumers take short-lived strong snapshots outside the registry lock. One
    identity slot per object also bounds repeated registration of a live handle.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: OrderedDict[int, _Entry] = OrderedDict()

    def register(self, context: RequestContext) -> None:
        identity = id(context)

        def collected(reference: weakref.ReferenceType[RequestContext]) -> None:
            with self._lock:
                entry = self._entries.get(identity)
                if entry is not None and entry.context is reference:
                    del self._entries[identity]

        with self._lock:
            self._entries[identity] = _Entry(
                context.org_id,
                context.storage_base_dir,
                weakref.ref(context, collected),
            )
            self._entries.move_to_end(identity)

    def current(self, org_id: str, directory: str | None) -> RequestContext | None:
        with self._lock:
            for entry in reversed(list(self._entries.values())):
                if entry.org_id == org_id and entry.directory == directory:
                    context = entry.context()
                    if context is not None:
                        return context
        return None

    def snapshot(self, directory: str | None) -> list[RequestContext]:
        with self._lock:
            current: dict[str, RequestContext] = {}
            # Copy weak entries: dropping a temporary context can synchronously
            # run its GC callback and remove a registry entry on this thread.
            for entry in reversed(list(self._entries.values())):
                if entry.directory == directory and entry.org_id not in current:
                    context = entry.context()
                    if context is not None:
                        current[entry.org_id] = context
            return list(current.values())

    def directories(self) -> set[str | None]:
        with self._lock:
            return {
                entry.directory
                for entry in list(self._entries.values())
                if entry.context() is not None
            }

    def discard_directory(self, directory: str | None) -> None:
        with self._lock:
            # Copy integer keys, without allocating a tracked tuple per entry.
            # GC may remove an identity before this loop reaches it, including
            # while the snapshot allocation itself completes.
            for identity in list(self._entries):
                entry = self._entries.get(identity)
                if entry is not None and entry.directory == directory:
                    self._entries.pop(identity, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
