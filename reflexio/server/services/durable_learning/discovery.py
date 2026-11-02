"""Compose independent source facts without turning incomplete reads into idle."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from reflexio.server.services.storage.storage_base import BaseStorage


@dataclass(frozen=True)
class WorkDiscovery:
    """Fixed turn and housekeeping facts; iteration retains the org-provider API.

    Iteration names execution candidates only. Expiry is explicitly consumed by
    the scheduler's separate housekeeping path, never as a model/user admission.
    """

    org_ids: tuple[str, ...] = ()
    expiry_org_ids: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[str]:
        return iter(self.org_ids)


class IncompleteWorkDiscoveryError(RuntimeError):
    """Proven turn candidates survive even when another source is unavailable."""

    def __init__(self, org_ids: Iterable[str], *, expiry_org_ids: Iterable[str] = ()):
        self.org_ids = tuple(sorted(set(org_ids)))
        self.expiry_org_ids = tuple(sorted(set(expiry_org_ids)))
        super().__init__("Durable work discovery incomplete")


def discover_work(storage: BaseStorage) -> WorkDiscovery:
    """Preserve the existing learning roster and inspect this route's debt.

    Expiry is a separate housekeeping fact, not permission to dispatch user work.
    A partial result retains its cause for classification and never means idle.
    """
    orgs: set[str] = set()
    expiry: set[str] = set()
    failure: Exception | None = None
    try:
        orgs.update(storage.list_extraction_orgs())
    except Exception as exc:
        failure = exc
    try:
        presence = storage._deferred_work_presence()
        if presence.needs_turn:
            orgs.add(storage.org_id)
        if presence.expiry:
            expiry.add(storage.org_id)
    except Exception as exc:
        if failure is None:
            failure = exc
    if failure is not None:
        raise IncompleteWorkDiscoveryError(orgs, expiry_org_ids=expiry) from failure
    return WorkDiscovery(tuple(sorted(orgs)), tuple(sorted(expiry)))
