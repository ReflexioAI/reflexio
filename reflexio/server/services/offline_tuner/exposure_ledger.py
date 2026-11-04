"""Composition for synchronous search exposure persistence."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Protocol, cast

from reflexio.server import search_runtime
from reflexio.server.cache import reflexio_cache
from reflexio.server.services.governance.config import (
    get_governance_ref_secret,
    governance_subject_ref,
)
from reflexio.server.services.search_exposure import (
    ExposureEventWriteResult,
    SearchExposureBatch,
    UserPlaybookExposureEvent,
    build_user_playbook_exposure_event,
    validate_exposure_batch_size,
)
from reflexio.server.tracing import profile_step


class ExposureLedgerStore(Protocol):
    def record_user_playbook_exposure_events(
        self,
        events: Sequence[UserPlaybookExposureEvent],
        *,
        reuse_existing_timestamps: bool = False,
    ) -> list[ExposureEventWriteResult]: ...


def _storage_for_org(org_id: str) -> ExposureLedgerStore:
    if search_runtime.current() is not None:
        storage = search_runtime.search_storage(org_id)
        if storage is None or str(getattr(storage, "org_id", "")) != org_id:
            raise RuntimeError("Exposure storage is not bound to this search scope")
        return cast(ExposureLedgerStore, storage)
    return cast(ExposureLedgerStore, reflexio_cache.get_reflexio(org_id).get_storage())


class ExposureLedgerRecorder:
    """Translate the final route batch into durable exposure envelopes."""

    def __init__(
        self,
        *,
        storage_for_org: Callable[[str], ExposureLedgerStore] = _storage_for_org,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._storage_for_org = storage_for_org
        self._clock = clock

    def record(self, batch: SearchExposureBatch) -> None:
        if not batch.user_playbooks:
            return
        with search_runtime.bounded_operation(3.0):
            self._record(batch)

    def _record(self, batch: SearchExposureBatch) -> None:
        validate_exposure_batch_size(batch.user_playbooks)
        recorded_at = int(self._clock())
        secret = get_governance_ref_secret()
        events = [
            build_user_playbook_exposure_event(
                batch,
                playbook,
                exposed_at=recorded_at,
                ingested_at=recorded_at,
                governance_subject_ref=(
                    governance_subject_ref(batch.org_id, batch.user_id, secret)
                    if batch.user_id
                    else None
                ),
                playbook_owner_governance_subject_ref=(
                    governance_subject_ref(batch.org_id, playbook.user_id, secret)
                    if playbook.user_id
                    else None
                ),
            )
            for playbook in batch.user_playbooks
        ]
        with profile_step("search.exposure.storage_lookup"):
            storage = self._storage_for_org(batch.org_id)
        with profile_step("search.exposure.write"):
            storage.record_user_playbook_exposure_events(
                events,
                reuse_existing_timestamps=True,
            )


__all__ = ["ExposureLedgerRecorder", "ExposureLedgerStore"]
