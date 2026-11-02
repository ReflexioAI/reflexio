"""One bounded financial opportunity on an already-admitted org turn."""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass

from reflexio.server.background_work import (
    report_background_failure,
    report_background_success,
)
from reflexio.server.services.extraction.publication import _deliver, _failure_detail
from reflexio.server.services.storage.storage_base import BaseStorage
from reflexio.server.work_scope import WorkScope, bind_work_scope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Position:
    route: object
    generation: int = 0
    after: int | None = None
    through: int | None = None


class FinancialDeliverySweep:
    """Process-local progress, never ownership or proof of billing acceptance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: dict[str, _Position] = {}

    def deliver_one(self, storage: BaseStorage, org_id: str) -> bool:
        selection_scope = f"durable-financial-selection:{org_id}"
        try:
            if storage.org_id != org_id:
                raise ValueError("Financial storage org changed")
            route = storage._durable_work_route()
            with self._lock:
                position = self._positions.get(org_id)
                if position is None or position.route != route:
                    position = _Position(route)
                    self._positions[org_id] = position
            # No lock spans data I/O or the external financial recorder.
            through, item = storage._publication_delivery_page(
                after=position.after, through=position.through
            )
            with self._lock:
                if self._positions.get(org_id) is not position:
                    return False  # Another inspector advanced or replaced this route.
                self._positions[org_id] = _Position(
                    route,
                    position.generation + 1,
                    after=int(item["delivery_order"]) if item is not None else None,
                    through=through if item is not None else None,
                )
            report_background_success(selection_scope)
        except Exception as exc:
            report_background_failure(
                logger,
                "financial_selection_failed",
                exc,
                scope=selection_scope,
                detail=_failure_detail,
                org_id=org_id,
            )
            return False
        if item is None:
            return False
        # Advance on the raw descriptor BEFORE scope/financial parsing or
        # delivery. A corrupt/rejected head cannot starve later obligations.
        key = hashlib.sha256(str(item["receipt_id"]).encode()).hexdigest()[:12]
        delivery_scope = f"durable-financial-delivery:{org_id}:{key}"
        try:
            if item["org_id"] != org_id or not isinstance(item["project_id"], str):
                raise ValueError("Original financial scope unavailable")
            with bind_work_scope(WorkScope(org_id, item["project_id"] or None)):
                receipt = storage._get_publication_receipt(
                    str(item["receipt_id"]), project_id=item["project_id"]
                )
                if receipt is None:
                    # Missing BEFORE our strict acceptance is merely a race.
                    return False
                _deliver(storage, receipt)
            report_background_success(delivery_scope)
            return True
        except Exception as exc:
            report_background_failure(
                logger,
                "financial_delivery_failed",
                exc,
                scope=delivery_scope,
                detail=_failure_detail,
                org_id=org_id,
            )
            return False
