"""SQLite-backed sink for the process-global ``record_usage_event`` hook.

Wires :data:`reflexio.server.usage_metrics._recorder` so each emitted
``UsageEvent`` becomes a row in the ``usage_events`` table. The wiring
is performed by :func:`reflexio.server.api.create_app` when the
``REFLEXIO_DISABLE_USAGE_EVENT_SINK`` env var is unset, so production
runs persist events but tests (which set the env var in
``tests/conftest.py``) see the no-op default.

The recorder is process-global, so the sink does not depend on the
storage at construction time — the storage resolver is called on the
first event of each org. This decouples the wiring from when the
SQLite file becomes available.

All recorder errors are caught and logged so the caller's hot path
(search / publish / extraction) is never affected by observability
failures. ``UsageEvent.metadata`` is serialised as JSON for
SQLite-friendly storage.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from typing import Any

from reflexio.server.usage_metrics import UsageEvent

logger = logging.getLogger(__name__)


StorageResolver = Callable[[str], Any]
"""Callable that returns the storage for an ``org_id``.

Returns ``None`` (or any non-attribute-having value) when no storage
is configured for the org; the sink silently drops the event in that
case.
"""


class SqliteUsageEventSink:
    """Append-only sink that writes :class:`UsageEvent` rows to SQLite.

    The sink holds a ``StorageResolver`` (a callable) rather than a
    direct storage reference, so the storage is resolved lazily on the
    first event for each org. This keeps the wiring simple (no need to
    await storage initialisation) and matches how
    :func:`reflexio.server.cache.reflexio_cache.get_reflexio` works
    elsewhere in the codebase.

    Args:
        get_storage: Callable that resolves ``org_id`` to a storage
            object exposing ``record_usage_event(**kwargs) -> None``.
            Typically a partial application of ``get_reflexio(...).request_context.storage``.
    """

    def __init__(self, get_storage: StorageResolver) -> None:
        self._get_storage = get_storage

    def __call__(self, event: UsageEvent) -> None:
        """Forward a :class:`UsageEvent` to the storage as one row.

        Failures (e.g., storage locked, schema drift) are caught and
        logged at WARNING level. The caller's hot path MUST NOT be
        affected by observability failures — this is the contract.

        Args:
            event: The :class:`UsageEvent` to persist.
        """
        try:
            storage = self._get_storage(event.org_id)
        except Exception as exc:  # noqa: BLE001 — sink must never raise.
            logger.debug(
                "SqliteUsageEventSink: storage resolver failed for org %s: %s",
                event.org_id,
                exc,
            )
            return
        if storage is None:
            # No storage configured for this org; silently drop the event.
            return
        try:
            storage.record_usage_event(
                org_id=event.org_id,
                event_name=event.event_name,
                event_category=event.event_category,
                user_id=event.user_id,
                request_id=event.request_id,
                session_id=event.session_id,
                pipeline=event.pipeline,
                entity_type=event.entity_type,
                entity_id=event.entity_id,
                caller_type=event.caller_type,
                count_value=event.count_value,
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                billing_input_tokens=event.billing_input_tokens,
                platform_llm=event.platform_llm,
                platform_storage=event.platform_storage,
                duration_ms=event.duration_ms,
                error_kind=event.error_kind,
                metadata=_metadata_to_jsonable(event.metadata),
            )
        except Exception as exc:  # noqa: BLE001 — sink must never raise.
            logger.warning(
                "SqliteUsageEventSink failed to persist event %s/%s: %s",
                event.event_category,
                event.event_name,
                exc,
            )


def _metadata_to_jsonable(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Best-effort conversion of ``metadata`` to a JSON-serialisable dict.

    The :class:`UsageEvent.metadata` field is typed ``Mapping[str, Any]``
    but the storage layer serialises it via :func:`json.dumps`. If a
    non-serialisable value sneaks in (e.g., a custom object), this
    helper coerces it to a string so the row write never fails.
    """
    if not metadata:
        return {}
    try:
        # Fast path: round-tripping through json catches non-serialisable
        # values and avoids surprises downstream.
        return json.loads(json.dumps(dict(metadata), default=str))
    except (TypeError, ValueError):
        return {k: str(v) for k, v in metadata.items()}


__all__ = ["SqliteUsageEventSink"]
