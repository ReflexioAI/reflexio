"""Process-local scheduler for lineage cleanup.

Startup is bootstrap-config-gated. Each tick then evaluates every org
independently, running tombstone GC from ``lineage_gc`` and governance
retention GC from ``governance_retention`` according to that org's config.
One org's failure never stalls the loop; errors are captured as Sentry
anomalies and the scheduler continues to the next org.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from reflexio.models.config_schema import GovernanceRetentionConfig
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 86400
_MIN_POLL_SECONDS = 1

# Window-misconfiguration tripwire: if a single tick deletes more than this
# many tombstones for one org, something is likely wrong with the grace window.
# Governance retention uses separate policy knobs and must not trigger this.
_HIGH_VOLUME_THRESHOLD = 1000

_ENTITY_TYPES = ("user_playbook", "agent_playbook", "profile")


def _is_governance_retention_enabled(
    governance_retention: GovernanceRetentionConfig,
) -> bool:
    return governance_retention.audit_events_retention_enabled


class LineageGCScheduler:
    """Polling daemon that runs tombstone GC and governance retention per org."""

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        bootstrap_org_id: str,
    ) -> None:
        self.request_context_factory = request_context_factory
        self.bootstrap_org_id = bootstrap_org_id
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the daemon thread (idempotent if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="reflexio-lineage-gc-scheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("event=lineage_gc_scheduler_started")

    def stop(self, *, timeout_seconds: float = 5.0) -> None:
        """Signal the daemon to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_seconds)
        self._thread = None
        logger.info("event=lineage_gc_scheduler_stopped")

    def _discover_org_ids(self, bootstrap_ctx: RequestContext) -> list[str]:
        """Return every known org, always including the bootstrap org."""
        storage = getattr(bootstrap_ctx, "storage", None)
        org_ids: list[str] = []
        if storage is not None:
            try:
                org_ids = storage.list_org_ids()
            except NotImplementedError:
                logger.warning(
                    "event=lineage_gc_list_org_ids_not_implemented "
                    "backend=%s bootstrap_org_id=%s — GC will only process bootstrap org",
                    type(storage).__name__,
                    bootstrap_ctx.org_id,
                )
                org_ids = []
        if bootstrap_ctx.org_id not in org_ids:
            org_ids = [bootstrap_ctx.org_id, *org_ids]
        return org_ids

    def _gc_tick(self, org_ids: list[str]) -> None:
        """Run one GC pass across the given org IDs.

        Factored out of ``_run_loop`` so tests can exercise it without threads.

        Args:
            org_ids (list[str]): Org IDs to process in this tick.
        """
        for org_id in org_ids:
            if self._stop_event.is_set():
                break
            try:
                ctx = self.request_context_factory(org_id)
                cfg = ctx.configurator.get_config()
                if ctx.storage is None:
                    continue
                run_tombstone_gc = cfg.lineage_gc.enabled
                governance_retention = getattr(
                    cfg, "governance_retention", GovernanceRetentionConfig()
                )
                run_governance_gc = _is_governance_retention_enabled(
                    governance_retention
                )
                if not run_tombstone_gc and not run_governance_gc:
                    continue

                tombstone_deleted = 0
                if run_tombstone_gc:
                    older_than_epoch = (
                        int(time.time())
                        - cfg.lineage_gc.tombstone_grace_window_days * 86400
                    )
                    for entity_type in _ENTITY_TYPES:
                        count = ctx.storage.gc_expired_tombstones(
                            entity_type=entity_type,
                            older_than_epoch=older_than_epoch,
                        )
                        tombstone_deleted += count
                governance_deleted = 0
                if run_governance_gc:
                    governance_deleted = ctx.storage.gc_governance_retention(
                        config=governance_retention
                    )
                total_deleted = tombstone_deleted + governance_deleted
                if total_deleted:
                    logger.info(
                        (
                            "event=lineage_gc_tick org_id=%s deleted=%d "
                            "tombstone_deleted=%d governance_deleted=%d"
                        ),
                        org_id,
                        total_deleted,
                        tombstone_deleted,
                        governance_deleted,
                    )
                if tombstone_deleted > _HIGH_VOLUME_THRESHOLD:
                    capture_anomaly(
                        "lineage.gc.high_volume",
                        org_id=org_id,
                        count=tombstone_deleted,
                    )
            except Exception:
                capture_anomaly("lineage.gc.run_failed", org_id=org_id)
                logger.exception("event=lineage_gc_org_failed org_id=%s", org_id)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            poll_interval = _DEFAULT_POLL_INTERVAL_SECONDS
            try:
                bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
                cfg = bootstrap_ctx.configurator.get_config()
                poll_interval = cfg.lineage_gc.poll_interval_seconds
                org_ids = self._discover_org_ids(bootstrap_ctx)
                self._gc_tick(org_ids)
            except Exception:
                logger.exception("event=lineage_gc_scheduler_tick_failed")
            self._stop_event.wait(max(poll_interval, _MIN_POLL_SECONDS))


def maybe_start_lineage_gc(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
) -> LineageGCScheduler | None:
    """Start the scheduler only when bootstrap config enables some GC work.

    Off by default. Startup requires bootstrap-org config to enable tombstone GC
    or any governance retention gate. Tombstone-GC enablement criteria (must
    ALL hold before enabling for a production org):

    1. **Mechanism**: GC ages tombstones by ``retired_at`` (the INTEGER epoch
       written at every tombstone write-path).  Rows with ``retired_at = NULL``
       (created before the column was added) are never eligible and are retained.
    2. **Grace window**: 90 days is the vetted default (``tombstone_grace_window_days``).
       This is a per-deployment policy knob — do not shorten without reviewing
       PII-lifetime obligations (GDPR Art. 5(1)(e)) and audit-depth requirements.
    3. **B2↔B3 timing gate**: enable per-org only once the grace window is ≥ the
       reconstruction read-back horizon used by B3 changelog replay, OR once B3 is
       fully shipped and the horizon is confirmed.  Enabling before this point risks
       GC'ing tombstones the B3 replay still needs.
    4. **DPO sign-off**: obtain sign-off on the PII-lifetime and audit-depth
       implications before enabling in any deployment that processes personal data.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
        bootstrap_org_id: Org used to read config and seed cross-org discovery.

    Returns:
        LineageGCScheduler: The started scheduler, or ``None`` if not enabled.
    """
    try:
        ctx = request_context_factory(bootstrap_org_id)
        cfg = ctx.configurator.get_config()
        governance_retention = getattr(
            cfg, "governance_retention", GovernanceRetentionConfig()
        )
        if not cfg.lineage_gc.enabled and not _is_governance_retention_enabled(
            governance_retention
        ):
            return None
    except Exception as exc:
        logger.warning(
            "event=lineage_gc_scheduler_start_skipped error_type=%s error=%s",
            type(exc).__name__,
            exc,
        )
        return None

    scheduler = LineageGCScheduler(
        request_context_factory=request_context_factory,
        bootstrap_org_id=bootstrap_org_id,
    )
    scheduler.start()
    return scheduler
