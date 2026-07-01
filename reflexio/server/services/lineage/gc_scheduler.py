"""Process-local scheduler for lineage tombstone garbage collection and expiry reclamation.

Startup runs when EITHER ``lineage_gc.enabled`` OR ``expiry_reclamation.enabled``
is set in the bootstrap org's config. Each tick evaluates every org independently.

- **Class A** (profile expiry sweep + tombstone GC): gated on ``lineage_gc.enabled``.
  Hard-deletes expired tombstones per that org's ``lineage_gc`` config.
- **Class B** (plain-row direct-delete sweeps, no PII/audit obligation): gated on
  ``expiry_reclamation.enabled``. Currently sweeps expired share links and expired
  pending tool calls.

One org's failure never stalls the loop; errors are captured as Sentry anomalies
and the loop continues to the next org.

Note: the poll interval is always taken from ``lineage_gc.poll_interval_seconds``
even when only ``expiry_reclamation`` is enabled; Class B currently shares that cadence.

Governance-retention reclamation is a premium concern handled by the enterprise
GovernanceRetentionCapability (reflexio_ext), not here.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 86400
_MIN_POLL_SECONDS = 1

# Window-misconfiguration tripwire: if a single tick deletes more than this
# many tombstones for one org, something is likely wrong with the grace window.
_HIGH_VOLUME_THRESHOLD = 1000

_ENTITY_TYPES = ("user_playbook", "agent_playbook", "profile")

# Class B: direct-delete sweeps for expired plain rows (no PII/audit obligation).
# Each entry is (storage_method_name, grace_seconds, batch_limit).
# Methods added in Tasks 2.2 / 2.3; getattr-guarded so missing impls are skipped.
_CLASS_B_SWEEPS: tuple[tuple[str, int, int], ...] = (
    ("delete_expired_share_links", 7 * 86400, 1000),
    ("delete_expired_pending_tool_calls", 1 * 86400, 1000),
)


class LineageGCScheduler:
    """Polling daemon that hard-deletes expired tombstones per org."""

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
                if ctx.storage is None:
                    continue
                cfg = ctx.configurator.get_config()

                # Class A: profile expiry sweep + tombstone GC (requires PII/grace
                # sign-off; gated on lineage_gc.enabled independently of Class B).
                if cfg.lineage_gc.enabled:
                    expired_tombstoned = ctx.storage.expire_active_profiles(
                        now=int(time.time())
                    )
                    if expired_tombstoned:
                        logger.info(
                            "event=expiry_sweep org_id=%s profiles_tombstoned=%d",
                            org_id,
                            expired_tombstoned,
                        )
                    if expired_tombstoned > _HIGH_VOLUME_THRESHOLD:
                        capture_anomaly(
                            "lineage.expiry_sweep.high_volume",
                            org_id=org_id,
                            count=expired_tombstoned,
                        )
                    older_than_epoch = (
                        int(time.time())
                        - cfg.lineage_gc.tombstone_grace_window_days * 86400
                    )
                    tombstone_deleted = 0
                    for entity_type in _ENTITY_TYPES:
                        tombstone_deleted += ctx.storage.gc_expired_tombstones(
                            entity_type=entity_type,
                            older_than_epoch=older_than_epoch,
                        )
                    if tombstone_deleted:
                        logger.info(
                            "event=lineage_gc_tick org_id=%s tombstone_deleted=%d",
                            org_id,
                            tombstone_deleted,
                        )
                    if tombstone_deleted > _HIGH_VOLUME_THRESHOLD:
                        capture_anomaly(
                            "lineage.gc.high_volume",
                            org_id=org_id,
                            count=tombstone_deleted,
                        )

                # Class B: direct-delete of expired plain rows (no audit/grace
                # obligation; independent of lineage_gc).
                if (
                    getattr(cfg, "expiry_reclamation", None) is not None
                    and cfg.expiry_reclamation.enabled
                ):
                    now = int(time.time())
                    for method_name, grace, limit in _CLASS_B_SWEEPS:
                        method = getattr(ctx.storage, method_name, None)
                        if method is None:
                            continue
                        deleted = method(now=now, grace_seconds=grace, limit=limit)
                        if deleted:
                            logger.info(
                                "event=class_b_reclaim org_id=%s method=%s deleted=%d",
                                org_id,
                                method_name,
                                deleted,
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
    """Start the scheduler when bootstrap config enables tombstone GC or expiry reclamation.

    Startup runs when bootstrap-org config sets EITHER ``lineage_gc.enabled``
    (Class A: profile expiry + tombstone GC) OR ``expiry_reclamation.enabled``
    (Class B: plain-row direct-delete sweeps). Returns ``None`` if neither flag
    is set.

    Tombstone-GC enablement criteria (must ALL hold before enabling ``lineage_gc``
    for a production org):

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

    Note: the poll cadence is always controlled by ``lineage_gc.poll_interval_seconds``
    even when only ``expiry_reclamation`` is enabled; Class B shares that interval.

    Args:
        request_context_factory: Builds an org-scoped :class:`RequestContext`.
        bootstrap_org_id: Org used to read config and seed cross-org discovery.

    Returns:
        LineageGCScheduler: The started scheduler, or ``None`` if neither flag is set.
    """
    try:
        ctx = request_context_factory(bootstrap_org_id)
        cfg = ctx.configurator.get_config()

        # Dead-knob warning: audit-event retention is an ENTERPRISE-only feature
        # (handled by reflexio_ext GovernanceRetentionCapability). In an OSS-only
        # deployment the knob is accepted but does nothing — warn loudly. Detected
        # via the configurator class: enterprise swaps in EnterpriseConfigurator at
        # construction, so the OSS DefaultConfigurator means "no enterprise here".
        from reflexio.server.services.configurator.configurator import (  # noqa: PLC0415
            DefaultConfigurator,
            get_configurator_class,
        )

        gr = getattr(cfg, "governance_retention", None)
        if (
            gr is not None
            and getattr(gr, "audit_events_retention_enabled", False)
            and get_configurator_class() is DefaultConfigurator
        ):
            logger.warning(
                "event=governance_retention_knob_ignored "
                "audit_events_retention_enabled=True but this is an OSS-only "
                "deployment — audit-event retention is an enterprise-only feature "
                "and will NOT run here."
            )

        expiry_reclamation_enabled = getattr(
            getattr(cfg, "expiry_reclamation", None), "enabled", False
        )
        if not (cfg.lineage_gc.enabled or expiry_reclamation_enabled):
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
