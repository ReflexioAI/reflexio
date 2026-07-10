"""Process-local scheduler for lineage tombstone garbage collection and expiry reclamation.

Startup runs when EITHER ``lineage_gc.enabled`` OR ``expiry_reclamation.enabled``
OR ``governance_retention.audit_events_retention_enabled`` is set in the bootstrap
org's config. Each tick evaluates every org independently.

- **Class A** (profile expiry sweep + tombstone GC): gated on ``lineage_gc.enabled``.
  Hard-deletes expired tombstones per that org's ``lineage_gc`` config.
- **Class B** (plain-row direct-delete sweeps, no PII/audit obligation): gated on
  ``expiry_reclamation.enabled``. Currently sweeps expired share links and expired
  pending tool calls.
- **Per-org sweeps**: registered via :func:`register_per_org_sweep`; invoked
  unconditionally once per org after the Class-B block. Enterprise registers a
  governance-retention closure here at startup (see ``reflexio_ext``).

One org's failure never stalls the loop; errors are captured as Sentry anomalies
and the loop continues to the next org.

Note: the poll interval is always taken from ``lineage_gc.poll_interval_seconds``
even when only ``expiry_reclamation`` or ``governance_retention`` is enabled;
all sweep classes share that cadence.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.env_utils import env_str
from reflexio.server.org_fanout import iterate_orgs_bounded
from reflexio.server.scheduling import LeaderGate, ThreadedScheduler
from reflexio.server.tracing import capture_anomaly

logger = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL_SECONDS = 86400
_MIN_POLL_SECONDS = 1
_ORG_SWEEP_TIMEOUT_SECONDS = 60.0
_DEFAULT_ORG_FANOUT_WORKERS = 8

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

# Injection seam: a deployment (e.g. enterprise/managed) sets a tenant-enumerating
# provider here at startup so the single OSS-started scheduler sweeps ALL tenant
# orgs. maybe_start_lineage_gc reads this when no explicit provider is passed.
# None (OSS default) keeps behavior byte-for-byte identical to before.
_org_id_provider_hook: Callable[[], list[str]] | None = None


def set_org_id_provider(provider: Callable[[], list[str]] | None) -> None:
    """Register the org-id provider consulted by ``maybe_start_lineage_gc``.

    Managed multi-tenant deployments call this at app-construction time so the
    lineage GC / expiry sweep reaches every tenant org (``storage.list_org_ids()``
    raises ``NotImplementedError`` on Supabase, degrading to bootstrap-only).
    Pass ``None`` to clear the hook (used by tests to restore OSS defaults).

    Args:
        provider (Callable[[], list[str]] | None): Callable returning the tenant
            org ids to sweep, or ``None`` to clear.
    """
    global _org_id_provider_hook  # noqa: PLW0603
    _org_id_provider_hook = provider


# Injection seam: enterprise sets the fleet leader gate here at composition
# time (spec 4.3), exactly like set_org_id_provider. None (OSS default) keeps
# the always-tick behavior.
_leader_gate_hook: LeaderGate | None = None


def set_leader_gate(gate: LeaderGate | None) -> None:
    """Register the leader gate consulted by ``maybe_start_lineage_gc``.

    Args:
        gate (LeaderGate | None): Fleet-coordination gate, or ``None`` to clear
            (tests restore OSS defaults).
    """
    global _leader_gate_hook  # noqa: PLW0603
    _leader_gate_hook = gate


# Global sweeps run once per tick (for GLOBAL tables like invitation_codes),
# gated on expiry_reclamation.enabled. Each fn takes `now` (unix epoch) and
# returns a deleted-row count. Enterprise registers its sweeps here at startup;
# empty (OSS default) keeps behavior byte-for-byte identical to before.
_global_sweep_hooks: list[Callable[[int], int]] = []


def register_global_sweep(fn: Callable[[int], int]) -> None:
    """Register a global (once-per-tick) reclamation sweep.

    Note: a registered sweep that absorbs its own exceptions (returns instead
    of raising) will NOT trigger the generic ``lineage.global_sweep.failed``
    backstop; such a sweep must emit its own failure anomaly.

    Args:
        fn (Callable[[int], int]): Called with the current unix epoch; returns
            the number of rows it deleted.
    """
    _global_sweep_hooks.append(fn)


def clear_global_sweeps() -> None:
    """Clear all registered global sweeps (tests restore OSS defaults)."""
    _global_sweep_hooks.clear()


# Per-org sweeps run once per org per tick (for per-org reclamation concerns).
# Each fn takes (org_id, now) and returns a deleted-row count. Enterprise
# registers its closure here at startup so governance retention folds into the
# shared scheduler loop; empty (OSS default) keeps behavior identical to before.
_per_org_sweep_hooks: list[Callable[[str, int], int]] = []


def register_per_org_sweep(fn: Callable[[str, int], int]) -> None:
    """Register a per-org reclamation sweep.

    The sweep is invoked once per org inside ``_gc_tick``'s org loop, after
    the Class-B block, unconditionally (not gated on ``lineage_gc`` or
    ``expiry_reclamation`` — the real gate lives in the enterprise closure).

    Note: a registered sweep that absorbs its own exceptions (returns instead
    of raising) will NOT trigger the generic ``lineage.per_org_sweep.failed``
    backstop; such a sweep must emit its own failure anomaly.

    Args:
        fn (Callable[[str, int], int]): Called with ``(org_id, now)`` where
            ``now`` is the current unix epoch; returns the number of rows deleted.
    """
    _per_org_sweep_hooks.append(fn)


def clear_per_org_sweeps() -> None:
    """Clear all registered per-org sweeps (tests restore OSS defaults)."""
    _per_org_sweep_hooks.clear()


class LineageGCScheduler(ThreadedScheduler):
    """Polling daemon that hard-deletes expired tombstones per org."""

    def __init__(
        self,
        *,
        request_context_factory: Callable[[str], RequestContext],
        bootstrap_org_id: str,
        org_id_provider: Callable[[], list[str]] | None = None,
        leader_gate: LeaderGate | None = None,
    ) -> None:
        super().__init__(
            thread_name="reflexio-lineage-gc-scheduler", leader_gate=leader_gate
        )
        self.request_context_factory = request_context_factory
        self.bootstrap_org_id = bootstrap_org_id
        # Optional injectable org-id source. When set (managed/multi-tenant mode),
        # it is the authoritative enumeration seam; enterprise supplies a
        # provider that lists tenant orgs via the control-plane repository — the
        # only seam that surfaces managed tenants where storage.list_org_ids()
        # raises NotImplementedError on Supabase. When None (OSS default),
        # discovery falls back to storage.list_org_ids() exactly as before.
        self.org_id_provider = org_id_provider
        # Orgs that timed out on the PREVIOUS tick; a repeat escalates.
        self._prior_timeout_orgs: set[str] = set()

    def _on_started(self) -> None:
        logger.info("event=lineage_gc_scheduler_started")

    def _on_stopped(self) -> None:
        logger.info("event=lineage_gc_scheduler_stopped")

    def _discover_org_ids(self, bootstrap_ctx: RequestContext) -> list[str]:
        """Return every known org, always including the bootstrap org.

        When an ``org_id_provider`` was injected (managed/multi-tenant mode), it
        is the authoritative org-id source and ``storage.list_org_ids()`` is not
        consulted. Otherwise the OSS single-tenant path is used unchanged
        (``storage.list_org_ids()`` with a visible NotImplementedError fallback).
        The bootstrap org is always unioned in either way.
        """
        if self.org_id_provider is not None:
            try:
                org_ids = list(self.org_id_provider())
            except Exception:
                capture_anomaly(
                    "lineage.gc.org_id_provider_failed",
                    bootstrap_org_id=self.bootstrap_org_id,
                )
                logger.exception(
                    "event=lineage_gc_org_id_provider_failed bootstrap_org_id=%s "
                    "— falling back to bootstrap org only",
                    self.bootstrap_org_id,
                )
                org_ids = []
        else:
            storage = getattr(bootstrap_ctx, "storage", None)
            org_ids = []
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

    def _sweep_org(self, org_id: str) -> None:
        """Run Class A, Class B, and per-org sweeps for a single org.

        Extracted from ``_gc_tick``'s per-org loop body so it can be handed to
        :func:`iterate_orgs_bounded` as the per-org work item; the stop-event
        check that used to guard the loop iteration now lives in that helper.

        Args:
            org_id (str): The org to sweep this tick.
        """
        try:
            ctx = self.request_context_factory(org_id)
            if ctx.storage is None:
                return
            cfg = ctx.configurator.get_config()
        except Exception:
            capture_anomaly("lineage.gc.run_failed", org_id=org_id)
            logger.exception("event=lineage_gc_org_failed org_id=%s", org_id)
            return

        # Class A: profile expiry sweep (requires PII/grace sign-off; gated on
        # lineage_gc.enabled independently of Class B).
        if cfg.lineage_gc.enabled:
            try:
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
            except Exception:
                capture_anomaly("lineage.expiry_sweep.failed", org_id=org_id)
                logger.exception("event=lineage_expiry_sweep_failed org_id=%s", org_id)

            # Tombstone GC: each entity type is independent within the loop.
            try:
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
            except Exception:
                capture_anomaly("lineage.gc.tombstone_gc_failed", org_id=org_id)
                logger.exception(
                    "event=lineage_gc_tombstone_gc_failed org_id=%s", org_id
                )

        # Class B: direct-delete of expired plain rows (no audit/grace
        # obligation; independent of lineage_gc).  Each sweep is isolated so
        # one failing method does not skip the rest.
        if (
            getattr(cfg, "expiry_reclamation", None) is not None
            and cfg.expiry_reclamation.enabled
        ):
            now = int(time.time())
            for method_name, grace, limit in _CLASS_B_SWEEPS:
                method = getattr(ctx.storage, method_name, None)
                if method is None:
                    continue
                try:
                    deleted = method(now=now, grace_seconds=grace, limit=limit)
                    if deleted:
                        logger.info(
                            "event=class_b_reclaim org_id=%s method=%s deleted=%d",
                            org_id,
                            method_name,
                            deleted,
                        )
                except Exception:
                    capture_anomaly(
                        "lineage.class_b_reclaim.failed",
                        org_id=org_id,
                        method=method_name,
                    )
                    logger.exception(
                        "event=class_b_reclaim_failed org_id=%s method=%s",
                        org_id,
                        method_name,
                    )

        # Per-org sweeps: invoked unconditionally once per org (the real gate
        # lives in each enterprise closure). Extracted to keep _gc_tick's
        # cyclomatic complexity in check and to mirror _run_global_sweeps.
        self._run_per_org_sweeps(org_id)

    def _gc_tick(self, org_ids: list[str], *, max_workers: int = 1) -> None:
        """Run one GC pass across the given org IDs.

        Factored out of ``_run_loop`` so tests can exercise it without threads.
        Default ``max_workers=1`` is the serial path (used by existing tests);
        ``_run_once`` passes the configured fan-out width.

        Args:
            org_ids (list[str]): Org IDs to process in this tick.
            max_workers (int): Bounded fan-out width (1 = serial).
        """
        timed_out = iterate_orgs_bounded(
            org_ids,
            self._sweep_org,
            max_workers=max_workers,
            per_org_timeout_seconds=_ORG_SWEEP_TIMEOUT_SECONDS,
            stop_event=self._stop_event,
        )
        for org_id in set(timed_out) & self._prior_timeout_orgs:
            capture_anomaly("lineage.gc.org_sweep_timeout_repeat", org_id=org_id)
        self._prior_timeout_orgs = set(timed_out)

    def _org_fanout_workers(self, bootstrap_ctx: RequestContext) -> int:
        """Resolve the fan-out width: SQLite pins to 1; else env (default 8).

        SQLite: each worker would construct a fresh storage (connection +
        migrate()) against one shared DB file — contention the serial loop
        never produces — and OSS-local is usually one org anyway (spec 6.1).

        Args:
            bootstrap_ctx (RequestContext): This tick's bootstrap context.

        Returns:
            int: The pool width for this tick.
        """
        storage = getattr(bootstrap_ctx, "storage", None)
        if storage is not None and "sqlite" in type(storage).__name__.lower():
            return 1
        raw = env_str(
            "REFLEXIO_SCHEDULER_ORG_WORKERS", str(_DEFAULT_ORG_FANOUT_WORKERS)
        )
        try:
            value = int(raw)
        except ValueError:
            return _DEFAULT_ORG_FANOUT_WORKERS
        return value if value > 0 else _DEFAULT_ORG_FANOUT_WORKERS

    def _run_per_org_sweeps(self, org_id: str) -> None:
        """Invoke each registered per-org sweep once for this org.

        Invoked unconditionally (not gated on ``lineage_gc``/``expiry_reclamation``
        — the real gate lives in each enterprise closure). Per-sweep failure
        isolation: one sweep raising does not skip the others; a sweep that
        absorbs its own exceptions bypasses the ``lineage.per_org_sweep.failed``
        backstop and must emit its own anomaly.

        Args:
            org_id: The org being swept this tick.
        """
        now = int(time.time())
        for sweep in _per_org_sweep_hooks:
            sweep_id = getattr(sweep, "__qualname__", repr(sweep))
            try:
                deleted = sweep(org_id, now)
                if deleted:
                    logger.info(
                        "event=per_org_sweep org_id=%s sweep=%s deleted=%d",
                        org_id,
                        sweep_id,
                        deleted,
                    )
            except Exception:
                capture_anomaly(
                    "lineage.per_org_sweep.failed",
                    org_id=org_id,
                    sweep=sweep_id,
                )
                logger.exception(
                    "event=per_org_sweep_failed org_id=%s sweep=%s",
                    org_id,
                    sweep_id,
                )

    def _run_global_sweeps(self, cfg: object) -> None:
        """Invoke each registered global sweep once, gated on expiry_reclamation.

        Per-sweep failure isolation: one sweep raising does not skip the others.
        The specific per-entity anomaly is emitted inside each registered sweep;
        this is a generic backstop.

        Args:
            cfg: The bootstrap-org config read for this tick.
        """
        expiry = getattr(cfg, "expiry_reclamation", None)
        if expiry is None or not expiry.enabled:
            return
        now = int(time.time())
        for sweep in _global_sweep_hooks:
            try:
                deleted = sweep(now)
                if deleted:
                    logger.info("event=global_sweep deleted=%d", deleted)
            except Exception:
                sweep_id = getattr(sweep, "__qualname__", repr(sweep))
                capture_anomaly("lineage.global_sweep.failed", sweep=sweep_id)
                logger.exception("event=global_sweep_failed sweep=%s", sweep_id)

    def _run_once(self) -> float:
        poll_interval = _DEFAULT_POLL_INTERVAL_SECONDS
        try:
            bootstrap_ctx = self.request_context_factory(self.bootstrap_org_id)
            cfg = bootstrap_ctx.configurator.get_config()
            poll_interval = cfg.lineage_gc.poll_interval_seconds
            org_ids = self._discover_org_ids(bootstrap_ctx)
            self._gc_tick(org_ids, max_workers=self._org_fanout_workers(bootstrap_ctx))
            self._run_global_sweeps(cfg)
        except Exception:
            logger.exception("event=lineage_gc_scheduler_tick_failed")
        return max(poll_interval, _MIN_POLL_SECONDS)


def maybe_start_lineage_gc(
    request_context_factory: Callable[[str], RequestContext],
    *,
    bootstrap_org_id: str,
    org_id_provider: Callable[[], list[str]] | None = None,
    leader_gate: LeaderGate | None = None,
) -> LineageGCScheduler | None:
    """Start the scheduler when bootstrap config enables tombstone GC or expiry reclamation.

    Startup runs when the bootstrap-org config sets ``lineage_gc.enabled``
    (Class A: profile expiry + tombstone GC), ``expiry_reclamation.enabled``
    (Class B: plain-row direct-delete sweeps), or
    ``governance_retention.audit_events_retention_enabled`` — OR when any sweep
    hook has been registered via :func:`register_per_org_sweep` /
    :func:`register_global_sweep`. Registered hooks carry their own per-org gate
    in the enterprise closure, so the scheduler must start to evaluate them even
    when none of the bootstrap-org flags are set. Returns ``None`` only if none
    of these conditions hold.

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
        org_id_provider: Optional tenant-enumerating org-id source. When ``None``
            (OSS default), falls back to the module-level provider hook set via
            :func:`set_org_id_provider`; when both are ``None`` discovery uses
            ``storage.list_org_ids()`` exactly as before.
        leader_gate: Optional fleet-coordination gate. When ``None`` (OSS
            default), falls back to the module-level hook set via
            :func:`set_leader_gate`; when both are ``None`` the scheduler ticks
            unconditionally (today's behavior).

    Returns:
        LineageGCScheduler: The started scheduler, or ``None`` if no start
        condition holds (see the startup criteria above).
    """
    try:
        ctx = request_context_factory(bootstrap_org_id)
        cfg = ctx.configurator.get_config()

        # Dead-knob warning: audit-event retention is an ENTERPRISE-only feature
        # (handled by an enterprise per-org reclamation sweep registered via
        # register_per_org_sweep). In an OSS-only deployment the knob is accepted
        # but does nothing — warn loudly. Detected via the configurator class:
        # enterprise swaps in EnterpriseConfigurator at construction, so the OSS
        # DefaultConfigurator means "no enterprise here".
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
        governance_retention_enabled = getattr(
            gr, "audit_events_retention_enabled", False
        )
        # Also start when any reclamation sweep has been registered — registered
        # hooks carry their own per-org gates (in the enterprise closure), so the
        # scheduler must run even if the bootstrap org's config has all flags off.
        # This preserves the "start unconditionally, gate per-org" invariant of
        # the deleted GovernanceRetentionScheduler.
        if not (
            cfg.lineage_gc.enabled
            or expiry_reclamation_enabled
            or governance_retention_enabled
            or bool(_per_org_sweep_hooks)
            or bool(_global_sweep_hooks)
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
        org_id_provider=(
            org_id_provider if org_id_provider is not None else _org_id_provider_hook
        ),
        leader_gate=leader_gate if leader_gate is not None else _leader_gate_hook,
    )
    scheduler.start()
    return scheduler
