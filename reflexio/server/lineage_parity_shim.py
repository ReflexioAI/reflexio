"""Read-time dual-read divergence shim for the profile-change-log route.

Compares the legacy ``get_profile_change_logs`` path against
``reconstruct_profile_change_log`` and emits metric/anomaly events for any
divergence found.  Runs for SIDE EFFECTS ONLY — the served response is never
altered.  The function MUST NEVER raise; all exceptions are swallowed and
reported as anomalies.
"""

from __future__ import annotations

from reflexio.lib._lineage_parity import (
    ParityClass,
    ParityResult,
    classify_change_log_parity,
    profile_reconstructible_request_ids,
)
from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import ProfileChangeLog
from reflexio.server.site_var.feature_flags import is_lineage_dual_read_diff_enabled
from reflexio.server.tracing import capture_anomaly
from reflexio.server.usage_metrics import record_usage_event

# The shim compares the whole-org change-log at a 10k read-cap (not the served
# limit=100), so the served-response window (100) and parity window (10k)
# intentionally differ.  The coverage metric measures parity coverage across the
# full org history, not just the currently-served slice.
_PARITY_READ_CAP = 10_000

# Max per-run divergence anomalies emitted before collapsing into an aggregate.
# Prevents Sentry quota exhaustion / high-cardinality explosion on first enablement.
_MAX_DIVERGENCE_ANOMALIES = 20


def dual_read_diff(reflexio: object, org_id: str) -> None:
    """Compare legacy and reconstructed profile-change-log paths; emit divergence metrics.

    Runs for SIDE EFFECTS ONLY.  The function never raises — all exceptions are
    caught and forwarded to ``capture_anomaly``.  The caller's served response is
    never modified.

    Args:
        reflexio: A ``Reflexio`` instance (typed as ``object`` to avoid a circular
            import; duck-typed to ``reflexio.request_context.storage``).
        org_id (str): Organization ID for flag-gating and metric tagging.

    Returns:
        None
    """
    try:
        if not is_lineage_dual_read_diff_enabled(org_id):
            return

        storage = reflexio.request_context.storage  # type: ignore[union-attr]

        legacy_cmp = storage.get_profile_change_logs(limit=_PARITY_READ_CAP)
        recon_cmp = reconstruct_profile_change_log(
            storage, limit=_PARITY_READ_CAP
        ).profile_change_logs
        read_cap_hit = (
            len(legacy_cmp) >= _PARITY_READ_CAP or len(recon_cmp) >= _PARITY_READ_CAP
        )
        reconstructible = profile_reconstructible_request_ids(storage)
        results = classify_change_log_parity(
            legacy_cmp,
            recon_cmp,
            reconstructible_request_ids=reconstructible,
            read_cap_hit=read_cap_hit,
        )

        _emit_metrics(
            org_id=org_id,
            results=results,
            legacy_cmp=legacy_cmp,
            recon_cmp=recon_cmp,
        )

    except Exception as e:  # noqa: BLE001 — intentionally broad; BaseException (signals) still propagates
        # error-level so a genuine shim failure reaches the Discord
        # (environment:production AND level:error) rule.  A real prod failure
        # (the 42501) hid as 17 escalating warnings for hours because this was
        # level="warning".  Fires once per failed page-view — acceptable for a
        # "the shim is broken" signal.  (Per-run *divergence* anomalies stay at
        # level="warning" on purpose — during the parity window divergences are
        # expected findings and would flood Discord.)
        capture_anomaly(
            "lineage.reconstruct.error",
            level="error",
            org_id=org_id,
            error=type(e).__name__,
        )


def _emit_metrics(
    *,
    org_id: str,
    results: list[ParityResult],
    legacy_cmp: list[ProfileChangeLog],
    recon_cmp: list[ProfileChangeLog],
) -> None:
    """Emit divergence anomalies and a coverage usage event.

    Args:
        org_id (str): Organization ID for tagging.
        results (list[ParityResult]): Classified results from ``classify_change_log_parity``.
        legacy_cmp (list[ProfileChangeLog]): Legacy rows used to compute add-only
            vs remove-bearing breakdown for matched runs, and to detect the
            false-clean (degenerate) case.
        recon_cmp (list[ProfileChangeLog]): Reconstructed rows, used to detect a
            degenerate reconstruction (empty recon while legacy is non-empty).
    """
    divergent_classes = (ParityClass.RECON_MISSING, ParityClass.CONTENT_MISMATCH)

    divergences = [r for r in results if r.classification in divergent_classes]

    # Emit per-run anomalies, capped at _MAX_DIVERGENCE_ANOMALIES to prevent
    # Sentry quota exhaustion on first enablement (could be ~10k divergences).
    for r in divergences[:_MAX_DIVERGENCE_ANOMALIES]:
        capture_anomaly(
            "lineage.reconstruct.divergence",
            level="warning",
            org_id=org_id,
            kind=r.classification.value,
            run_request_id=r.request_id,
        )
    if len(divergences) > _MAX_DIVERGENCE_ANOMALIES:
        capture_anomaly(
            "lineage.reconstruct.divergence_truncated",
            level="warning",
            org_id=org_id,
            total_divergences=len(divergences),
            emitted=_MAX_DIVERGENCE_ANOMALIES,
        )

    # Build a lookup from request_id -> legacy row for coverage dimension.
    legacy_by_req = {row.request_id: row for row in legacy_cmp}

    matches = [r for r in results if r.classification == ParityClass.MATCH]
    inconclusive = [r for r in results if r.classification == ParityClass.INCONCLUSIVE]

    add_only_runs = 0
    remove_bearing_runs = 0
    for r in matches:
        row = legacy_by_req.get(r.request_id)
        if row is not None and row.removed_profiles:
            remove_bearing_runs += 1
        else:
            add_only_runs += 1

    # False-clean guard: a run that could NOT actually reconstruct anything must
    # NOT be reported as "match"/clean.  The failure mode: reconstruction reads
    # succeed but yield an EMPTY signal set (e.g. get_lineage_events returns [] —
    # or, pre-B1-fix, partially fails) while the LEGACY change log still has rows
    # (especially remove-bearing ones).  Then classify_change_log_parity tags
    # every legacy row LEGACY_MISSING (tolerated) → zero divergences → a FALSE
    # "match" that would wrongly satisfy the parity gate on incomplete data.
    #
    # Be precise: a legitimately add-only org with genuinely no removals (no
    # remove-bearing legacy rows) must still be able to reach a true "match".  We
    # only flag the case where legacy HAS reconstructible-worthy rows (it is
    # non-empty AND carries removals) but reconstruction produced nothing usable
    # (no matches AND an empty reconstructed change log).
    #
    # KNOWN residual (intentionally NOT covered, to preserve the precision above):
    # the guard fires ONLY when legacy carries removals.  Any *add-only* degeneracy —
    # a physically-purged add-only run (profiles hard-deleted/GDPR-purged so
    # reconstruction is empty), OR an add-only org whose reads genuinely returned [] —
    # has no remove-bearing legacy row, so it is NOT flagged degraded and still
    # collapses to "match" (the tolerated LEGACY_MISSING class).  Accepted because:
    # widening the guard to add-only would mislabel every legitimately-purged org, and
    # post-B1 a broken (anon-keyed) ref fails LOUD at storage construction rather than
    # silently returning [], so the add-only-empty-read path is largely unreachable.
    legacy_has_any_remove_bearing = any(row.removed_profiles for row in legacy_cmp)
    degenerate = (
        bool(legacy_cmp)
        and legacy_has_any_remove_bearing
        and not matches
        and not recon_cmp
    )

    if degenerate:
        outcome = "degraded"
        # "We think it's clean but reconstruction saw nothing."  error-level so
        # an all-degenerate run is visible on the Discord production rule — this
        # is the alarm that the false-clean guard fired.
        capture_anomaly(
            "lineage.reconstruct.degraded",
            level="error",
            org_id=org_id,
            legacy_rows=len(legacy_cmp),
            recon_rows=len(recon_cmp),
        )
    elif divergences:
        outcome = "diverged"
    elif inconclusive:
        outcome = "inconclusive"
    else:
        outcome = "match"

    record_usage_event(
        org_id=org_id,
        event_name="lineage.reconstruct.coverage",
        event_category="lineage",
        count_value=len(matches),
        outcome=outcome,
        metadata={
            "add_only_runs": add_only_runs,
            "remove_bearing_runs": remove_bearing_runs,
            "matches": len(matches),
            "divergences": len(divergences),
            "inconclusive": len(inconclusive),
            "degraded": degenerate,
        },
    )
