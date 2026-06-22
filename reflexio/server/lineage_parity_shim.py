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
    classify_change_log_parity,
    profile_reconstructible_request_ids,
)
from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.server.site_var.feature_flags import is_lineage_dual_read_diff_enabled
from reflexio.server.tracing import capture_anomaly
from reflexio.server.usage_metrics import record_usage_event

_PARITY_READ_CAP = 10_000


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

        _emit_metrics(org_id=org_id, results=results, legacy_cmp=legacy_cmp)

    except Exception as e:  # noqa: BLE001
        capture_anomaly(
            "lineage.reconstruct.error",
            level="warning",
            org_id=org_id,
            error=type(e).__name__,
        )


def _emit_metrics(*, org_id: str, results: list, legacy_cmp: list) -> None:
    """Emit divergence anomalies and a coverage usage event.

    Args:
        org_id (str): Organization ID for tagging.
        results (list): Classified ``ParityResult`` list from ``classify_change_log_parity``.
        legacy_cmp (list): Legacy ``ProfileChangeLog`` rows used to compute add-only
            vs remove-bearing breakdown for matched runs.
    """
    divergent_classes = (ParityClass.RECON_MISSING, ParityClass.CONTENT_MISMATCH)

    # Emit one anomaly per divergent result.
    for r in results:
        if r.classification in divergent_classes:
            capture_anomaly(
                "lineage.reconstruct.divergence",
                level="warning",
                org_id=org_id,
                kind=r.classification.value,
                run_request_id=r.request_id,
            )

    # Build a lookup from request_id -> legacy row for coverage dimension.
    legacy_by_req = {row.request_id: row for row in legacy_cmp}

    matches = [r for r in results if r.classification == ParityClass.MATCH]
    divergences = [r for r in results if r.classification in divergent_classes]
    inconclusive = [r for r in results if r.classification == ParityClass.INCONCLUSIVE]

    add_only_runs = 0
    remove_bearing_runs = 0
    for r in matches:
        row = legacy_by_req.get(r.request_id)
        if row is not None and row.removed_profiles:
            remove_bearing_runs += 1
        else:
            add_only_runs += 1

    record_usage_event(
        org_id=org_id,
        event_name="lineage.reconstruct.coverage",
        event_category="lineage",
        count_value=len(matches),
        outcome="diverged" if divergences else "match",
        metadata={
            "add_only_runs": add_only_runs,
            "remove_bearing_runs": remove_bearing_runs,
            "matches": len(matches),
            "divergences": len(divergences),
            "inconclusive": len(inconclusive),
        },
    )
