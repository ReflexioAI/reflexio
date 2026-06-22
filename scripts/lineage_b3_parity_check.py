#!/usr/bin/env python3
"""B3 pre-cutover parity check: reconstruct_profile_change_log vs legacy table.

Runs ``reconstruct_profile_change_log`` and ``get_profile_change_logs`` side-
by-side against the same storage, diffs them per ``request_id``, and classifies
each discrepancy:

  MATCH          — reconstruction and legacy agree on added/removed profile
                   content and profile_ids.  OK.
  RECON-MISSING  — legacy has a row but reconstruction produced nothing for
                   that request_id.  REAL GAP → exits non-zero, reports failure.
  LEGACY-MISSING — reconstruction produced a row but legacy has no matching
                   request_id (e.g. dedup events exist but old log was not
                   written).  Tolerated — best-effort drop.

The reconstruction uses the time-travel-stable model:
  - added(R)   = profiles with immutable generated_from_request_id == R
                 (includes tombstones).
  - removed(R) = entity_ids of status_change+superseded events with request_id == R
                 (the dedup soft-delete signature; NOT reflection revise events).

Prints a summary with counts per class.  Exits 0 when there are no RECON-MISSING
or CONTENT-MISMATCH gaps; exits 1 when at least one such gap is found.  Note:
recon-only runs (reconstruction produced a run absent from legacy) are classified
as CONTENT_MISMATCH and therefore count as a gap → exit 1.  The old offline
analysis tolerated recon-only as exit 0; this script does not.

Usage (SQLite):
    uv run python scripts/lineage_b3_parity_check.py --db-path path/to/db --org-id myorg

This is a one-time pre-cutover tool — no scheduler, no Sentry channel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.lib._lineage_parity import (
    ParityClass,
    ParityResult,
    classify_change_log_parity,
    profile_reconstructible_request_ids,
)
from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.server.services.storage.storage_base import BaseStorage

_READ_CAP = 10_000


def run_parity_check(storage: BaseStorage) -> list[ParityResult]:
    """Run both paths against storage and classify each request_id.

    Args:
        storage (BaseStorage): Any ``BaseStorage`` instance that implements both
            ``get_profile_change_logs`` and ``get_lineage_events``.

    Returns:
        list[ParityResult]: Classified results, one per distinct request_id.

    Raises:
        SystemExit: If either side returns exactly ``_READ_CAP`` rows, the
            comparison may be based on truncated data and cannot be trusted.
            The run is treated as INCONCLUSIVE and exits non-zero.
    """
    legacy_rows = storage.get_profile_change_logs(limit=_READ_CAP)
    recon_resp = reconstruct_profile_change_log(storage, limit=_READ_CAP)

    read_cap_hit = (
        len(legacy_rows) >= _READ_CAP
        or len(recon_resp.profile_change_logs) >= _READ_CAP
    )

    if read_cap_hit:
        truncated: list[str] = []
        if len(legacy_rows) >= _READ_CAP:
            truncated.append(f"legacy ({len(legacy_rows)} rows == cap)")
        if len(recon_resp.profile_change_logs) >= _READ_CAP:
            truncated.append(
                f"reconstruction ({len(recon_resp.profile_change_logs)} rows == cap)"
            )
        print(
            f"\nINCONCLUSIVE: data may be truncated at the {_READ_CAP}-row cap — "
            f"{', '.join(truncated)}. "
            "Re-run after raising the cap or filtering the dataset.",
            file=sys.stderr,
        )
        sys.exit(2)

    reconstructible = profile_reconstructible_request_ids(storage)
    return classify_change_log_parity(
        legacy_rows,
        recon_resp.profile_change_logs,
        reconstructible_request_ids=reconstructible,
        read_cap_hit=False,
    )


def print_summary(results: list[ParityResult]) -> None:
    """Print a human-readable summary of parity check results.

    Args:
        results (list[ParityResult]): Classified parity results.
    """
    counts: dict[ParityClass, int] = dict.fromkeys(ParityClass, 0)
    for r in results:
        counts[r.classification] += 1

    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"B3 Parity Check — {total} request_ids checked")
    print(f"{'=' * 60}")
    print(f"  MATCH             : {counts[ParityClass.MATCH]}")
    print(
        f"  RECON-MISSING     : {counts[ParityClass.RECON_MISSING]}  (REAL GAPs — legacy-only, reconstructible)"
    )
    print(
        f"  LEGACY-MISSING    : {counts[ParityClass.LEGACY_MISSING]}  (tolerated — no reconstructible signal)"
    )
    print(f"  CONTENT-MISMATCH  : {counts[ParityClass.CONTENT_MISMATCH]}  (divergence)")
    print(
        f"  INCONCLUSIVE      : {counts[ParityClass.INCONCLUSIVE]}  (duplicate ids or cap hit)"
    )
    print(f"{'=' * 60}")

    gaps = [
        r
        for r in results
        if r.classification in (ParityClass.RECON_MISSING, ParityClass.CONTENT_MISMATCH)
    ]
    if gaps:
        print("\nREAL GAPs (reconstruction missing/wrong or content divergence):")
        for g in gaps[:20]:  # cap output for large dbs
            print(
                f"  request_id={g.request_id!r}  class={g.classification}  detail={g.detail!r}"
            )
        if len(gaps) > 20:
            print(f"  ... and {len(gaps) - 20} more")
    else:
        print(
            "\nNo RECON-MISSING or CONTENT-MISMATCH gaps found — safe to drop legacy table."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B3 pre-cutover parity check (SQLite only)."
    )
    parser.add_argument("--db-path", type=Path, required=True)
    parser.add_argument("--org-id", type=str, default="default")
    args = parser.parse_args()

    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    storage = SQLiteStorage(org_id=args.org_id, db_path=str(args.db_path))
    results = run_parity_check(storage)
    print_summary(results)

    gaps = [
        r
        for r in results
        if r.classification in (ParityClass.RECON_MISSING, ParityClass.CONTENT_MISMATCH)
    ]
    sys.exit(1 if gaps else 0)


if __name__ == "__main__":
    main()
