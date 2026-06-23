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

Usage (Supabase / production data ref — read-only):
    LINEAGE_PARITY_SERVICE_KEY=<service_role key> \
    uv run python scripts/lineage_b3_parity_check.py \
        --supabase-url https://<ref>.supabase.co --org-id <org> [--schema org_<id>]

``--schema`` defaults to ``public`` (a dedicated per-org ref); pass ``org_<id>``
for an org living in the shared global-default cohort. The Supabase path is
strictly read-only (PostgREST GETs); it never constructs the writable
``SupabaseStorage``.

This is a one-time pre-cutover tool — no scheduler, no Sentry channel.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.lib._lineage_parity import (
    ParityClass,
    ParityReadStorage,
    ParityResult,
    run_parity_check,
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
        f"  INCONCLUSIVE      : {counts[ParityClass.INCONCLUSIVE]}  (duplicate ids, cap hit, or truncated reads)"
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


def _build_storage(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> ParityReadStorage:
    """Construct the storage to check from CLI args (SQLite or read-only Supabase)."""
    if args.supabase_url:
        key = os.environ.get(args.service_key_env)
        if not key:
            parser.error(
                f"--supabase-url requires the service_role key in ${args.service_key_env}"
            )
        from reflexio.lib._lineage_parity_readers import RestStorageReader

        # Read-only reader implementing the ParityReadStorage protocol.
        return RestStorageReader(
            args.supabase_url, key, org_id=args.org_id, schema=args.schema
        )

    from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

    return SQLiteStorage(org_id=args.org_id, db_path=str(args.db_path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B3 pre-cutover parity check (SQLite or read-only Supabase)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db-path", type=Path, help="SQLite DB path (local mode)")
    source.add_argument(
        "--supabase-url",
        type=str,
        help="Supabase project URL, e.g. https://<ref>.supabase.co (read-only)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="public",
        help="schema holding the org's data: public (dedicated ref) or org_<id> (shared cohort)",
    )
    parser.add_argument(
        "--service-key-env",
        type=str,
        default="LINEAGE_PARITY_SERVICE_KEY",
        help="env var holding the service_role key for --supabase-url",
    )
    parser.add_argument("--org-id", type=str, default="default")
    args = parser.parse_args()

    storage = _build_storage(args, parser)
    results = run_parity_check(storage)
    print_summary(results)

    # Exit 2 = INCONCLUSIVE (truncated/duplicate — verdict untrustworthy),
    # 1 = real gaps, 0 = clean.
    if any(r.classification is ParityClass.INCONCLUSIVE for r in results):
        sys.exit(2)
    gaps = [
        r
        for r in results
        if r.classification in (ParityClass.RECON_MISSING, ParityClass.CONTENT_MISMATCH)
    ]
    sys.exit(1 if gaps else 0)


if __name__ == "__main__":
    main()
