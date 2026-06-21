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
gaps; exits 1 when at least one RECON-MISSING gap is found.

Usage (SQLite):
    uv run python scripts/lineage_b3_parity_check.py --db-path path/to/db --org-id myorg

This is a one-time pre-cutover tool — no scheduler, no Sentry channel.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # scripts/
_PROJECT_ROOT = _THIS_DIR.parent  # repo root

sys.path.insert(0, str(_PROJECT_ROOT))

from reflexio.lib._profiles import reconstruct_profile_change_log
from reflexio.models.api_schema.domain.entities import ProfileChangeLog, UserProfile
from reflexio.server.services.storage.storage_base import BaseStorage


class ParityClass(StrEnum):
    """Classification of a per-request_id parity comparison."""

    MATCH = "MATCH"
    RECON_MISSING = "RECON-MISSING"
    LEGACY_MISSING = "LEGACY-MISSING"


@dataclass
class ParityResult:
    """One row in the parity report."""

    request_id: str
    classification: ParityClass
    detail: str = ""


def _profile_content_set(profiles: list[UserProfile]) -> set[tuple[str, str]]:
    """Extract a (profile_id, content) set from a list of UserProfile."""
    return {(p.profile_id, p.content) for p in profiles}


def _rows_match(legacy: ProfileChangeLog, recon: ProfileChangeLog) -> bool:
    """Return True when legacy and recon agree on added/removed by content."""
    return _profile_content_set(legacy.added_profiles) == _profile_content_set(
        recon.added_profiles
    ) and _profile_content_set(legacy.removed_profiles) == _profile_content_set(
        recon.removed_profiles
    )


def classify_parity(
    legacy_rows: list[ProfileChangeLog],
    recon_rows: list[ProfileChangeLog],
) -> list[ParityResult]:
    """Compare legacy and reconstructed rows per request_id; return classified results.

    Args:
        legacy_rows: Rows from ``get_profile_change_logs`` (legacy table).
        recon_rows: Rows from ``reconstruct_profile_change_log``.

    Returns:
        list[ParityResult]: One entry per distinct request_id, classified as
            MATCH, RECON-MISSING, or LEGACY-MISSING.
    """
    legacy_by_req: dict[str, ProfileChangeLog] = {}
    legacy_dupes: list[str] = []
    for row in legacy_rows:
        if row.request_id in legacy_by_req:
            legacy_dupes.append(row.request_id)
        legacy_by_req[row.request_id] = row

    recon_by_req: dict[str, ProfileChangeLog] = {}
    recon_dupes: list[str] = []
    for row in recon_rows:
        if row.request_id in recon_by_req:
            recon_dupes.append(row.request_id)
        recon_by_req[row.request_id] = row

    if legacy_dupes or recon_dupes:
        raise ValueError(
            f"Duplicate request_ids detected — parity comparison is ambiguous. "
            f"Legacy dupes: {legacy_dupes!r}. Recon dupes: {recon_dupes!r}."
        )

    all_req_ids = set(legacy_by_req) | set(recon_by_req)
    results: list[ParityResult] = []

    for req_id in sorted(all_req_ids):
        in_legacy = req_id in legacy_by_req
        in_recon = req_id in recon_by_req

        if in_legacy and in_recon:
            if _rows_match(legacy_by_req[req_id], recon_by_req[req_id]):
                results.append(
                    ParityResult(request_id=req_id, classification=ParityClass.MATCH)
                )
            else:
                # Content differs — treat as a real gap (reconstruction wrong).
                results.append(
                    ParityResult(
                        request_id=req_id,
                        classification=ParityClass.RECON_MISSING,
                        detail="content mismatch between legacy and reconstruction",
                    )
                )
        elif in_legacy and not in_recon:
            results.append(
                ParityResult(
                    request_id=req_id,
                    classification=ParityClass.RECON_MISSING,
                    detail="legacy row exists but no lineage event found",
                )
            )
        else:
            # in_recon and not in_legacy
            results.append(
                ParityResult(
                    request_id=req_id,
                    classification=ParityClass.LEGACY_MISSING,
                    detail="lineage event exists but legacy row absent",
                )
            )

    return results


_READ_CAP = 10_000


def run_parity_check(storage: BaseStorage) -> list[ParityResult]:
    """Run both paths against storage and classify each request_id.

    Args:
        storage: Any ``BaseStorage`` instance that implements both
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

    truncated: list[str] = []
    if len(legacy_rows) >= _READ_CAP:
        truncated.append(f"legacy ({len(legacy_rows)} rows == cap)")
    if len(recon_resp.profile_change_logs) >= _READ_CAP:
        truncated.append(
            f"reconstruction ({len(recon_resp.profile_change_logs)} rows == cap)"
        )

    if truncated:
        print(
            f"\nINCONCLUSIVE: data may be truncated at the {_READ_CAP}-row cap — "
            f"{', '.join(truncated)}. "
            "Re-run after raising the cap or filtering the dataset.",
            file=sys.stderr,
        )
        sys.exit(2)

    return classify_parity(legacy_rows, recon_resp.profile_change_logs)


def print_summary(results: list[ParityResult]) -> None:
    """Print a human-readable summary of parity check results.

    Args:
        results: Classified parity results.
    """
    counts: dict[ParityClass, int] = dict.fromkeys(ParityClass, 0)
    for r in results:
        counts[r.classification] += 1

    total = len(results)
    print(f"\n{'=' * 60}")
    print(f"B3 Parity Check — {total} request_ids checked")
    print(f"{'=' * 60}")
    print(f"  MATCH          : {counts[ParityClass.MATCH]}")
    print(f"  RECON-MISSING  : {counts[ParityClass.RECON_MISSING]}  (REAL GAPs)")
    print(f"  LEGACY-MISSING : {counts[ParityClass.LEGACY_MISSING]}  (tolerated)")
    print(f"{'=' * 60}")

    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    if gaps:
        print("\nREAL GAPs (reconstruction missing/wrong):")
        for g in gaps[:20]:  # cap output for large dbs
            print(f"  request_id={g.request_id!r}  detail={g.detail!r}")
        if len(gaps) > 20:
            print(f"  ... and {len(gaps) - 20} more")
    else:
        print("\nNo RECON-MISSING gaps found — safe to drop legacy table.")


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

    gaps = [r for r in results if r.classification == ParityClass.RECON_MISSING]
    sys.exit(1 if gaps else 0)


if __name__ == "__main__":
    main()
