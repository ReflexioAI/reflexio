"""Shared parity predicate for the lineage dual-read divergence shim.

Compares the legacy ``ProfileChangeLog`` table against
``reconstruct_profile_change_log`` output.  This module is the single source
of truth for classification logic — used by the one-shot parity script and,
later, by the online dual-read shim.

All public functions are pure (no I/O, no storage access) except
``profile_reconstructible_request_ids``, which reads from storage but makes
no mutations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from reflexio.models.api_schema.domain.entities import ProfileChangeLog, UserProfile

if TYPE_CHECKING:
    from reflexio.server.services.storage.storage_base import BaseStorage


class ParityClass(StrEnum):
    """Classification of a per-request_id parity comparison."""

    MATCH = "MATCH"
    RECON_MISSING = "RECON-MISSING"
    LEGACY_MISSING = "LEGACY-MISSING"
    CONTENT_MISMATCH = "CONTENT-MISMATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class ParityResult:
    """One row in the parity report.

    Attributes:
        request_id (str): The request_id being classified.
        classification (ParityClass): The outcome classification.
        detail (str): Optional human-readable description of why this
            classification was chosen.
    """

    request_id: str
    classification: ParityClass
    detail: str = field(default="")


def _profile_content_set(profiles: list[UserProfile]) -> set[tuple[str, str]]:
    """Extract a (profile_id, content) set from a list of UserProfile.

    Args:
        profiles (list[UserProfile]): Profiles to extract from.

    Returns:
        set[tuple[str, str]]: Set of (profile_id, content) tuples.
    """
    return {(p.profile_id, p.content) for p in profiles}


def _rows_match(legacy: ProfileChangeLog, recon: ProfileChangeLog) -> bool:
    """Return True when legacy and recon agree on added/removed by content.

    Deliberately ignores ``updated_profiles`` — the updated delta is tolerated
    as a non-material difference between the two paths.

    Args:
        legacy (ProfileChangeLog): Row from the legacy table.
        recon (ProfileChangeLog): Row from reconstruction.

    Returns:
        bool: True when added and removed sets match by (profile_id, content).
    """
    return _profile_content_set(legacy.added_profiles) == _profile_content_set(
        recon.added_profiles
    ) and _profile_content_set(legacy.removed_profiles) == _profile_content_set(
        recon.removed_profiles
    )


def classify_change_log_parity(
    legacy_rows: list[ProfileChangeLog],
    recon_rows: list[ProfileChangeLog],
    *,
    reconstructible_request_ids: set[str],
    read_cap_hit: bool,
) -> list[ParityResult]:
    """Compare legacy and reconstructed rows per request_id; return classified results.

    This is a pure function — it performs no I/O or storage access.

    Classification rules:
    - ``read_cap_hit=True``: return a single INCONCLUSIVE result immediately.
    - Any request_id appearing more than once on either side → INCONCLUSIVE
      (detail="duplicate request_id"); that id is excluded from all other
      classification. Never raises.
    - In both sides, rows match → MATCH.
    - In both sides, rows differ → CONTENT_MISMATCH.
    - Legacy-only, id in ``reconstructible_request_ids`` → RECON_MISSING
      (dangerous: signals exist but reconstruction dropped the run).
    - Legacy-only, id NOT in ``reconstructible_request_ids`` → LEGACY_MISSING
      (tolerated: no reconstructible signal; predates soft-delete or purged).
    - Recon-only → CONTENT_MISMATCH (reconstruction produced a run absent
      from legacy).

    Args:
        legacy_rows (list[ProfileChangeLog]): Rows from ``get_profile_change_logs``.
        recon_rows (list[ProfileChangeLog]): Rows from ``reconstruct_profile_change_log``.
        reconstructible_request_ids (set[str]): Request ids with reconstructible
            signals, computed independently of reconstruction via
            ``profile_reconstructible_request_ids``.
        read_cap_hit (bool): True when either side hit the read cap, making the
            comparison unreliable.

    Returns:
        list[ParityResult]: One entry per distinct request_id, classified.
    """
    if read_cap_hit:
        return [
            ParityResult(
                request_id="*",
                classification=ParityClass.INCONCLUSIVE,
                detail="read cap hit; comparison truncated",
            )
        ]

    # Build per-side dicts; track duplicates.
    legacy_by_req: dict[str, ProfileChangeLog] = {}
    legacy_dupes: set[str] = set()
    for row in legacy_rows:
        if row.request_id in legacy_by_req:
            legacy_dupes.add(row.request_id)
        legacy_by_req[row.request_id] = row

    recon_by_req: dict[str, ProfileChangeLog] = {}
    recon_dupes: set[str] = set()
    for row in recon_rows:
        if row.request_id in recon_by_req:
            recon_dupes.add(row.request_id)
        recon_by_req[row.request_id] = row

    all_dupes = legacy_dupes | recon_dupes

    # Emit one INCONCLUSIVE per duplicate id; skip duplicates in all other logic.
    results: list[ParityResult] = [
        ParityResult(
            request_id=req_id,
            classification=ParityClass.INCONCLUSIVE,
            detail="duplicate request_id",
        )
        for req_id in sorted(all_dupes)
    ]

    all_req_ids = (set(legacy_by_req) | set(recon_by_req)) - all_dupes
    for req_id in sorted(all_req_ids):
        in_legacy = req_id in legacy_by_req
        in_recon = req_id in recon_by_req

        if in_legacy and in_recon:
            if _rows_match(legacy_by_req[req_id], recon_by_req[req_id]):
                results.append(
                    ParityResult(request_id=req_id, classification=ParityClass.MATCH)
                )
            else:
                results.append(
                    ParityResult(
                        request_id=req_id,
                        classification=ParityClass.CONTENT_MISMATCH,
                        detail="added/removed sets differ",
                    )
                )
        elif in_legacy:
            if req_id in reconstructible_request_ids:
                results.append(
                    ParityResult(
                        request_id=req_id,
                        classification=ParityClass.RECON_MISSING,
                        detail="reconstructible signals exist but reconstruction dropped the run",
                    )
                )
            else:
                results.append(
                    ParityResult(
                        request_id=req_id,
                        classification=ParityClass.LEGACY_MISSING,
                        detail="no reconstructible signal; run predates soft-delete / purged — tolerated",
                    )
                )
        else:
            # recon-only: reconstruction produced a run that legacy lacks
            results.append(
                ParityResult(
                    request_id=req_id,
                    classification=ParityClass.CONTENT_MISMATCH,
                    detail="reconstruction produced a run absent from legacy",
                )
            )

    return results


def profile_reconstructible_request_ids(storage: BaseStorage) -> set[str]:
    """Compute the set of request_ids that have reconstructible signals.

    These are request_ids where reconstruction *could* produce a row:
    either a profile was stamped with ``generated_from_request_id``, or a
    ``status_change`` / ``superseded`` lineage event was recorded.  Used to
    distinguish RECON_MISSING (a real gap) from LEGACY_MISSING (tolerated).

    Args:
        storage (BaseStorage): Any ``BaseStorage`` instance that implements
            ``get_distinct_generated_from_request_ids`` and
            ``get_lineage_events``.

    Returns:
        set[str]: Request ids with at least one reconstructible signal.
    """
    ids: set[str] = set(storage.get_distinct_generated_from_request_ids())
    for evt in storage.get_lineage_events(
        entity_type="profile",
        org_id=storage.org_id,
    ):
        if (
            evt.op == "status_change"
            and evt.to_status == "superseded"
            and evt.request_id
        ):
            ids.add(evt.request_id)
    return ids
