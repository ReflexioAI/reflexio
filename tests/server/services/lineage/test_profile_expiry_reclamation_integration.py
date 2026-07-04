"""Integration tests: TTL-expiry reclamation regression guard (Task 1.6).

Three tests guard the specific pathology where an active (status=NULL) profile
whose ``expiration_timestamp`` has elapsed was invisible to both the read filter
AND the tombstone GC — so it leaked indefinitely until the ``expire_active_profiles``
sweep was added.

Why integration (not e2e):
    The e2e test conftest bypasses the global LLM mock so historical e2e tests can
    hit real APIs.  Under ``tests/server/`` the global mock IS active (no paid key
    needed).  The LLM mock returns ``time_to_live: "one_month"`` for every profile
    extraction call, so ``calculate_expiration_timestamp`` runs in the real extractor
    code-path and stores a non-NEVER expiration_timestamp — the real computation path
    is exercised without any hand-crafted literal on the row.

Time control:
    ``expire_active_profiles`` and ``gc_expired_tombstones`` accept explicit ``now``
    / ``older_than_epoch`` parameters, so we pass ``profile.expiration_timestamp + N``
    to simulate being past expiry without touching the wall clock.  For the
    ``get_user_profile`` read-filter assertion (which uses ``_epoch_now()`` internally)
    we monkeypatch ``_epoch_now`` in the profiles module for the duration of the check.
"""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from reflexio.lib.reflexio_lib import Reflexio
from reflexio.models.api_schema.common import NEVER_EXPIRES_TIMESTAMP
from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.models.api_schema.service_schemas import InteractionData
from reflexio.models.config_schema import (
    Config,
    ProfileExtractorConfig,
    StorageConfigSQLite,
)
from reflexio.server.services.configurator.configurator import DefaultConfigurator

pytestmark = pytest.mark.integration

# Patch target for the expiry filter used inside get_user_profile
_EPOCH_NOW_PATH = (
    "reflexio.server.services.storage.sqlite_storage.profiles._profile_store._epoch_now"
)


@pytest.fixture()
def reflexio_instance(tmp_path: pathlib.Path, worker_id: str) -> Reflexio:
    """Create a Reflexio instance with real SQLite storage and a profile-only config.

    ``stride_size=1`` ensures the extractor runs after a single published interaction
    without requiring the default stride of 8.

    Args:
        tmp_path: Pytest-provided temporary directory, unique per test.
        worker_id: Pytest-xdist worker id for org isolation under parallel runs.

    Returns:
        Reflexio: A fully configured instance backed by an isolated SQLite DB.
    """
    org_id = f"expiry_regression_{worker_id}"
    config = Config(
        storage_config=StorageConfigSQLite(db_path=str(tmp_path / "expiry_test.db")),
        agent_context_prompt="test agent for expiry regression",
        stride_size=1,
        profile_extractor_config=ProfileExtractorConfig(
            extractor_name="expiry_test_extractor",
            context_prompt="extract user facts",
            extraction_definition_prompt="name, intent",
            tagging_definition_prompt="choice of ['fact']",
        ),
    )
    configurator = DefaultConfigurator(org_id=org_id, config=config)
    return Reflexio(org_id=org_id, configurator=configurator)


def _publish_and_get_profile(reflexio: Reflexio) -> object:
    """Publish one interaction through the real extraction path and return the profile.

    The mock LLM returns ``time_to_live: "one_month"`` for profile extraction,
    so ``calculate_expiration_timestamp`` runs in the extractor and stores a real
    (non-NEVER) ``expiration_timestamp`` on the persisted row.

    Args:
        reflexio: A configured Reflexio instance.

    Returns:
        UserProfile: The first profile created by ``publish_interaction``.

    Raises:
        AssertionError: If publication fails or no profile is created.
    """
    user_id = "u_expiry_test"
    response = reflexio.publish_interaction(
        {
            "user_id": user_id,
            "session_id": "sess_expiry_regression",
            "interaction_data_list": [
                InteractionData(content="User shared their preferences.", role="user"),
            ],
            "source": "expiry_regression_test",
        }
    )
    assert response.success, f"publish_interaction failed: {response.message}"
    storage = reflexio.request_context.storage
    profiles = storage.get_user_profile(user_id)
    assert len(profiles) == 1, (
        "publish_interaction must produce exactly one profile via the real "
        f"calculate_expiration_timestamp path; got {len(profiles)}"
    )
    return profiles[0]


# ---------------------------------------------------------------------------
# Test 1: faithful path — profile minted with real TTL is swept and GC'd away
# ---------------------------------------------------------------------------


def test_ttl_expired_active_profile_is_physically_reclaimed(
    reflexio_instance: Reflexio,
) -> None:
    """Faithful path: a profile created with a real TTL is swept to EXPIRED then hard-deleted.

    Guards the full reclamation pipeline:
      1. publish_interaction → mock LLM emits one_month TTL → calculate_expiration_timestamp
         stores a non-NEVER expiration_timestamp.
      2. expire_active_profiles(now=past_expiry) tombstones the row (status → EXPIRED).
      3. gc_expired_tombstones(older_than_epoch=past_expiry+1) hard-deletes the tombstone.
      4. get_profile_by_id(include_tombstones=True) → None (row is gone).
      5. Lineage events contain both ``status_change`` (reason ttl-expired) and ``hard_delete``.
    """
    storage = reflexio_instance.request_context.storage
    profile = _publish_and_get_profile(reflexio_instance)

    # The mock LLM returns one_month TTL → real expiration_timestamp, not the sentinel.
    assert profile.expiration_timestamp != NEVER_EXPIRES_TIMESTAMP, (
        "expiration_timestamp must be a real finite value produced by "
        "calculate_expiration_timestamp, not the NEVER_EXPIRES sentinel"
    )

    past_expiry = profile.expiration_timestamp + 1

    # Step 1: sweep tombstones the TTL-expired active profile.
    swept = storage.expire_active_profiles(now=past_expiry)
    assert swept == 1, "expire_active_profiles must tombstone exactly 1 profile"

    # Step 2: GC hard-deletes the tombstone (retired_at == past_expiry < older_than_epoch).
    deleted = storage.gc_expired_tombstones(
        entity_type="profile",
        older_than_epoch=past_expiry + 1,
    )
    assert deleted == 1, (
        "gc_expired_tombstones must physically delete exactly 1 tombstone"
    )

    # Profile row is physically gone.
    assert (
        storage.get_profile_by_id(profile.profile_id, include_tombstones=True) is None
    ), "profile must be gone after GC (include_tombstones=True must return None)"

    # Lineage log contains both the TTL-expiry status_change and the hard_delete.
    events = storage.get_lineage_events(entity_id=profile.profile_id)
    ops = {e.op for e in events}
    assert "status_change" in ops, (
        "Expected a status_change lineage event (reason=ttl-expired) from the sweep"
    )
    sc_events = [e for e in events if e.op == "status_change"]
    assert any(getattr(e, "reason", None) == "ttl-expired" for e in sc_events), (
        "Expected a status_change event with reason='ttl-expired' from expire_active_profiles"
    )
    assert "hard_delete" in ops, (
        "Expected a hard_delete lineage event from gc_expired_tombstones"
    )


# ---------------------------------------------------------------------------
# Test 2: pre-fix negative — before the sweep the bug is demonstrable
# ---------------------------------------------------------------------------


def test_pre_fix_pathology_active_expired_is_not_gc_eligible_until_swept(
    reflexio_instance: Reflexio,
) -> None:
    """Pre-fix negative: an active-but-expired profile is invisible to reads and GC until swept.

    This test pins the exact bug that ``expire_active_profiles`` was introduced to fix:
      - A profile past its ``expiration_timestamp`` has status=NULL (CURRENT).
      - Normal reads filter ``expiration_timestamp >= now`` → the row is invisible.
      - The tombstone GC filters ``status IN (MERGED, SUPERSEDED, ARCHIVED, EXPIRED)``
        → the row is NOT eligible for hard-deletion.
      - Only ``expire_active_profiles`` (the sweep) transitions it to status=EXPIRED,
        making it eligible for GC.

    We monkeypatch ``_epoch_now`` — used by ``get_user_profile``'s expiry filter —
    to simulate being past the profile's expiration, without touching the wall clock.
    """
    storage = reflexio_instance.request_context.storage
    profile = _publish_and_get_profile(reflexio_instance)

    assert profile.expiration_timestamp != NEVER_EXPIRES_TIMESTAMP

    past_expiry = profile.expiration_timestamp + 10

    # Simulate being past expiry: get_user_profile filters expiration_timestamp >= _epoch_now().
    with patch(_EPOCH_NOW_PATH, return_value=past_expiry):
        visible = storage.get_user_profile(profile.user_id)

    # The expired profile is invisible to normal reads (filtered by expiry timestamp).
    assert not any(p.profile_id == profile.profile_id for p in visible), (
        "An active-but-expired profile must be invisible to get_user_profile "
        "at simulated post-expiry time"
    )

    # But the row still exists in storage with status=None (CURRENT) — not yet a tombstone.
    raw = storage.get_profile_by_id(profile.profile_id, include_tombstones=True)
    assert raw is not None, (
        "Profile must still exist in storage — it has not been swept yet"
    )
    assert raw.status is None, (
        "Profile status must be None (CURRENT) before the expiry sweep; "
        "it is NOT yet a tombstone"
    )
    # retired_at is storage-internal (not on the domain model) — read via direct SQL.
    raw_retired_at = storage.conn.execute(
        "SELECT retired_at FROM profiles WHERE profile_id = ?",
        (profile.profile_id,),
    ).fetchone()["retired_at"]
    assert raw_retired_at is None, (
        "Profile retired_at must be None before the expiry sweep; "
        "the un-reclaimable state requires BOTH status None and retired_at None"
    )

    # GC without sweep skips it — status=None is not in the eligible set.
    gc_before_sweep = storage.gc_expired_tombstones(
        entity_type="profile",
        older_than_epoch=past_expiry + 1,
    )
    assert gc_before_sweep == 0, (
        "gc_expired_tombstones must return 0 for a status=None profile: "
        "only tombstones (MERGED/SUPERSEDED/ARCHIVED/EXPIRED) are eligible"
    )

    # The sweep is the fix: transitions the row from active to EXPIRED.
    swept = storage.expire_active_profiles(now=past_expiry)
    assert swept == 1, (
        "expire_active_profiles must tombstone the active-but-expired profile"
    )


# ---------------------------------------------------------------------------
# Test 3: race — supersede on a sweep-EXPIRED source is a clean no-op
# ---------------------------------------------------------------------------


def test_expiry_sweep_does_not_break_concurrent_supersede(
    reflexio_instance: Reflexio,
) -> None:
    """Race: a supersede attempt on a sweep-tombstoned profile is a clean no-op.

    After ``expire_active_profiles`` transitions a profile to status=EXPIRED, a
    concurrent ``supersede_profiles_by_ids`` on that id must silently skip it
    (returns [] — an empty committed set), produce no crash, and emit no phantom
    ``status_change`` lineage event for the SUPERSEDED transition.

    Guards spec RD-ADV-005: the sweep must not corrupt dedup lineage.
    """
    storage = reflexio_instance.request_context.storage
    profile = _publish_and_get_profile(reflexio_instance)

    assert profile.expiration_timestamp != NEVER_EXPIRES_TIMESTAMP

    past_expiry = profile.expiration_timestamp + 10

    # Sweep tombstones the profile to EXPIRED.
    swept = storage.expire_active_profiles(now=past_expiry)
    assert swept == 1

    # A supersede attempt on the now-EXPIRED source must be a clean no-op.
    actually_superseded = storage.supersede_profiles_by_ids(
        profile.user_id,
        [profile.profile_id],
        "race_test_request_id",
    )
    assert actually_superseded == [], (
        "supersede_profiles_by_ids must return [] when the target is already EXPIRED; "
        "EXPIRED is not in the eligible set {NULL, PENDING}"
    )

    # No phantom SUPERSEDED status_change must be emitted for the already-EXPIRED row.
    events = storage.get_lineage_events(entity_id=profile.profile_id)
    supersede_events = [
        e
        for e in events
        if e.op == "status_change" and e.to_status == Status.SUPERSEDED.value
    ]
    assert not supersede_events, (
        "No SUPERSEDED status_change event must be emitted after the sweep already "
        "tombstoned the profile to EXPIRED"
    )

    # -- Lineage-consistency check: the sweep must not break a concurrent survivor --
    # Create a second profile as the survivor (NEVER_EXPIRES; not touched by the sweep).
    survivor = UserProfile(
        profile_id="survivor_profile_race_test",
        user_id=profile.user_id,
        content="survivor profile for race-consistency check",
        last_modified_timestamp=profile.last_modified_timestamp + 1,
        generated_from_request_id="race_test_survivor_req",
        expiration_timestamp=NEVER_EXPIRES_TIMESTAMP,
    )
    storage.add_user_profile(profile.user_id, [survivor])

    # A supersede call "in favor of" the survivor (source = the EXPIRED profile) must
    # remain a no-op and must NOT corrupt the survivor.
    storage.supersede_profiles_by_ids(
        profile.user_id,
        [profile.profile_id],
        "race_test_request_id_survivor",
    )

    # The survivor must be resolvable after the no-op supersede on the EXPIRED source.
    assert (
        storage.get_profile_by_id(survivor.profile_id, include_tombstones=False)
        is not None
    ), (
        "Survivor profile must remain resolvable (status=NULL) after a no-op "
        "supersede_profiles_by_ids call on the already-EXPIRED source"
    )
