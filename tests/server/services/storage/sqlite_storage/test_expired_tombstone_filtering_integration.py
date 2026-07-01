"""Task 1.2: get_profile_by_id hides EXPIRED profiles by default."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from reflexio.models.api_schema.domain.entities import UserProfile
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage import SQLiteStorage

pytestmark = pytest.mark.integration


def _mk_profile(pid: str) -> UserProfile:
    return UserProfile(
        profile_id=pid,
        user_id="u1",
        content="c",
        last_modified_timestamp=int(datetime.now(UTC).timestamp()),
        generated_from_request_id="r1",
        status=None,
    )


def test_get_profile_by_id_hides_expired(tmp_path):
    with patch.object(SQLiteStorage, "_get_embedding", return_value=[0.0] * 512):
        storage = SQLiteStorage(org_id="org_test", db_path=str(tmp_path / "t.db"))
        storage.add_user_profile("u1", [_mk_profile("p1")])
        # A freshly-added profile is stored with a NULL status (live/CURRENT), so
        # the CURRENT->EXPIRED transition is driven from old_status=None. Passing
        # Status.CURRENT here would query `status = 'None'` (the enum's literal
        # value) and match zero NULL-status rows.
        storage.update_all_profiles_status(None, Status.EXPIRED, user_ids=["u1"])
        assert storage.get_profile_by_id("p1", include_tombstones=False) is None
        assert storage.get_profile_by_id("p1", include_tombstones=True) is not None
