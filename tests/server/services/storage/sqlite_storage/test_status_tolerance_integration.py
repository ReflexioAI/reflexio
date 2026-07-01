import pytest
from reflexio.models.api_schema.domain.enums import Status
from reflexio.server.services.storage.sqlite_storage._base import _parse_status

pytestmark = pytest.mark.integration


def test_parse_status_known_and_null():
    assert _parse_status(None) is None
    assert _parse_status("") is None
    assert _parse_status("merged") == Status.MERGED


def test_parse_status_unknown_maps_to_tombstone_not_current():
    # An unknown status (e.g. a future 'expired' seen by an older build) must NOT
    # raise and must NOT be treated as CURRENT (None) — else it would leak as live.
    result = _parse_status("some_future_status")
    assert result is not None
    assert result in (Status.MERGED, Status.SUPERSEDED)
