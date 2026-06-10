"""Unit tests for the shared storage session-id contract helper."""

import pytest

from reflexio.server.services.storage.error import (
    StorageError,
    require_non_empty_session_id,
)


def test_require_non_empty_session_id_returns_stripped_value():
    assert require_non_empty_session_id("  s-1  ") == "s-1"


@pytest.mark.parametrize("value", [None, "", "   ", 123, b"s-1"])
def test_require_non_empty_session_id_rejects_missing_or_blank(value):
    with pytest.raises(StorageError, match="run the latest data migrations"):
        require_non_empty_session_id(value)
