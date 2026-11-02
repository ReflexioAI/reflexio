"""Existing read-only adapters remain constructible; new execution fails on use."""

from contextlib import contextmanager

import pytest

from reflexio.server.services.storage.storage_base._commit_scope import CommitScopeMixin

pytestmark = pytest.mark.unit


def test_legacy_transaction_adapter_constructs_without_new_ownership_property():
    class ExistingAdapter(CommitScopeMixin):
        @contextmanager
        def commit_scope(self):
            yield

    adapter = ExistingAdapter()
    with adapter.commit_scope():
        pass
    with pytest.raises(NotImplementedError):
        _ = adapter.in_commit_scope
