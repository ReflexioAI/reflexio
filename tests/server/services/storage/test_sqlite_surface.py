"""Surface guard: SQLiteStorage public methods vs BaseStorage ABC.

Asserts that SQLiteStorage exposes the full BaseStorage public interface plus
only the explicitly-allowed RetentionMixin extras.  This is the OSS-side
drop-a-method tripwire for the playbook decomposition.
"""

from __future__ import annotations

from reflexio.server.services.storage.sqlite_storage import SQLiteStorage
from reflexio.server.services.storage.storage_base import BaseStorage

_OPTIONAL_OPTIMIZER_METHODS = frozenset(
    {
        "advance_playbook_optimization_stage",
        "claim_playbook_optimization_job",
        "commit_user_playbook_publication",
        "create_or_get_playbook_optimization_job",
        "get_playbook_optimization_artifact",
        "get_playbook_optimization_job",
        "get_unconsumed_gepa_user_playbook_publishing_job",
        "get_user_playbook_publication_subject_epochs",
        "load_user_playbook_publication_result",
        "prepare_gepa_user_playbook_publication",
        "reclaim_gepa_user_playbook_publishing_job",
        "reclaim_playbook_optimization_job",
        "renew_playbook_optimization_job_lease",
        "stage_user_playbook_publication",
        "upsert_playbook_optimization_artifact",
        "claim_user_playbook_publication",
    }
)

# RetentionMixin helpers are present on SQLiteStorage but not declared in the
# BaseStorage ABC — intentional; subclasses opt into retention without the ABC
# requiring it.
_RETENTION_MIXIN_METHODS: frozenset[str] = frozenset(
    {
        "count_retention_target_rows",
        "delete_oldest_retention_target_rows",
        "gc_retired_optimization_jobs",
    }
)

_HELPER_METHODS: frozenset[str] = frozenset({"close", "handle_exceptions"})


def _public_methods(cls: type) -> frozenset[str]:
    return frozenset(
        name
        for name in dir(cls)
        if not name.startswith("_")
        and name not in _HELPER_METHODS
        and callable(getattr(cls, name))
    )


def test_sqlite_surface_matches_base_abc() -> None:
    """SQLiteStorage public surface == BaseStorage ABC + RetentionMixin allowlist.

    Fails if any method is dropped from or added to SQLiteStorage without a
    matching update to BaseStorage.  The allowlist captures the retention
    helpers that are intentionally on SQLiteStorage but not on the ABC.
    """
    sqlite_methods = _public_methods(SQLiteStorage)
    base_methods = _public_methods(BaseStorage)

    # Membership guard: fail if a retention-allowlisted method is dropped from
    # SQLiteStorage (the subtraction check below stays green in that case).
    assert _RETENTION_MIXIN_METHODS.issubset(sqlite_methods), (
        f"Retention allowlist contains methods absent from SQLiteStorage: "
        f"{_RETENTION_MIXIN_METHODS - sqlite_methods}"
    )

    assert sqlite_methods - _RETENTION_MIXIN_METHODS == base_methods, (
        f"SQLiteStorage method delta vs BaseStorage ABC "
        f"(retention allowlist={_RETENTION_MIXIN_METHODS}):\n"
        f"  extra sqlite (not in allowlist): "
        f"{sqlite_methods - base_methods - _RETENTION_MIXIN_METHODS}\n"
        f"  extra base (missing in sqlite): {base_methods - sqlite_methods}"
    )


def test_optimizer_capability_is_optional_for_legacy_storage_backends() -> None:
    """A backend implementing the pre-replay ABC remains instantiable."""

    def legacy_implementation(*args: object, **kwargs: object) -> None:
        del args, kwargs

    legacy_methods = dict.fromkeys(
        BaseStorage.__abstractmethods__ - _OPTIONAL_OPTIMIZER_METHODS,
        legacy_implementation,
    )
    legacy_storage_type = type("LegacyStorage", (BaseStorage,), legacy_methods)

    storage = legacy_storage_type(org_id="legacy-org")

    for operation in (
        lambda: storage.get_playbook_optimization_job(1),
        lambda: storage.get_user_playbook_publication_subject_epochs(1),
    ):
        try:
            operation()
        except NotImplementedError as exc:
            assert "does not support" in str(exc)
        else:
            raise AssertionError("optional optimizer operation unexpectedly succeeded")
