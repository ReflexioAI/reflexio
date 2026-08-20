"""``StorageConfig`` must not silently reinterpret a malformed config.

``StorageConfig`` is an *untagged* union. ``StorageConfigSQLite`` is its only
member with no required fields, so unless every member forbids extra keys it
becomes a catch-all: any payload failing the stricter members lands there
instead of raising.

That is not hypothetical. A managed tenant's Supabase config lost its
``db_url`` and validated as SQLite -- dropping ``url``/``key``/``schema`` --
which routed the org to a local file, produced a storage object with no
``_rpc``, and turned a background sweep into an error every few seconds for a
week (org 56, 2026-08). A config missing only ``key`` was worse still: it
validated as *Postgres* and connected successfully by a path nobody intended.

The invariant these tests defend:

    A storage config that loses a field fails loudly, rather than becoming a
    different backend.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from reflexio.models.config_schema import (
    StorageConfig,
    StorageConfigPostgres,
    StorageConfigSQLite,
    StorageConfigSupabase,
    validate_stored_config,
)

_ADAPTER: TypeAdapter[StorageConfig] = TypeAdapter(StorageConfig)

# The exact shape every managed org carries in production (verified 2026-08-20
# across the whole fleet: 29 managed orgs, all identical, zero extra keys).
_PROD_SUPABASE = {
    "url": "https://example.supabase.co",
    "key": "service-role-key",
    "db_url": "postgresql://user:pw@host:5432/postgres",
    "schema": "org_56",
    "read_url": "https://example-read.supabase.co",
    "read_key": "read-key",
}


@pytest.mark.parametrize(
    ("label", "payload", "expected"),
    [
        ("prod supabase shape", _PROD_SUPABASE, StorageConfigSupabase),
        (
            "supabase without optional read replica",
            {k: v for k, v in _PROD_SUPABASE.items() if not k.startswith("read_")},
            StorageConfigSupabase,
        ),
        # An empty payload resolving to SQLite is deliberate: it is the OSS
        # local-mode default (local_file_config_storage._default_storage_config).
        ("empty payload (OSS default)", {}, StorageConfigSQLite),
        (
            "sqlite with explicit path",
            {"db_path": "/var/lib/reflexio.db"},
            StorageConfigSQLite,
        ),
        (
            "postgres",
            {
                "type": "postgres",
                "db_url": "postgresql://u:p@h:5432/db",
                "schema": "org_1",
            },
            StorageConfigPostgres,
        ),
    ],
)
def test_valid_configs_resolve_to_their_own_backend(
    label: str, payload: dict[str, object], expected: type
) -> None:
    """Legitimate configs must keep resolving to the backend they name."""
    resolved = _ADAPTER.validate_python(payload)
    assert type(resolved) is expected, (
        f"{label}: expected {expected.__name__}, got {type(resolved).__name__}"
    )


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        # The org 56 shape: without extra="forbid" this validated as
        # StorageConfigSQLite and discarded url/key/schema in silence.
        (
            "supabase missing db_url",
            {k: v for k, v in _PROD_SUPABASE.items() if k != "db_url"},
        ),
        # Worse than the SQLite case: this validated as StorageConfigPostgres,
        # which then connects successfully via an unintended path.
        (
            "supabase missing key",
            {k: v for k, v in _PROD_SUPABASE.items() if k != "key"},
        ),
        (
            "supabase with empty-string key",
            {**_PROD_SUPABASE, "key": ""},
        ),
        (
            "supabase missing url",
            {k: v for k, v in _PROD_SUPABASE.items() if k != "url"},
        ),
        ("unrecognised payload", {"nonsense": 1}),
        ("supabase field misspelled", {**_PROD_SUPABASE, "db_urls": "x"}),
    ],
)
def test_malformed_configs_are_rejected_not_reinterpreted(
    label: str, payload: dict[str, object]
) -> None:
    """A config that loses or misspells a field must raise, not change backend."""
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(payload)


def test_sqlite_does_not_absorb_a_supabase_payload() -> None:
    """The specific coercion behind the org 56 outage, asserted directly.

    Constructing the SQLite model straight from a Supabase payload used to
    succeed by ignoring every key, yielding ``db_path=None``.
    """
    with pytest.raises(ValidationError):
        StorageConfigSQLite.model_validate(_PROD_SUPABASE)


def test_postgres_does_not_absorb_a_supabase_payload() -> None:
    """The sibling coercion -- same class of defect, different member."""
    with pytest.raises(ValidationError):
        StorageConfigPostgres.model_validate(_PROD_SUPABASE)


class TestThroughValidateStoredConfig:
    """The same invariant, asserted at the seam production actually uses.

    Every stored config is loaded through ``validate_stored_config``, not
    through the union adapter directly. That function passes ``extra="ignore"``
    for schema-evolution read compatibility, and pydantic cascades that setting
    into nested models -- so it silently overrode each StorageConfig member's
    ``extra="forbid"``. Asserting only against the adapter passes while the real
    load path stays broken.
    """

    @pytest.mark.parametrize(
        ("label", "storage_config"),
        [
            (
                "supabase missing db_url",
                {k: v for k, v in _PROD_SUPABASE.items() if k != "db_url"},
            ),
            (
                "supabase missing key",
                {k: v for k, v in _PROD_SUPABASE.items() if k != "key"},
            ),
            ("unrecognised payload", {"nonsense": 1}),
        ],
    )
    def test_malformed_storage_config_is_rejected_on_the_load_path(
        self, label: str, storage_config: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            validate_stored_config({"storage_config": storage_config})

    def test_valid_storage_configs_survive_the_load_path(self) -> None:
        for storage_config, expected in (
            (_PROD_SUPABASE, StorageConfigSupabase),
            ({"db_path": "/var/lib/reflexio.db"}, StorageConfigSQLite),
            (
                {"type": "postgres", "db_url": "postgresql://u:p@h:5432/db"},
                StorageConfigPostgres,
            ),
        ):
            loaded = validate_stored_config({"storage_config": storage_config})
            assert type(loaded.storage_config) is expected

    def test_storage_config_none_still_means_deployment_managed(self) -> None:
        """Self-host orgs persist None; it must stay None, not become SQLite."""
        assert validate_stored_config({"storage_config": None}).storage_config is None

    def test_retired_top_level_fields_are_still_ignored(self) -> None:
        """The leniency this function exists for must survive the new strictness.

        Schema evolution applies to Config's own retired fields -- not to
        reinterpreting which storage backend an org is on.
        """
        loaded = validate_stored_config(
            {"storage_config": None, "a_field_retired_three_releases_ago": 123}
        )
        assert loaded.storage_config is None


class TestTypeTagCompatibility:
    """The ``type`` tag must be accepted on input but absent from output.

    Two constraints pull in opposite directions and both are load-bearing:

    * The config API emits ``type`` alongside the storage config. Tightening the
      models without accepting it made a response from ``/get_config`` invalid
      to post back to ``/set_config`` -- a 422 on an unmodified round trip.
    * The boot-time config upgrade rewrites every org's persisted blob. Adding
      ``type`` to what the models serialize would change that document for the
      whole fleet, which this change has no business doing.
    """

    @pytest.mark.parametrize(
        ("tag", "payload", "expected"),
        [
            ("sqlite", {"db_path": None}, StorageConfigSQLite),
            ("supabase", _PROD_SUPABASE, StorageConfigSupabase),
            (
                "postgres",
                {"db_url": "postgresql://u:p@h:5432/db"},
                StorageConfigPostgres,
            ),
        ],
    )
    def test_tagged_payload_is_accepted(
        self, tag: str, payload: dict[str, object], expected: type
    ) -> None:
        """Exactly what the config API hands back to a client."""
        resolved = _ADAPTER.validate_python({**payload, "type": tag})
        assert type(resolved) is expected

    def test_a_wrong_tag_is_rejected(self) -> None:
        """The tag must agree with the shape, not override it."""
        with pytest.raises(ValidationError):
            _ADAPTER.validate_python({**_PROD_SUPABASE, "type": "sqlite"})

    @pytest.mark.parametrize(
        ("label", "payload"),
        [
            ("supabase", _PROD_SUPABASE),
            ("sqlite", {"db_path": "/var/lib/reflexio.db"}),
        ],
    )
    def test_serialized_shape_carries_no_type_tag(
        self, label: str, payload: dict[str, object]
    ) -> None:
        """Persisted documents must not gain a key from this change."""
        dumped = _ADAPTER.dump_python(
            _ADAPTER.validate_python(payload), mode="json"
        )
        assert "type" not in dumped, (
            f"{label}: serializing the tag would rewrite every org's stored "
            f"config blob at the next boot upgrade"
        )
        assert set(dumped) == set(payload) | {
            k for k in dumped if dumped[k] is None
        }


def test_round_trip_preserves_the_backend() -> None:
    """Serialising and revalidating must not change which backend a config is.

    Config blobs are persisted as JSON and reloaded on every request, so a
    round trip that drifts would reintroduce the same silent degradation.
    """
    for payload in (
        _PROD_SUPABASE,
        {},
        {"db_path": "/var/lib/reflexio.db"},
        {"type": "postgres", "db_url": "postgresql://u:p@h:5432/db"},
    ):
        original = _ADAPTER.validate_python(payload)
        reloaded = _ADAPTER.validate_python(_ADAPTER.dump_python(original, mode="json"))
        assert type(reloaded) is type(original), (
            f"round trip changed backend: {type(original).__name__} -> "
            f"{type(reloaded).__name__}"
        )
