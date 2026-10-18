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

    Missing required data fields fail loudly. Retired HTTP fields never choose
    execution, and canonical backend identity survives serialization.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from reflexio.config_migration import StoredConfigurationError
from reflexio.models.config_schema import (
    Config,
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
        ("supabase blank db_url", {**_PROD_SUPABASE, "db_url": ""}),
        ("unknown backend", {**_PROD_SUPABASE, "type": "unknown"}),
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
                "supabase blank db_url",
                {**_PROD_SUPABASE, "db_url": ""},
            ),
            ("unrecognised payload", {"nonsense": 1}),
        ],
    )
    def test_malformed_storage_config_is_rejected_on_the_load_path(
        self, label: str, storage_config: dict[str, object]
    ) -> None:
        with pytest.raises(StoredConfigurationError):
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
    """Remote backend identity must survive credential-free serialization."""

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
    def test_canonical_shape_preserves_backend_identity(
        self, label: str, payload: dict[str, object]
    ) -> None:
        dumped = _ADAPTER.dump_python(_ADAPTER.validate_python(payload), mode="json")
        if label == "supabase":
            assert dumped == {
                "type": "supabase",
                "db_url": payload["db_url"],
                "schema": payload["schema"],
            }
        else:
            assert dumped == payload


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


@pytest.mark.parametrize(
    "private_key",
    ["private-tool-map-key", "enabled", "model", "db_url", "similarity_threshold"],
)
def test_stored_policy_error_and_traceback_omit_values_and_dynamic_keys(private_key):
    import traceback

    secret = "private-customer-content"
    payload = {
        "storage_config": None,
        "agent_context_prompt": secret,
        "pending_tool_call_config": {
            "tool_overrides": {private_key: {"similarity_threshold": 2}},
        },
    }
    try:
        validate_stored_config(payload)
    except StoredConfigurationError as exc:
        rendered = "".join(traceback.format_exception(exc))
        assert secret not in rendered
        assert f".{private_key}." not in str(exc)
        assert "tool_overrides" not in str(exc)
        assert "pending_tool_call_config" in str(exc)
        if private_key == "private-tool-map-key":
            assert private_key not in rendered
        assert "input_value" not in rendered
    else:
        pytest.fail("invalid recognized saved policy was accepted")


@pytest.mark.parametrize("value", [-1, "private-invalid-merge-budget"])
def test_invalid_current_value_in_legacy_conversion_also_fails_closed(value):
    import traceback

    payload = {
        "storage_config": None,
        "playbook_optimizer_config": {
            "use_merge": True,
            "max_merge_invocations": value,
        },
    }
    with pytest.raises(StoredConfigurationError) as error:
        validate_stored_config(payload)
    assert "private-invalid-merge-budget" not in "".join(
        traceback.format_exception(error.value)
    )


@pytest.mark.parametrize(
    "read_url,read_key", [(None, None), ("https://unused.example", "unused"), ("", "")]
)
def test_obsolete_http_reader_drops_without_changing_storage_identity(
    read_url, read_key
):
    payload = {**_PROD_SUPABASE, "read_url": read_url, "read_key": read_key}
    before = dict(payload)
    config = validate_stored_config({"storage_config": payload})
    assert isinstance(config.storage_config, StorageConfigSupabase)
    canonical = config.storage_config.model_dump(mode="json")
    assert canonical == {
        "type": "supabase",
        "db_url": payload["db_url"],
        "schema": payload["schema"],
    }
    assert payload == before
    assert _ADAPTER.validate_python(canonical) == config.storage_config
    assert "read_url" not in StorageConfigSupabase.model_fields
    assert "read_key" not in StorageConfigSupabase.model_fields


@pytest.mark.parametrize("field", ["url", "key"])
@pytest.mark.parametrize("remove", [False, True])
def test_obsolete_http_credentials_are_not_required_or_serialized(field, remove):
    payload = dict(_PROD_SUPABASE)
    if remove:
        payload.pop(field)
    else:
        payload[field] = ""
    loaded = validate_stored_config({"storage_config": payload}).storage_config
    assert isinstance(loaded, StorageConfigSupabase)
    canonical = loaded.model_dump(mode="json")
    assert canonical["type"] == "supabase"
    assert not set(canonical).intersection({"url", "key", "read_url", "read_key"})


@pytest.mark.parametrize("stored", [False, True])
@pytest.mark.parametrize("tag", [None, "supabase", "postgres"])
def test_same_native_shape_keeps_legacy_or_explicit_backend_identity(stored, tag):
    payload = {"db_url": "postgresql://user:pw@host/db", "schema": "org_56"}
    if tag:
        payload["type"] = tag
    if stored:
        resolved = validate_stored_config({"storage_config": payload}).storage_config
    else:
        resolved = _ADAPTER.validate_python(payload)
    expected = StorageConfigSupabase if tag == "supabase" else StorageConfigPostgres
    assert isinstance(resolved, expected)
    canonical = resolved.model_dump(mode="json")
    assert type(_ADAPTER.validate_python(canonical)) is type(resolved)


def test_managed_response_cannot_absorb_an_invalid_remote_reference():
    payload = {"managed_by": "platform", "type": "supabase", "db_url": ""}
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(payload)


@pytest.mark.parametrize("backend", ["supabase", "postgres"])
def test_managed_native_marker_serializes_its_backend_identity(backend):
    payload = {"type": backend, "managed_by": "platform", "schema_present": False}
    config = Config.model_validate({"storage_config": payload})
    assert config.model_dump()["storage_config"] == payload
    assert (
        validate_stored_config(config.model_dump()).storage_config.storage_type
        == backend
    )
