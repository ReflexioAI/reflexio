"""Every configuration reader chooses the same canonical database target."""

import pytest

from reflexio.config_env import resolve_data_db_url


@pytest.mark.parametrize("backend", ["postgres", "supabase", ""])
def test_canonical_target_is_backend_independent(backend):
    assert (
        resolve_data_db_url({"REFLEXIO_STORAGE": backend, "DATA_DB_URL": " canonical "})
        == "canonical"
    )
    assert resolve_data_db_url({"REFLEXIO_STORAGE": backend, "DATA_DB_URL": " "}) == ""


@pytest.mark.parametrize("name", ["DATA_SUPABASE_DB_URL", "POSTGRES_DB_URL"])
@pytest.mark.parametrize("canonical", [None, "same", "different"])
@pytest.mark.parametrize("legacy", ["", " ", "same", "private-secret"])
def test_old_targets_are_rejected_without_values(name, canonical, legacy):
    env = {name: legacy}
    if canonical is not None:
        env["DATA_DB_URL"] = canonical
    with pytest.raises(ValueError, match=name) as error:
        resolve_data_db_url(env)
    assert "DATA_DB_URL" in str(error.value)
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, (10, 30.0, None, None)),
        ({"REFLEXIO_STORAGE_POOL_SIZE": "24"}, (10, 30.0, None, None)),
        (
            {
                "REFLEXIO_POSTGRES_POOL_SIZE": " 3 ",
                "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT": " 0.2 ",
            },
            (3, 0.2, None, None),
        ),
        (
            {
                "REFLEXIO_POSTGRES_READ_POOL_SIZE": "2",
                "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT": "0.1",
            },
            (10, 30.0, 2, 0.1),
        ),
        (
            {
                "REFLEXIO_POSTGRES_POOL_SIZE": "3",
                "REFLEXIO_STORAGE_POOL_SIZE": "24",
                "REFLEXIO_POSTGRES_READ_POOL_SIZE": " ",
            },
            (3, 30.0, None, None),
        ),
        (
            {
                "REFLEXIO_POSTGRES_POOL_SIZE": " ",
                "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT": " ",
                "REFLEXIO_POSTGRES_READ_POOL_SIZE": " ",
                "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT": " ",
            },
            (10, 30.0, None, None),
        ),
    ],
)
def test_native_pool_defaults_preserve_backend_scope(env, expected):
    from dataclasses import astuple

    from reflexio.config_env import resolve_postgres_pool_defaults

    assert astuple(resolve_postgres_pool_defaults(env)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "REFLEXIO_POSTGRES_POOL_SIZE",
        "REFLEXIO_POSTGRES_READ_POOL_SIZE",
        "REFLEXIO_STORAGE_POOL_SIZE",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", "nan", "inf"])
def test_invalid_pool_capacity_fails_with_setting_name(name, value):
    from reflexio.config_env import (
        resolve_postgres_pool_defaults,
        resolve_storage_pool_size,
    )

    resolve = (
        resolve_storage_pool_size
        if name == "REFLEXIO_STORAGE_POOL_SIZE"
        else resolve_postgres_pool_defaults
    )
    with pytest.raises(ValueError, match=name):
        resolve({name: value})


@pytest.mark.parametrize(
    "name",
    [
        "REFLEXIO_POSTGRES_POOL_ACQUIRE_TIMEOUT",
        "REFLEXIO_POSTGRES_READ_POOL_ACQUIRE_TIMEOUT",
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "invalid", "nan", "inf", "-inf"])
def test_invalid_pool_wait_fails_with_setting_name(name, value):
    from reflexio.config_env import resolve_postgres_pool_defaults

    with pytest.raises(ValueError, match=name):
        resolve_postgres_pool_defaults({name: value})
