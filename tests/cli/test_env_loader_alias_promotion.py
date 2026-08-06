"""Legacy DSN alias promotion — see ``env_loader.promote_legacy_env_aliases``.

The bug this guards: ``.env`` files load with ``override=False``, which only
protects a canonical name already present in the process environment. It does
nothing when the environment supplied the LEGACY name and the file supplies the
CANONICAL one — the file wins, so a process handed a real prod DSN as
``DATA_SUPABASE_DB_URL`` silently reads the developer's local ``DATA_DB_URL``.
"""

import os

import pytest

from reflexio.cli import env_loader

_PROD = "postgresql://u:p@prod-db.example.com:5432/reflexio"
_LOCAL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        "DATA_DB_URL",
        "DATA_SUPABASE_DB_URL",
        "POSTGRES_DB_URL",
        "REFLEXIO_STORAGE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_promotes_legacy_when_canonical_unset(monkeypatch):
    monkeypatch.setenv("DATA_SUPABASE_DB_URL", _PROD)
    env_loader.promote_legacy_env_aliases()
    assert os.environ["DATA_DB_URL"] == _PROD


def test_a_dotenv_canonical_value_can_no_longer_shadow_the_legacy_one(
    monkeypatch, tmp_path
):
    """The actual regression: legacy injected, canonical only in the .env file.

    Promotion runs first, so the file's ``override=False`` load can no longer
    replace the injected value — which is what silently redirected a prod
    verification at a local database.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(f"DATA_DB_URL={_LOCAL}\n")
    monkeypatch.setenv("DATA_SUPABASE_DB_URL", _PROD)

    env_loader.promote_legacy_env_aliases()
    env_loader._load_dotenv_pruned(env_file)

    assert os.environ["DATA_DB_URL"] == _PROD, "local .env shadowed the injected DSN"


def test_never_overwrites_an_explicit_canonical_value(monkeypatch):
    monkeypatch.setenv("DATA_DB_URL", _PROD)
    monkeypatch.setenv("DATA_SUPABASE_DB_URL", _LOCAL)
    env_loader.promote_legacy_env_aliases()
    assert os.environ["DATA_DB_URL"] == _PROD


def test_blank_canonical_is_treated_as_unset(monkeypatch):
    """``.env.self_host`` ships ``DATA_DB_URL=`` — blank must not block promotion."""
    monkeypatch.setenv("DATA_DB_URL", "")
    monkeypatch.setenv("DATA_SUPABASE_DB_URL", _PROD)
    env_loader.promote_legacy_env_aliases()
    assert os.environ["DATA_DB_URL"] == _PROD


@pytest.mark.parametrize(
    "key,value",
    [("POSTGRES_DB_URL", _LOCAL), ("REFLEXIO_STORAGE", "postgres")],
)
def test_defers_to_the_storage_aware_resolver_when_ambiguous(monkeypatch, key, value):
    """Postgres in play → the Supabase alias is not necessarily the intended DSN."""
    monkeypatch.setenv("DATA_SUPABASE_DB_URL", _PROD)
    monkeypatch.setenv(key, value)
    env_loader.promote_legacy_env_aliases()
    assert os.environ.get("DATA_DB_URL", "") == ""


def test_no_legacy_value_is_a_noop():
    env_loader.promote_legacy_env_aliases()
    assert "DATA_DB_URL" not in os.environ
