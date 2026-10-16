"""Every configuration reader chooses the same canonical database target."""

import pytest

from reflexio.config_env import resolve_data_db_url


@pytest.mark.parametrize("backend", ["postgres", "supabase", ""])
def test_canonical_target_wins(backend: str) -> None:
    assert (
        resolve_data_db_url(
            {
                "REFLEXIO_STORAGE": backend,
                "DATA_DB_URL": " canonical ",
                "POSTGRES_DB_URL": "postgres",
                "DATA_SUPABASE_DB_URL": "supabase",
            }
        )
        == "canonical"
    )


@pytest.mark.parametrize(
    "backend,expected",
    [("postgres", "postgres"), ("supabase", "supabase"), ("", "supabase")],
)
def test_blank_canonical_uses_selected_backend_alias(
    backend: str, expected: str
) -> None:
    assert (
        resolve_data_db_url(
            {
                "REFLEXIO_STORAGE": backend,
                "DATA_DB_URL": "  ",
                "POSTGRES_DB_URL": " postgres ",
                "DATA_SUPABASE_DB_URL": " supabase ",
            }
        )
        == expected
    )


@pytest.mark.parametrize(
    "backend,other_alias",
    [("postgres", "DATA_SUPABASE_DB_URL"), ("supabase", "POSTGRES_DB_URL")],
)
def test_selected_backend_does_not_use_other_backend_alias(
    backend: str, other_alias: str
) -> None:
    assert (
        resolve_data_db_url({"REFLEXIO_STORAGE": backend, other_alias: "other"}) == ""
    )
