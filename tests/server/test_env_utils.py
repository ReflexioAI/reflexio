from __future__ import annotations

import pytest

from reflexio.server.env_utils import (
    env_get,
    env_required,
    env_required_literal,
    env_str,
    env_truthy,
)


def test_env_get_strips_values() -> None:
    assert env_get({"KEY": "  value  "}, "KEY") == "value"
    assert env_get({}, "KEY") == ""


def test_env_str_treats_unset_and_blank_as_default() -> None:
    # Unset -> default.
    assert env_str("KEY", "fallback", env={}) == "fallback"
    # Blank / whitespace-only -> default (the empty-equals-unset invariant).
    assert env_str("KEY", "fallback", env={"KEY": ""}) == "fallback"
    assert env_str("KEY", "fallback", env={"KEY": "   "}) == "fallback"
    # Real value -> stripped value, never the default.
    assert env_str("KEY", "fallback", env={"KEY": "  real  "}) == "real"


def test_env_str_default_is_empty_string() -> None:
    assert env_str("KEY", env={}) == ""


def test_env_str_reads_os_environ_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REFLEXIO_TEST_ENV_STR", "  from-os  ")
    assert env_str("REFLEXIO_TEST_ENV_STR", "fallback") == "from-os"
    monkeypatch.setenv("REFLEXIO_TEST_ENV_STR", "")
    assert env_str("REFLEXIO_TEST_ENV_STR", "fallback") == "fallback"


def test_env_required_raises_for_missing_value() -> None:
    with pytest.raises(RuntimeError, match="KEY is required for enterprise startup"):
        env_required({"KEY": " "}, "KEY")


def test_env_required_literal_normalizes_and_validates() -> None:
    assert (
        env_required_literal(
            {"MODE": " Platform "},
            "MODE",
            allowed=("platform", "self_host"),
        )
        == "platform"
    )
    with pytest.raises(RuntimeError, match="MODE must be 'platform' or 'self_host'"):
        env_required_literal(
            {"MODE": "local"}, "MODE", allowed=("platform", "self_host")
        )


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_env_truthy_true_values(value: str) -> None:
    assert env_truthy(value) is True


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no"])
def test_env_truthy_false_values(value: str) -> None:
    assert env_truthy(value) is False
