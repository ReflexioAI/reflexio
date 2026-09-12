from __future__ import annotations

import pytest

from reflexio.server.env_utils import (
    EnvBoolError,
    env_bool,
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


def test_env_bool_accepts_true_and_false_case_insensitively() -> None:
    for raw in ("true", "TRUE", "True", "  true  "):
        assert env_bool("KEY", default=False, env={"KEY": raw}) is True
    for raw in ("false", "FALSE", "False", "  false  "):
        assert env_bool("KEY", default=True, env={"KEY": raw}) is False


def test_env_bool_treats_unset_and_blank_as_the_default() -> None:
    """Same invariant env_str carries: a `KEY=` line reads as unset."""
    assert env_bool("KEY", default=True, env={}) is True
    assert env_bool("KEY", default=False, env={}) is False
    assert env_bool("KEY", default=True, env={"KEY": ""}) is True
    assert env_bool("KEY", default=True, env={"KEY": "   "}) is True


@pytest.mark.parametrize("raw", ["1", "0", "yes", "no", "on", "off", "y", "n"])
def test_env_bool_refuses_the_values_that_were_parsed_inconsistently(raw: str) -> None:
    """These are the spellings that meant different things in different modules.

    `on` was truthy under env_truthy and falsy under the `{1,true,yes}` set; `y`
    worked in exactly one module. Accepting them here would preserve the
    ambiguity this helper exists to remove.
    """
    with pytest.raises(EnvBoolError):
        env_bool("KEY", default=False, env={"KEY": raw})


def test_env_bool_refuses_a_typo_rather_than_silently_returning_false() -> None:
    """The defect that motivated this helper.

    `env_truthy("ture")` is False -- indistinguishable from a deliberate
    `false`, so a mistyped safety knob switches itself off in silence. A wrong
    value must be loud.
    """
    with pytest.raises(EnvBoolError) as excinfo:
        env_bool(
            "REFLEXIO_REQUIRE_DATA_DB",
            default=False,
            env={"REFLEXIO_REQUIRE_DATA_DB": "ture"},
        )

    message = str(excinfo.value)
    assert "REFLEXIO_REQUIRE_DATA_DB" in message, "the error must name the variable"
    assert "ture" in message, "the error must quote the offending value"


def test_env_truthy_keeps_its_permissive_set_for_external_callers() -> None:
    """Deliberate: it is a public export, so narrowing it would break callers.

    Undocumented dev/CI knobs still route through it -- `REFLEXIO_REQUIRE_DOCKER=1`
    is set in ci-fast.yml and must keep working.
    """
    assert env_truthy("1") is True
    assert env_truthy("on") is True
    assert env_truthy("ture") is False
