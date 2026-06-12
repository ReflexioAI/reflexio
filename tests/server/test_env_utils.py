from __future__ import annotations

import pytest

from reflexio.server.env_utils import (
    env_get,
    env_required,
    env_required_literal,
    env_truthy,
)


def test_env_get_strips_values() -> None:
    assert env_get({"KEY": "  value  "}, "KEY") == "value"
    assert env_get({}, "KEY") == ""


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
