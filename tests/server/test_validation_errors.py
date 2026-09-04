"""Contract for rendering a ``ValidationError`` into an HTTP response body.

Two hazards, and the second is the one that makes the obvious fix wrong.

A ``model_validator`` that raises ``ValueError`` puts the *live exception
object* into each error's ``ctx``, so ``json.dumps`` fails and the handler
building the 400 dies inside itself -- the caller gets a bare 500. The obvious
fix is "make it serializable", and that is exactly wrong: each error's ``input``
is the whole document that failed validation, so a serializable payload would
have returned ``storage_config``'s ``db_url`` password, ``api_key_config``,
``llm_config`` and ``pending_tool_call_config.hmac_secrets`` to the caller. The
serialization failure was the only thing preventing a credential leak.
"""

import json

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from reflexio.server.validation_errors import safe_validation_errors


class _Window(BaseModel):
    """Mirrors Config's stride/window relationship: an after-validator raising ValueError."""

    window_size: int
    stride_size: int
    db_url: str = ""

    @model_validator(mode="after")
    def _stride_fits(self) -> "_Window":
        if self.stride_size > self.window_size:
            raise ValueError("stride_size must be <= window_size")
        return self


def _error() -> ValidationError:
    with pytest.raises(ValidationError) as caught:
        _Window(
            window_size=4,
            stride_size=8,
            db_url="postgresql://u:SUPERSECRET@db.example.com/prod",
        )
    return caught.value


def test_raw_pydantic_errors_are_not_json_serializable() -> None:
    """The premise. If this ever passes, the 500 had a different cause."""
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(_error().errors())


def test_sanitised_errors_serialize() -> None:
    payload = json.dumps(safe_validation_errors(_error().errors()))
    assert "stride_size must be <= window_size" in payload


def test_sanitised_errors_do_not_echo_the_submitted_document() -> None:
    """The assertion a "just make it serializable" fix would fail.

    Asserting only on serializability would pass while shipping the password,
    so this checks the payload for the submitted values themselves.
    """
    payload = json.dumps(safe_validation_errors(_error().errors()))
    assert "SUPERSECRET" not in payload
    assert "db.example.com" not in payload
    assert "input" not in payload
    assert "ctx" not in payload


def test_sanitised_errors_keep_what_identifies_the_failure() -> None:
    """Stripping must not go so far that the response stops being useful."""
    errors = safe_validation_errors(_error().errors())
    assert len(errors) == 1
    assert set(errors[0]) == {"type", "loc", "msg"}
    assert errors[0]["type"] == "value_error"
    assert "stride_size must be <= window_size" in errors[0]["msg"]


def test_loc_is_a_list_so_it_round_trips_through_json() -> None:
    class _Nested(BaseModel):
        window_size: int

    with pytest.raises(ValidationError) as caught:
        _Nested(window_size="not-an-int")  # type: ignore[arg-type]
    errors = safe_validation_errors(caught.value.errors())
    assert errors[0]["loc"] == ["window_size"]
    assert json.loads(json.dumps(errors))[0]["loc"] == ["window_size"]
