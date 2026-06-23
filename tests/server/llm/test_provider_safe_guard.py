"""Tests for the provider-safe schema boundary guard (assert_provider_safe_schema).

The guard is the runtime net for structured-output schemas that bypass the
``StrictStructuredOutput`` base / the registry contract test (e.g. a model that
forgot the base, or a tool-argument schema). Under tests it must RAISE so a
regression fails CI loudly.
"""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from reflexio.models.structured_output import (
    StrictStructuredOutput,
    find_schema_keyword,
)
from reflexio.server.llm.llm_utils import assert_provider_safe_schema


class _VariantA(BaseModel):
    kind: Literal["a"] = "a"
    a: int


class _VariantB(BaseModel):
    kind: Literal["b"] = "b"
    b: str


_Union = Annotated[_VariantA | _VariantB, Field(discriminator="kind")]


class _UnsafeUnionModel(BaseModel):
    """Plain BaseModel with a discriminated union -> emits oneOf/discriminator."""

    choice: _Union


class _SafeUnionModel(StrictStructuredOutput):
    """Same union, but provider-safe by construction via the base."""

    choice: _Union


def test_guard_raises_on_model_that_forgot_the_base():
    """A discriminated-union model NOT inheriting the base trips the guard."""
    with pytest.raises(ValueError, match="provider-unsafe"):
        assert_provider_safe_schema(
            _UnsafeUnionModel.model_json_schema(), name="_UnsafeUnionModel"
        )


def test_guard_passes_for_base_inheriting_model():
    """The same union is accepted once the model inherits StrictStructuredOutput."""
    # Must not raise.
    assert_provider_safe_schema(
        _SafeUnionModel.model_json_schema(), name="_SafeUnionModel"
    )


def test_guard_ignores_property_literally_named_oneof():
    """A field literally named ``oneOf`` is a property name, not a keyword."""

    class _Weird(StrictStructuredOutput):
        oneOf: str = "x"  # noqa: N815 — intentional pathological field name

    schema = _Weird.model_json_schema()
    assert not find_schema_keyword(schema, "oneOf")  # structure-aware: no match
    assert_provider_safe_schema(schema, name="_Weird")  # must not raise
