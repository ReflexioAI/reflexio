import pytest
from pydantic import ValidationError

from reflexio.models.api_schema.domain.entities import Request


def test_request_has_metadata_field_with_default_empty_dict():
    """Request gains a metadata field. Default is an empty dict, NOT None."""
    r = Request(request_id="r1", user_id="u1")
    assert r.metadata == {}


def test_request_metadata_accepts_arbitrary_keys():
    r = Request(
        request_id="r1",
        user_id="u1",
        metadata={"reflexio_retrieval_enabled": True, "custom_key": "v"},
    )
    assert r.metadata["reflexio_retrieval_enabled"] is True
    assert r.metadata["custom_key"] == "v"


def test_request_metadata_roundtrips_through_model_dump():
    r = Request(
        request_id="r1",
        user_id="u1",
        metadata={"reflexio_retrieval_enabled": False},
    )
    parsed = Request(**r.model_dump())
    assert parsed.metadata == {"reflexio_retrieval_enabled": False}


def test_request_metadata_rejects_none():
    """metadata is always a dict — None is a ValidationError, not silently coerced."""
    with pytest.raises(ValidationError):
        Request(request_id="r1", user_id="u1", metadata=None)  # type: ignore[arg-type]


def test_request_metadata_default_is_independent_per_instance():
    """default_factory=dict guards against the shared-mutable-default trap.

    Mutating one instance's metadata must not bleed into another instance.
    """
    r1 = Request(request_id="r1", user_id="u1")
    r2 = Request(request_id="r2", user_id="u1")
    r1.metadata["k"] = "v"
    assert r2.metadata == {}
    assert r1.metadata is not r2.metadata


def test_request_metadata_nested_values_roundtrip():
    """Arbitrarily nested values survive model_dump() -> Request(**...) roundtrip."""
    nested = {"a": {"b": [1, 2, {"c": True}]}, "list": ["x", "y"]}
    r = Request(request_id="r1", user_id="u1", metadata=nested)
    parsed = Request(**r.model_dump())
    assert parsed.metadata == nested
