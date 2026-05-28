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
