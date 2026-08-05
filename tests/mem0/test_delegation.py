"""Unwrapped methods delegate to the mem0 base class unchanged."""


def test_isinstance_of_mem0_client(wrapped_cls, mem0_stub, reflexio_mock):
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    assert isinstance(client, mem0_stub.MemoryClient)


def test_unwrapped_methods_hit_base_without_reflexio(wrapped_cls, reflexio_mock):
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    assert client.get("m1") == {"id": "m1"}
    assert client.delete_all(user_id="u1") == {"message": "ok"}
    assert [c[0] for c in client.calls] == ["get", "delete_all"]
    reflexio_mock.publish_interaction.assert_not_called()
    reflexio_mock.search.assert_not_called()


def test_constructor_forwards_mem0_args(wrapped_cls, reflexio_mock):
    sentinel_http = object()
    client = wrapped_cls(
        api_key="mk",
        host="https://api.example",
        client=sentinel_http,
        reflexio_client=reflexio_mock,
    )
    assert client.api_key == "mk"
    assert client.host == "https://api.example"
    assert client.client is sentinel_http
