"""Async hosted mem0 wrapper parity and cancellation behavior."""

import asyncio

import pytest


@pytest.mark.asyncio
async def test_async_add_mirrors_after_mem0_and_returns_exact_result(
    async_wrapped_cls, reflexio_mock
):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    result = await client.add("hello", user_id="u1", agent_id="a1")
    assert result is client.add_result
    reflexio_mock.publish_interaction_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_search_default_is_exact_and_opt_in_is_namespaced(
    async_wrapped_cls, reflexio_mock
):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    original = await client.search("q", filters={"user_id": "u1"})
    assert original is client.search_result
    reflexio_mock.search_async.assert_not_awaited()

    result = await client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert result is not client.search_result
    assert result["results"] is client.search_result["results"]
    assert result["reflexio"]["status"] == "ok"
    reflexio_mock.search_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_mem0_error_prevents_reflexio_call(
    async_wrapped_cls, reflexio_mock
):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    client.raise_on_add = ValueError("bad messages")
    with pytest.raises(ValueError, match="bad messages"):
        await client.add("hello", user_id="u1")
    reflexio_mock.publish_interaction_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_inherited_deletes_remain_mem0_only(
    async_wrapped_cls, reflexio_mock
):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    assert await client.delete("m1", delete_linked=True) == {"id": "m1"}
    assert await client.delete_users(user_id="u1") == {"message": "ok"}
    assert await client.reset() == {"message": "reset"}
    reflexio_mock.clear_user_data.assert_not_called()
    reflexio_mock.delete_request.assert_not_called()


@pytest.mark.asyncio
async def test_async_cancellation_is_not_swallowed(async_wrapped_cls, reflexio_mock):
    reflexio_mock.search_async.side_effect = asyncio.CancelledError
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    with pytest.raises(asyncio.CancelledError):
        await client.search("q", filters={"user_id": "u1"}, include_reflexio=True)


@pytest.mark.asyncio
async def test_async_reflexio_failure_returns_error_envelope(
    async_wrapped_cls, reflexio_mock
):
    reflexio_mock.search_async.side_effect = RuntimeError("down")
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    result = await client.search("q", filters={"user_id": "u1"}, include_reflexio=True)
    assert result["reflexio"]["status"] == "error"
    assert result["reflexio"]["reason"] == "request_failed"


@pytest.mark.asyncio
async def test_async_wrapper_yields_while_reflexio_search_waits(
    async_wrapped_cls, reflexio_mock
):
    async def delayed_search(**_kwargs):
        await asyncio.sleep(0.01)
        return reflexio_mock.search.return_value

    reflexio_mock.search_async.side_effect = delayed_search
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    heartbeat = False

    async def mark_heartbeat():
        nonlocal heartbeat
        await asyncio.sleep(0)
        heartbeat = True

    await asyncio.gather(
        client.search("q", filters={"user_id": "u1"}, include_reflexio=True),
        mark_heartbeat(),
    )
    assert heartbeat is True
