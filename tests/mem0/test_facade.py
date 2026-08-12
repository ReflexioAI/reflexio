"""Scope-aware explicit Reflexio lifecycle operations."""

import types

import pytest


def _success(**values):
    return types.SimpleNamespace(success=True, **values)


def test_facade_is_read_only_and_reports_configuration(wrapped_cls, reflexio_mock):
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    assert client.reflexio.configured is True
    with pytest.raises(AttributeError):
        client.reflexio = None


def test_unconfigured_facade_raises_clear_error(wrapped_cls, monkeypatch):
    from reflexio.mem0 import ReflexioNotConfiguredError

    monkeypatch.delenv("REFLEXIO_API_KEY", raising=False)
    monkeypatch.delenv("REFLEXIO_URL", raising=False)
    client = wrapped_cls(api_key="mk")
    assert client.reflexio.configured is False
    with pytest.raises(ReflexioNotConfiguredError):
        client.reflexio.clear_user_data(user_id="u1")


def test_clear_user_data_uses_app_scoped_user(wrapped_cls, reflexio_mock):
    reflexio_mock.clear_user_data.return_value = _success(deleted_counts={})
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    response = client.reflexio.clear_user_data(user_id="u1", app_id="app-a")
    scoped_user = reflexio_mock.clear_user_data.call_args.args[0]
    assert response.success is True
    assert scoped_user.startswith("mem0-user-v1-")


def test_facade_keeps_sibling_app_user_and_session_scopes_distinct(
    wrapped_cls, reflexio_mock
):
    reflexio_mock.clear_user_data.return_value = _success(deleted_counts={})
    reflexio_mock.delete_session.return_value = _success(deleted_requests_count=1)
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)

    for app_id in ("app-a", "app-b"):
        client.reflexio.clear_user_data(user_id="same-user", app_id=app_id)
        client.reflexio.delete_session_records(
            user_id="same-user",
            app_id=app_id,
            agent_id="same-agent",
            run_id="same-run",
        )

    users = [call.args[0] for call in reflexio_mock.clear_user_data.call_args_list]
    sessions = [call.args[0] for call in reflexio_mock.delete_session.call_args_list]
    assert len(set(users)) == 2
    assert len(set(sessions)) == 2


def test_delete_session_records_matches_published_explicit_run(
    wrapped_cls, reflexio_mock
):
    reflexio_mock.delete_session.return_value = _success(deleted_requests_count=1)
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    client.add("hello", user_id="u1", app_id="app", agent_id="agent", run_id="run")
    published_session = reflexio_mock.publish_interaction.call_args.kwargs["session_id"]
    response = client.reflexio.delete_session_records(
        user_id="u1", app_id="app", agent_id="agent", run_id="run"
    )
    assert response.success is True
    reflexio_mock.delete_session.assert_called_once_with(
        published_session, wait_for_response=True
    )


def test_delete_session_records_matches_no_run_fallback(wrapped_cls, reflexio_mock):
    reflexio_mock.delete_session.return_value = _success(deleted_requests_count=1)
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    client.add("hello", user_id="u1", agent_id="agent")
    published_session = reflexio_mock.publish_interaction.call_args.kwargs["session_id"]
    client.reflexio.delete_session_records(user_id="u1", agent_id="agent")
    reflexio_mock.delete_session.assert_called_once_with(
        published_session, wait_for_response=True
    )


def test_id_deletes_wait_and_check_success(wrapped_cls, reflexio_mock):
    from reflexio.mem0 import ReflexioOperationError

    reflexio_mock.delete_profile.return_value = _success()
    reflexio_mock.delete_agent_playbook.return_value = _success()
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    client.reflexio.delete_profile(user_id="u1", profile_id="p1", app_id="app")
    scoped_user = reflexio_mock.delete_profile.call_args.args[0]
    assert scoped_user.startswith("mem0-user-v1-")
    assert reflexio_mock.delete_profile.call_args.kwargs == {
        "profile_id": "p1",
        "wait_for_response": True,
    }

    reflexio_mock.delete_agent_playbook.return_value = _success()
    client.reflexio.delete_agent_playbook(agent_playbook_id=3)
    reflexio_mock.delete_agent_playbook.assert_called_once_with(
        3, wait_for_response=True
    )

    reflexio_mock.delete_request.return_value = _success()
    reflexio_mock.delete_request.return_value.success = False
    with pytest.raises(ReflexioOperationError, match="delete_request failed"):
        client.reflexio.delete_request(request_id="r1")


def test_mem0_delete_all_remains_mem0_only(wrapped_cls, reflexio_mock):
    client = wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    result = client.delete_all(filters={"user_id": "u1"})
    assert result == {"message": "ok"}
    assert client.calls[-1] == (
        "delete_all",
        None,
        {"filters": {"user_id": "u1"}},
    )
    reflexio_mock.clear_user_data.assert_not_called()
    reflexio_mock.delete_session.assert_not_called()


@pytest.mark.asyncio
async def test_async_facade_uses_native_async_transport(
    async_wrapped_cls, reflexio_mock
):
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    response = await client.reflexio.delete_session_records(
        user_id="u1", app_id="app", agent_id="agent", run_id="run"
    )
    assert response.success is True
    reflexio_mock._make_async_request.assert_awaited_once()
    method, endpoint = reflexio_mock._make_async_request.call_args.args[:2]
    assert (method, endpoint) == ("DELETE", "/api/delete_session")


@pytest.mark.asyncio
async def test_async_facade_transport_failure_is_wrapped(
    async_wrapped_cls, reflexio_mock
):
    from reflexio.mem0 import ReflexioOperationError

    reflexio_mock._make_async_request.side_effect = RuntimeError("secret")
    client = async_wrapped_cls(api_key="mk", reflexio_client=reflexio_mock)
    with pytest.raises(ReflexioOperationError, match="delete_request failed"):
        await client.reflexio.delete_request(request_id="r1")
