"""Tests for openclaw_smart.reflexio_adapter."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openclaw_smart.reflexio_adapter import Adapter

_ADAPTER_LOGGER = "openclaw_smart.reflexio_adapter"


def test_default_url_is_8071():
    # 8071/8072 matches claude-smart so the two plugins share one local
    # reflexio backend; 8061 is reserved for a developer's own instance.
    adapter = Adapter()
    assert adapter.url == "http://localhost:8071/"


def test_env_var_overrides_url(monkeypatch):
    monkeypatch.setenv("REFLEXIO_URL", "http://example.com:9000/")
    adapter = Adapter()
    assert adapter.url == "http://example.com:9000/"


def test_explicit_url_wins_over_env(monkeypatch):
    monkeypatch.setenv("REFLEXIO_URL", "http://example.com:9000/")
    adapter = Adapter(url="http://other:7000/")
    assert adapter.url == "http://other:7000/"


def test_get_client_returns_none_on_construction_failure():
    adapter = Adapter()
    with patch(
        "openclaw_smart.reflexio_adapter.ReflexioClient",
        side_effect=ConnectionError,
        create=True,
    ):
        # ReflexioClient is imported lazily inside _get_client; we patch on
        # the imported module to mimic that path.
        pass
    with patch.dict(
        "sys.modules",
        {"reflexio": MagicMock(ReflexioClient=MagicMock(side_effect=ConnectionError))},
    ):
        assert adapter._get_client() is None


def test_publish_returns_true_for_empty_interactions():
    adapter = Adapter()
    assert adapter.publish(session_id="s", project_id="p", interactions=[]) is True


def test_publish_returns_false_when_client_none():
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=None):
        assert (
            adapter.publish(
                session_id="s",
                project_id="p",
                interactions=[{"role": "User", "content": "x"}],
            )
            is False
        )


def test_publish_passes_openclaw_agent_version():
    fake_client = MagicMock()
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        adapter.publish(
            session_id="s1",
            project_id="proj",
            interactions=[{"role": "User", "content": "x"}],
        )
        kwargs = fake_client.publish_interaction.call_args[1]
        assert kwargs["agent_version"] == "openclaw"
        assert kwargs["user_id"] == "proj"
        assert kwargs["session_id"] == "s1"
        assert kwargs["wait_for_response"] is False


def test_publish_forwards_force_extraction():
    fake_client = MagicMock()
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        adapter.publish(
            session_id="s1",
            project_id="p",
            interactions=[{"role": "User"}],
            force_extraction=True,
            skip_aggregation=True,
        )
        kwargs = fake_client.publish_interaction.call_args[1]
        assert kwargs["force_extraction"] is True
        assert kwargs["skip_aggregation"] is True


def test_publish_returns_false_on_exception():
    fake_client = MagicMock()
    fake_client.publish_interaction.side_effect = RuntimeError("boom")
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        assert (
            adapter.publish(
                session_id="s",
                project_id="p",
                interactions=[{"role": "User"}],
            )
            is False
        )


def test_search_all_degrades_to_empty():
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=None):
        u, a, p = adapter.search_all(project_id="p", query="q", top_k=3)
        assert u == [] and a == [] and p == []


def test_search_all_passes_agent_version():
    fake_client = MagicMock()
    fake_client.search.return_value = MagicMock(
        user_playbooks=[], agent_playbooks=[], profiles=[]
    )
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        adapter.search_all(project_id="p", query="q", top_k=5)
        kwargs = fake_client.search.call_args[1]
        assert kwargs["agent_version"] == "openclaw"
        assert kwargs["user_id"] == "p"


def test_fetch_user_playbooks_degrades_to_empty():
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=None):
        assert adapter.fetch_user_playbooks(project_id="p") == []


def test_fetch_agent_playbooks_filters_rejected():
    fake_client = MagicMock()
    fake_client.search_agent_playbooks.return_value = MagicMock(
        agent_playbooks=[
            {"id": "a", "playbook_status": "approved"},
            {"id": "b", "playbook_status": "rejected"},
            {"id": "c", "playbook_status": "pending"},
        ]
    )
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        result = adapter.fetch_agent_playbooks(top_k=5)
        ids = [item["id"] for item in result]
        assert "b" not in ids
        assert "a" in ids and "c" in ids


def test_apply_extraction_defaults_skips_when_already_matching():
    fake_client = MagicMock()
    config = MagicMock(window_size=10, stride_size=5)
    fake_client.get_config.return_value = config
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        assert adapter.apply_extraction_defaults(window_size=10, stride_size=5) is True
        fake_client.set_config.assert_not_called()


def test_apply_extraction_defaults_writes_when_different():
    fake_client = MagicMock()
    config = MagicMock(window_size=99, stride_size=99)
    fake_client.get_config.return_value = config
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        adapter.apply_extraction_defaults(window_size=10, stride_size=5)
        assert config.window_size == 10
        assert config.stride_size == 5
        fake_client.set_config.assert_called_once_with(config)


def test_fetch_stall_state_returns_none_when_client_none():
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=None):
        assert adapter.fetch_stall_state() is None


def test_mark_stall_notified_swallows_errors():
    fake_client = MagicMock()
    fake_client.mark_stall_notified.side_effect = RuntimeError
    adapter = Adapter()
    with patch.object(adapter, "_get_client", return_value=fake_client):
        # Must not raise
        adapter.mark_stall_notified()


class _WarningsPropertyRaises:
    """A response object whose ``warnings`` access blows up.

    ``getattr(obj, "warnings", None)`` swallows only ``AttributeError`` — any
    other exception a property raises propagates straight through the default.
    A pydantic model with a computed field, or a lazy client wrapper that
    re-reads the socket on attribute access, is exactly this shape.
    """

    @property
    def warnings(self) -> list[str]:
        raise RuntimeError("warnings unavailable")


class _MappingGetRaises(dict):
    """A ``dict`` subclass whose ``.get`` raises.

    ``isinstance(response, dict)`` is True, so the mapping branch is taken and
    the override — not ``dict.get`` — is what runs.
    """

    def get(self, *args: object, **kwargs: object) -> object:
        raise KeyError("get is not safe on this mapping")


class _UnprintableWarning:
    """A warning item that cannot be rendered: ``str(item)`` raises."""

    def __str__(self) -> str:
        raise ValueError("cannot render this warning")


class TestPublishWarnings:
    """The server reports fields it could not bind; the hook log must show them.

    Silent field-dropping is the defect this channel exists for: a publish of
    50 mis-keyed interactions returned 200 and stored 50 empty rows.
    """

    def _publish_with_response(self, response, caplog) -> bool:
        fake_client = MagicMock()
        fake_client.publish_interaction.return_value = response
        adapter = Adapter()
        with (
            patch.object(adapter, "_get_client", return_value=fake_client),
            caplog.at_level(logging.WARNING, logger=_ADAPTER_LOGGER),
        ):
            return adapter.publish(
                session_id="s",
                project_id="p",
                interactions=[{"role": "User", "content": "x"}],
            )

    @staticmethod
    def _adapter_records(caplog):
        return [r for r in caplog.records if r.name == _ADAPTER_LOGGER]

    def test_server_warnings_are_logged(self, caplog):
        response = SimpleNamespace(
            warnings=["interaction_data_list[0]: ignored unrecognised field(s) Content"]
        )
        assert self._publish_with_response(response, caplog) is True
        assert "unrecognised field(s) Content" in caplog.text

    def test_dict_shaped_response_is_read(self, caplog):
        assert (
            self._publish_with_response({"warnings": ["dropped foo"]}, caplog) is True
        )
        assert "dropped foo" in caplog.text

    def test_quiet_when_there_is_nothing_to_report(self, caplog):
        # Assert on records, not on the absence of a log substring: a reworded
        # message would make a substring check silently vacuous, and this test
        # is the only thing standing between a clean publish and log noise.
        assert self._publish_with_response(SimpleNamespace(warnings=[]), caplog) is True
        assert self._adapter_records(caplog) == []

    @pytest.mark.parametrize(
        "response", [None, SimpleNamespace(), {"warnings": None}, {"warnings": 5}]
    )
    def test_odd_response_shapes_still_report_success(self, response, caplog):
        """A publish the server accepted must never be reported as failed.

        `publish_unpublished` advances the buffer watermark only on True, so
        raising while reading diagnostics would re-send the same accepted
        batch on every subsequent hook — duplicates forever, caused purely by
        the code that was supposed to improve observability. This is why the
        warning read sits outside the try that guards the publish call.
        """
        assert self._publish_with_response(response, caplog) is True

    @pytest.mark.parametrize(
        "response",
        [
            pytest.param(_WarningsPropertyRaises(), id="warnings-property-raises"),
            pytest.param(_MappingGetRaises(warnings=["x"]), id="mapping-get-raises"),
            pytest.param(
                {"warnings": [_UnprintableWarning()]}, id="warning-item-str-raises"
            ),
        ],
    )
    def test_hostile_response_shapes_still_report_success(self, response, caplog):
        """The shapes that actually escape the isinstance guard.

        The parametrisation above only exercises inputs the guard already
        rejects, so it asserts totality without testing it. These three reach
        past it — attribute access, ``.get``, and ``str()`` are each a call
        into caller-controlled code that can raise anything. The consequence of
        a leaked exception is not a lost log line: ``publish`` would propagate
        instead of returning True, ``publish_unpublished`` would leave the
        buffer watermark where it was, and the same server-accepted batch would
        be re-sent on every subsequent hook, forever.
        """
        assert self._publish_with_response(response, caplog) is True
