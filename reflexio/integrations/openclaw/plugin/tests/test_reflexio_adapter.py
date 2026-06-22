"""Tests for openclaw_smart.reflexio_adapter."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from openclaw_smart.reflexio_adapter import Adapter


def test_default_url_is_8071():
    # 8071/8072 matches claude-smart so the two plugins share one local
    # reflexio backend; 8081 is reserved for a developer's own instance.
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
