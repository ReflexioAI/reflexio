from unittest.mock import MagicMock

import pytest

from tests.e2e_tests import conftest as e2e_conftest


def test_cleanup_deletes_storage_before_reporting_tagging_drain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = MagicMock()
    instance = MagicMock()
    instance.request_context.storage = storage
    monkeypatch.setattr(e2e_conftest, "drain_tagging", lambda **_kwargs: False)
    monkeypatch.setattr(
        e2e_conftest, "_get_playbook_names", lambda _instance: ["playbook"]
    )

    with pytest.raises(
        AssertionError,
        match="background tagging callbacks did not drain before storage cleanup",
    ):
        e2e_conftest._cleanup_storage(instance)

    storage.delete_all_user_playbooks_by_playbook_name.assert_called_once_with(
        "playbook"
    )
    storage.delete_all_agent_playbooks_by_playbook_name.assert_called_once_with(
        "playbook"
    )
    storage.delete_all_interactions.assert_called_once_with()
    storage.delete_all_profiles.assert_called_once_with()
    storage.delete_all_agent_success_evaluation_results.assert_called_once_with()
    storage.delete_all_requests.assert_called_once_with()
    storage.delete_all_operation_states.assert_called_once_with()
