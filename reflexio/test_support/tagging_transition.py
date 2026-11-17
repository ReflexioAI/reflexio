# ruff: noqa: S101 -- Assertions in shared synthetic contract exercises.
"""Actual SQLite user/root ownership for direct and scheduled profile tagging."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain import UserProfile
from reflexio.server.services.durable_learning.learning_admission import (
    current_learning_work,
)
from reflexio.server.services.storage.storage_base.learning_transition import (
    LearningPolicyAuthority,
    LearningTransitionBlockedError,
)
from reflexio.server.services.tagging.service import TaggingService, TagsOutput


def build(context: Any, monkeypatch: Any) -> tuple[TaggingService, MagicMock]:
    monkeypatch.delenv("MOCK_LLM_RESPONSE", raising=False)
    monkeypatch.setattr(
        "reflexio.server.services.tagging.service.SiteVarManager.get_site_var",
        lambda *_: {},
    )
    context.prompt_manager.render_prompt.return_value = "synthetic tagging prompt"
    config = context.configurator.get_config()
    config.profile_extractor_config.tagging_definition_prompt = "synthetic tags"
    context.storage.add_user_profile(
        "user",
        [
            UserProfile(
                profile_id="tagged-profile",
                user_id="user",
                content="synthetic committed fact",
                embedding=[0.0] * 512,
                generated_from_request_id="request",
                last_modified_timestamp=1,
            )
        ],
        skip_embedding=True,
    )
    client = MagicMock()
    return TaggingService(client, context), client


def seal(storage: Any) -> LearningPolicyAuthority:
    from reflexio.server.services.storage.storage_base.learning_transition import (
        _change_learning_policy_authority_db,
    )

    with storage._learning_policy_authority_scope() as (db, root):
        return _change_learning_policy_authority_db(
            db,
            action="seal",
            operation_id="tagging-hold",
            expected_nonce=root.authority_nonce,
            expected_generation=root.generation,
            target_policy=root.active_policy,
        )


def exercise_open(context: Any, monkeypatch: Any) -> None:
    service, client = build(context, monkeypatch)

    def reply(**_kwargs: Any) -> TagsOutput:
        work = current_learning_work()
        assert work is not None
        assert work.kind == "tagging" and work.user_id == "user"
        assert not context.storage.in_commit_scope
        return TagsOutput(tags=["qualified"])

    client.generate_chat_response.side_effect = reply
    service.run(
        user_id="user", agent_version="v1", tag_playbooks=False, tag_evaluations=False
    )
    assert client.generate_chat_response.call_count == 1
    assert context.storage.get_user_profile("user")[0].tags == ["qualified"]
    assert current_learning_work() is None


def exercise_sealed(
    context: Any, monkeypatch: Any, *, seal_after_response: bool = False
) -> None:
    service, client = build(context, monkeypatch)
    if seal_after_response:

        def reply(**_kwargs: Any) -> TagsOutput:
            assert not context.storage.in_commit_scope
            seal(context.storage)
            return TagsOutput(tags=["must-not-write"])

        client.generate_chat_response.side_effect = reply
    else:
        seal(context.storage)
    with pytest.raises(LearningTransitionBlockedError):
        service.run(
            user_id="user",
            agent_version="v1",
            tag_playbooks=False,
            tag_evaluations=False,
        )
    assert client.generate_chat_response.call_count == int(seal_after_response)
    assert context.storage.get_user_profile("user")[0].tags is None
    assert current_learning_work() is None
