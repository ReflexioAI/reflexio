"""Verdict provenance must describe the actual prompt, independently of config."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest

from reflexio.models.api_schema.domain.entities import Interaction
from reflexio.models.api_schema.eval_overview_schema import ShadowComparisonOutput
from reflexio.models.config_schema import Config, StorageConfigSQLite
from reflexio.server.api_endpoints.request_context import RequestContext
from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.shadow_comparison.dispatcher import (
    dispatch_shadow_comparison_judge,
)
from reflexio.server.services.shadow_comparison.rubric import ShadowComparisonRubric

pytestmark = pytest.mark.unit


def _manager(tmp_path, override=None):
    bank = tmp_path / "prompts" / "shadow_comparison"
    bank.mkdir(parents=True)
    for version, active, marker in (
        ("1.0.0", False, "HISTORICAL"),
        ("2.0.0", True, "CURRENT"),
    ):
        (bank / f"v{version}.prompt.md").write_text(
            f"---\nactive: {str(active).lower()}\n"
            "variables: [request_1_response, request_2_response]\n---\n"
            f"{marker} {{request_1_response}} versus {{request_2_response}}\n"
        )
    return PromptManager(
        prompt_bank_path=bank.parent,
        version_override={"shadow_comparison": override} if override else None,
    )


@pytest.mark.parametrize("override", [None, "1.0.0"])
def test_dispatch_records_the_identity_of_the_template_it_rendered(tmp_path, override):
    manager = _manager(tmp_path, override)
    identity = manager.get_prompt_template_identity("shadow_comparison")
    config = Config(storage_config=StorageConfigSQLite())
    context = cast(
        RequestContext,
        SimpleNamespace(
            configurator=SimpleNamespace(get_config=lambda: config),
            prompt_manager=manager,
        ),
    )
    client = MagicMock()
    client.generate_chat_response.return_value = ShadowComparisonOutput(
        better_request="1",
        is_significantly_better=True,
        comparison_reason="The first response follows the request.",
    )
    storage = MagicMock()
    storage.supports_shadow_comparison_verdicts.return_value = True
    dispatch_shadow_comparison_judge(
        storage=storage,
        interactions=[
            Interaction(
                interaction_id=42,
                user_id="rubric-user",
                request_id="rubric-request",
                role="assistant",
                content="reflexio response",
                shadow_content="baseline response",
            )
        ],
        session_id="rubric-session",
        agent_version="agent-version",
        request_context=context,
        llm_client=client,
    )
    prompt = client.generate_chat_response.call_args.kwargs["messages"][0]["content"]
    assert prompt.startswith("HISTORICAL" if override else "CURRENT")
    storage.save_shadow_comparison_verdict.assert_called_once()
    verdict = storage.save_shadow_comparison_verdict.call_args.args[0]
    assert verdict.judge_prompt_version == (
        f"v{identity['active_version']}@sha256:{identity['template_content_digest']}"
    )


def test_frozen_rubric_keeps_selection_and_refuses_changed_template_bytes(tmp_path):
    manager = _manager(tmp_path, "1.0.0")
    rubric = ShadowComparisonRubric(manager)
    original_key = rubric.key
    manager.version_override = {"shadow_comparison": "2.0.0"}
    variables = {"request_1_response": "first", "request_2_response": "second"}
    assert rubric.render(variables).startswith("HISTORICAL")
    assert rubric.key == original_key

    path = tmp_path / "prompts" / "shadow_comparison" / "v1.0.0.prompt.md"
    path.write_text(path.read_text().replace("HISTORICAL", "CHANGED"))
    with pytest.raises(ValueError, match="drifted"):
        rubric.render(variables)
    manager.version_override = {"shadow_comparison": "1.0.0"}
    assert ShadowComparisonRubric(manager).key != original_key


def test_rubric_identity_does_not_depend_on_rendered_user_content(tmp_path):
    rubric = ShadowComparisonRubric(_manager(tmp_path))
    original_key = rubric.key
    first = rubric.render(
        {"request_1_response": "private A", "request_2_response": "B"}
    )
    second = rubric.render(
        {"request_1_response": "private C", "request_2_response": "D"}
    )
    assert first != second
    assert rubric.key == original_key
