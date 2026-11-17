"""Active production examples render and compile through native output contracts.

These are mechanics checks, not evidence that a model chooses the right intent.
"""

import json
import re

import pytest

from reflexio.server.prompt.prompt_manager import PromptManager
from reflexio.server.services.profile.components.consolidator import (
    ProfileDeduplicationOutput,
)
from reflexio.server.services.profile.profile_generation_service_utils import (
    StructuredProfilesOutput,
)


@pytest.mark.parametrize("kind", ["author", "consolidator"])
def test_active_production_prompt_examples_render_into_current_output_contract(kind):
    manager = PromptManager()
    if kind == "author":
        identity = "profile_update_instruction_start"
        variables = {
            "agent_context_prompt": "",
            "context_prompt": "",
            "extraction_definition_prompt": "User facts and preferences",
        }
        schema = StructuredProfilesOutput
    else:
        identity = "profile_deduplication"
        variables = {
            "new_profile_count": 0,
            "new_profiles": "(None)",
            "existing_profile_count": 0,
            "existing_profiles": "(None)",
        }
        schema = ProfileDeduplicationOutput
    rendered = manager.render_prompt(identity, variables)
    examples = [
        json.loads(match)
        for match in re.findall(r"```json\s*(.*?)\s*```", rendered, re.DOTALL)
    ]
    checked = 0
    for example in examples:
        if kind == "author" and "profiles" not in example:
            continue  # The optional attachment-tool example is a different schema.
        schema.model_validate(example)
        checked += 1
    assert checked > 1


@pytest.mark.parametrize(
    "identity,version,variables",
    [
        (
            "profile_update_instruction_start",
            "1.4.0",
            {
                "agent_context_prompt": "",
                "context_prompt": "",
                "extraction_definition_prompt": "User facts",
            },
        ),
        (
            "profile_deduplication",
            "1.2.0",
            {
                "new_profile_count": 0,
                "new_profiles": "(None)",
                "existing_profile_count": 0,
                "existing_profiles": "(None)",
            },
        ),
    ],
)
def test_frozen_historical_identity_still_renders_after_new_default_activation(
    identity, version, variables
):
    legacy = PromptManager(version_override={identity: version})
    frozen = legacy.get_prompt_template_identity(identity)
    active = PromptManager()
    assert active.get_active_version(identity) != version
    assert active.render_prompt_from_identity(
        identity, variables, frozen
    ) == legacy.render_prompt(identity, variables)
