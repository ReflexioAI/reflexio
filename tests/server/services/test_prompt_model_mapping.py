"""Trip-wire tests detecting prompt version changes that need mock updates.

When a prompt template is updated (new version file added to prompt_bank/),
the corresponding test here fails with a clear message asking the developer
to verify that the mock response in ``MODEL_REGISTRY`` still matches the
expected output schema.

This forces an explicit review of mock behavior whenever prompts change,
preventing silent mock drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# prompt_bank/ relative to this test file
_PROMPT_BANK_DIR = (
    Path(__file__).resolve().parents[3]
    / "reflexio"
    / "server"
    / "prompt"
    / "prompt_bank"
)

# Mapping: prompt_id -> (latest known version, model registry key or None).
# "model registry key" is the key in llm_model_registry.MODEL_REGISTRY that
# holds the expected mock response for this prompt's structured output.
# None means the prompt does not produce structured output relevant to mocking.
PROMPT_VERSION_MAP: dict[str, tuple[str, str | None]] = {
    "playbook_extraction_main": ("v1.0.0", "playbook_extraction"),
    "playbook_extraction_main_incremental": ("v1.0.0", "playbook_extraction"),
    "playbook_extraction_context": ("v4.0.0", None),
    "playbook_extraction_context_incremental": ("v4.0.0", None),
    "playbook_should_generate": ("v3.0.0", "boolean_evaluation"),
    "playbook_should_generate_expert": ("v1.0.0", "boolean_evaluation"),
    "playbook_extraction_context_expert": ("v3.0.0", None),
    "playbook_extraction_main_expert": ("v1.0.0", "playbook_extraction"),
    "playbook_aggregation": ("v2.0.0", "playbook_aggregation"),
    "playbook_deduplication": ("v2.0.0", "playbook_deduplication"),
    "profile_update_main": ("v1.0.0", "profile_extraction"),
    "profile_update_main_incremental": ("v1.0.0", "profile_extraction"),
    "profile_update_instruction_start": ("v1.0.0", None),
    "profile_update_instruction_incremental": ("v1.0.0", None),
    "profile_should_generate": ("v1.0.0", "boolean_evaluation"),
    "profile_should_generate_override": ("v1.0.0", "boolean_evaluation"),
    "profile_deduplication": ("v1.0.0", "profile_deduplication"),
    "agent_success_evaluation": ("v1.0.0", "agent_success_evaluation"),
    "agent_success_evaluation_with_comparison": (
        "v1.0.0",
        "agent_success_evaluation_comparison",
    ),
    "shadow_content_evaluation": ("v1.0.0", None),
    "query_reformulation": ("v1.0.0", None),
    "document_expansion": ("v1.0.0", None),
}


def _get_latest_prompt_version(prompt_id: str) -> str:
    """Scan prompt_bank/<prompt_id>/ for the latest v*.prompt.md file."""
    prompt_dir = _PROMPT_BANK_DIR / prompt_id
    if not prompt_dir.is_dir():
        pytest.fail(f"Prompt directory not found: {prompt_dir}")
    versions = sorted(prompt_dir.glob("v*.prompt.md"))
    if not versions:
        pytest.fail(f"No version files found in {prompt_dir}")
    return versions[-1].stem.split(".prompt")[0]


class TestPromptVersionMapping:
    """Detect prompt version changes that may require mock updates."""

    @pytest.mark.parametrize("prompt_id", list(PROMPT_VERSION_MAP.keys()))
    def test_prompt_version_matches_known(self, prompt_id):
        """Fail if a prompt has been updated without updating this mapping.

        When this test fails, you need to:
        1. Verify the mock response in MODEL_REGISTRY still matches what
           the new prompt version expects as output
        2. Update PROMPT_VERSION_MAP in this file to the new version
        3. Run ``pytest --snapshot-update`` if snapshot tests also fail
        """
        expected_version, registry_key = PROMPT_VERSION_MAP[prompt_id]
        actual_version = _get_latest_prompt_version(prompt_id)

        registry_hint = ""
        if registry_key:
            registry_hint = (
                f" Verify MODEL_REGISTRY['{registry_key}'] "
                f"still matches the expected output schema."
            )

        assert actual_version == expected_version, (
            f"Prompt '{prompt_id}' has been updated to {actual_version} "
            f"(expected {expected_version}).{registry_hint}"
        )

    def test_all_prompt_dirs_are_mapped(self):
        """Every prompt_bank directory should appear in PROMPT_VERSION_MAP."""
        prompt_dirs = {
            p.name
            for p in _PROMPT_BANK_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".") and any(p.glob("v*.prompt.md"))
        }
        mapped = set(PROMPT_VERSION_MAP.keys())
        unmapped = prompt_dirs - mapped
        assert not unmapped, (
            f"Prompt directories not in PROMPT_VERSION_MAP: {unmapped}. "
            f"Add them with their latest version and registry key."
        )
