"""Verify the F1 shadow_comparison prompt is registered and active."""

from pathlib import Path

_PROMPT_ROOT = (
    Path(__file__).resolve().parents[3]
    / "reflexio"
    / "server"
    / "prompt"
    / "prompt_bank"
    / "shadow_comparison"
)


def test_shadow_comparison_v1_0_0_prompt_file_exists():
    assert (_PROMPT_ROOT / "v1.0.0.prompt.md").is_file()


def test_shadow_comparison_v1_0_0_prompt_is_active():
    content = (_PROMPT_ROOT / "v1.0.0.prompt.md").read_text()
    assert "active: true" in content


def test_shadow_comparison_v1_0_0_prompt_declares_all_variables():
    content = (_PROMPT_ROOT / "v1.0.0.prompt.md").read_text()
    for var in ("user_message", "request_1_response", "request_2_response"):
        assert var in content, f"missing variable {var}"


def test_shadow_comparison_v1_0_0_prompt_warns_against_position_weighting():
    """The criterion-list in the prompt body must include a 'position
    is arbitrary' instruction so the judge doesn't blindly trust 1 or 2."""
    content = (_PROMPT_ROOT / "v1.0.0.prompt.md").read_text()
    assert "Position" in content or "position" in content


def test_shadow_comparison_v1_0_0_prompt_warns_against_assuming_prior_history():
    """Q2-1 MVP constraint: the judge sees current-turn-only and must
    not invent prior context."""
    content = (_PROMPT_ROOT / "v1.0.0.prompt.md").read_text()
    assert "prior" in content.lower() or "history" in content.lower()
