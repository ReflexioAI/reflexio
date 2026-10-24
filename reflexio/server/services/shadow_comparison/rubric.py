"""One frozen prompt identity for shadow verdict rendering and cohort reads."""

from typing import Any

from reflexio.server.prompt.prompt_manager import PromptManager

_PROMPT_ID = "shadow_comparison"


class ShadowComparisonRubric:
    """Bind a verdict cohort to the exact unrendered template being executed."""

    def __init__(self, prompt_manager: PromptManager) -> None:
        self._manager = prompt_manager
        self._identity = prompt_manager.get_prompt_template_identity(_PROMPT_ID)

    @property
    def key(self) -> str:
        """Return a template identity, without user/session content or labels."""
        return (
            f"v{self._identity['active_version']}@sha256:"
            f"{self._identity['template_content_digest']}"
        )

    def render(self, variables: dict[str, Any]) -> str:
        """Render the frozen template, refusing a changed template's bytes."""
        return self._manager.render_prompt_from_identity(
            _PROMPT_ID, variables, self._identity
        )
