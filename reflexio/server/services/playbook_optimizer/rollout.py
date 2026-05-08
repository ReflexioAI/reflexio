from __future__ import annotations

from reflexio.models.api_schema.domain import AgentPlaybook

from .assistant_webhook import AssistantCallable
from .models import ChatMessage, RolloutTrace, ScenarioWindow


class MultiTurnRollout:
    """Replay a scenario's user turns against an assistant backend.

    For each user turn in ``window.user_turns`` (capped by ``max_turns``),
    the rollout appends the user message, asks the backend for a reply, and
    appends the reply to the history. Importantly, the *user side* is fixed
    — replayed verbatim from the recorded scenario — so a paired
    incumbent/candidate run differs only in the playbook injected into the
    backend. That isolation is what makes the judge's verdict meaningful.
    """

    def __init__(self, assistant: AssistantCallable) -> None:
        self.assistant = assistant

    def run(
        self,
        *,
        window: ScenarioWindow,
        playbook: AgentPlaybook,
        max_turns: int,
    ) -> RolloutTrace:
        history: list[ChatMessage] = []
        for user_turn in window.user_turns[:max_turns]:
            history.append(user_turn)
            assistant_content = self.assistant(history, [playbook])
            history.append(ChatMessage(role="assistant", content=assistant_content))
        return RolloutTrace(messages=history, playbook_content=playbook.content)
