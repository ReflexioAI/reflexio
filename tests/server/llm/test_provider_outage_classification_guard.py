"""Every LLM failure classifier consults the one provider-outage rule.

``provider_wait_seconds`` decides whether a failure is a provider outage: the
provider's breaker is open (or refused the call), so the work waits and is not
charged. Durable extraction used it; the open-world attempt boundary, the
agent-success retry and the tagging pass each had their own transient/terminal
split, and during the 2026-11-06 MiniMax quota outage those ended jobs as
``infrastructure_failure`` or spent every bounded retry on refusals that never
reached the provider.

A module that classifies LLM failures -- it calls one of the transient
classifiers, names ``ProviderCircuitOpenError``, or catches a client error --
must consult ``provider_wait_seconds`` or be recorded below with the reason it
makes no retry decision of its own. The guard fails on an unrecorded module and
on an entry that no longer describes one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy_guard

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "reflexio"

#: Names whose use means "this module sorts LLM failures".
CLASSIFIER_NAMES = frozenset(
    {
        "describe_tool_loop_error",
        "is_provider_unavailable_error",
        "_is_expected_transient_llm_error",
        "rate_limit_evidence",
        "ProviderCircuitOpenError",
    }
)
#: Exceptions whose handler is a classification decision.
CAUGHT_NAMES = frozenset({"LiteLLMClientError", "ProviderCircuitOpenError"})
SHARED_RULE = "provider_wait_seconds"

#: Modules that classify LLM failures without deciding retry or terminality.
EXEMPT: dict[str, str] = {
    "server/llm/_litellm_text_generation.py": (
        "the client: defines the transient classification and feeds the breaker"
    ),
    "server/llm/provider_circuit.py": "defines provider_wait_seconds itself",
    "server/llm/tools.py": (
        "describe_tool_loop_error labels a tool-loop error for logs and the "
        "loop's own result; it schedules nothing"
    ),
    "server/services/offline_tuner/open_world/analyst.py": (
        "in-attempt rate-limit backoff inside one budget; exhaustion and every "
        "other failure reach rule_evidence_runtime's boundary, which consults "
        "provider_wait_seconds"
    ),
    "server/services/profile/components/extractor.py": (
        "chooses a log level only; the durable window (durable_learning/"
        "worker.py) decides the retry with provider_wait_seconds"
    ),
    "server/services/extraction/protected_pass.py": (
        "re-raises the client error (or the capacity error it caused); the "
        "durable window decides"
    ),
    "server/services/agent_success_evaluation/components/"
    "retrieved_learning_evaluator.py": (
        "KNOWN GAP, recorded rather than hidden: a judge failure becomes a "
        "failed/degraded chunk, so an outage spends generation_service's "
        "bounded retrieved-learning re-arms. Waiting needs the outage carried "
        "through the run outcome and a decision on partial (degraded) runs"
    ),
}


def _classifying_modules() -> dict[str, bool]:
    """Relative path -> whether it consults the shared rule, for classifiers."""
    found: dict[str, bool] = {}
    for path in sorted(_SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: set[str] = set()
        caught = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.alias):
                names.add(node.name.rsplit(".", 1)[-1])
            elif isinstance(node, ast.ExceptHandler) and node.type is not None:
                types = (
                    node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
                )
                caught |= any(
                    (t.id if isinstance(t, ast.Name) else getattr(t, "attr", ""))
                    in CAUGHT_NAMES
                    for t in types
                )
        if caught or names & CLASSIFIER_NAMES:
            found[path.relative_to(_SOURCE_ROOT).as_posix()] = SHARED_RULE in names
    return found


def test_every_failure_classifier_consults_the_outage_rule() -> None:
    unrecorded = sorted(
        module
        for module, shared in _classifying_modules().items()
        if not shared and module not in EXEMPT
    )
    assert unrecorded == [], (
        "These modules classify LLM failures without provider_wait_seconds. "
        "Defer while the provider is unavailable (see durable_learning/"
        "worker.py), or record in EXEMPT why the module decides no retry: "
        f"{unrecorded}"
    )


def test_every_exemption_still_describes_a_classifier_outside_the_rule() -> None:
    found = _classifying_modules()
    stale = sorted(module for module in EXEMPT if module not in found or found[module])
    assert stale == [], f"Retire these EXEMPT entries: {stale}"


def test_the_scan_sees_the_known_consumers() -> None:
    """A scan that finds nothing would pass both checks above."""
    found = _classifying_modules()
    for consumer in (
        "server/services/generation_service.py",
        "server/services/offline_tuner/open_world/rule_evidence_runtime.py",
        "server/services/extraction/resumable_agent.py",
    ):
        assert found.get(consumer) is True, consumer
