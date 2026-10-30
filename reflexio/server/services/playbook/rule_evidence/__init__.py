"""Four-dimensional rule evidence followed by one indexed, exact-clause repair.

Callers supply an invocation function using their existing LLM transport. This
method owns neither credentials, retries, completion labels nor publication.
"""

import json
from collections.abc import Callable
from importlib.resources import files
from typing import Any

import tiktoken

from reflexio.models.structured_output import StrictStructuredOutput

from .models import (
    Diagnostic,
    RuleEvidenceInput,
    RuleEvidenceResult,
    Session,
)

Invoke = Callable[
    [str, dict[str, object], type[StrictStructuredOutput]], StrictStructuredOutput
]
PAYLOAD_PROXY_TOKEN_LIMIT = 60_000


def prompt(name: str) -> str:
    """Read the packaged prompt whose exact text callers bind into identity."""
    if name not in {"judge", "tune"}:
        raise ValueError("unknown rule-evidence prompt")
    return files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")


def request_input_tokens(
    system: str, payload: dict[str, object], schema: type[StrictStructuredOutput]
) -> int:
    """Price the actual portable consumer envelope without dispatch or mutation."""
    rendered = system + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    rendered += json.dumps(schema.model_json_schema(), sort_keys=True)
    return len(
        tiktoken.get_encoding("cl100k_base").encode(rendered, disallowed_special=())
    )


def _check_budget(
    system: str, payload: dict[str, object], schema: type[StrictStructuredOutput]
) -> None:
    # Proxy only, not the provider tokenizer. Never silently drop evidence.
    if request_input_tokens(system, payload, schema) > PAYLOAD_PROXY_TOKEN_LIMIT:
        raise ValueError(
            "rule evidence exceeds input budget; use a smaller source batch"
        )


def _source(data: RuleEvidenceInput, session: Session) -> dict[str, object]:
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "success_definition": data.success_definition,
        "agent_context": data.agent_context,
        "available_tools": [
            tool.model_dump(mode="json") for tool in data.available_tools
        ],
        "tool_context_status": data.tool_context_status,
        "source_limitations": session.source_limitations,
        "observation_boundary": (
            "Stored history; end of capture does not prove task termination. "
            "Missing tool contracts are uncertainty, not evidence of unavailable tools."
        ),
        "messages": [message.model_dump(mode="json") for message in session.messages],
    }


def validate_diagnostic(
    diagnostic: Diagnostic, data: RuleEvidenceInput, session: Session
) -> None:
    """Reject invented/cross-session coordinates and unsupported edit spans."""
    if (diagnostic.session_id, diagnostic.user_id) != (
        session.session_id,
        session.user_id,
    ):
        raise ValueError("diagnostic must belong to this session and owner")
    turns = {message.turn for message in session.messages}
    assessment = diagnostic.rule_assessment
    for item in [*diagnostic.observations, assessment]:
        for reference in item.evidence_refs:
            if (
                reference.session_id != session.session_id
                or reference.turn not in turns
            ):
                raise ValueError("evidence must cite supplied turns of this session")
    quote = assessment.incumbent_quote
    if quote and quote not in data.incumbent.content:
        raise ValueError("rule quote must match the exposed incumbent")
    if (
        assessment.disposition in {"rule_gap", "adherence_failure"}
        and not quote.strip()
    ):
        raise ValueError("gap/adherence assessment needs the actual rule quote")
    if assessment.disposition == "rule_gap":
        if data.incumbent.content.find(quote) != data.incumbent.content.rfind(quote):
            raise ValueError("admitted gap must occur exactly once")
        if not any(item.kind == "problem" for item in diagnostic.observations):
            raise ValueError("rule gap requires an observed problem")
    if assessment.disposition == "unknown" and not diagnostic.missing_information:
        raise ValueError("unknown assessment requires deciding missing information")


def run_rule_evidence(data: RuleEvidenceInput, invoke: Invoke) -> RuleEvidenceResult:
    """Run the promoted V7 method; failures remain explicit, not abstentions."""
    data = RuleEvidenceInput.model_validate_json(data.model_dump_json())
    from .wire import diagnose, rebuild

    diagnostics, audits = diagnose(data, invoke)
    system, payload, schema = proposal_request(data, diagnostics, audits)
    _check_budget(system, payload, schema)
    value = invoke(system, payload, schema)
    return rebuild(
        data, diagnostics, schema.model_validate_json(value.model_dump_json())
    )


def proposal_request(
    data: RuleEvidenceInput, diagnostics: list[Diagnostic], audits: list[dict[str, Any]]
) -> tuple[str, dict[str, object], type[StrictStructuredOutput]]:
    """Render the complete proposal envelope; partial diagnostics are sizing only."""
    from .wire import catalog, reference_catalog, tune_schema

    payload = {
        "original_source": data.model_dump(mode="json"),
        "diagnostics": [d.model_dump(mode="json") for d in diagnostics],
        "target_audits": audits,
        "original_clauses": [
            {"id": e["id"], "text": e["text"]} for e in catalog(data.incumbent.content)
        ],
        "source_references": reference_catalog(data),
    }
    schema = tune_schema(data)
    system = prompt("tune")
    return system, payload, schema
