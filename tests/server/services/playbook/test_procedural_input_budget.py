"""Portable large-provenance regression; synthetic text and no provider dispatch."""

import copy
import json
import math
from typing import Any, cast

import litellm
import pytest
import yaml
from pydantic import Field, create_model

from reflexio.server.llm.input_budget import (
    InputBudgetExceededError,
    check_request_input_budget,
    measure_request_input_tokens,
    request_input_capacity,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.playbook.procedural import review, transport, workflow
from reflexio.server.services.playbook.procedural.catalog import catalog
from reflexio.server.services.playbook.procedural.compilation import compile_changes
from reflexio.server.services.playbook.procedural.models import (
    Change,
    Document,
    Input,
    Maintenance,
    Observation,
    Page,
    ResultVerdict,
    Source,
    Text,
)

pytestmark = pytest.mark.unit

MODEL = "minimax/MiniMax-M3"
CAPACITY = 65_536
# Fictional provenance includes the quotes, slashes and line breaks that JSON
# escapes. Different prefixes prevent accidental whole-span deduplication.
HISTORY = 'Fixture event: {"action": "stage", "path": "\\fixture\\release"}.\n'
SOURCE = 'Fixture instruction: {"action": "verify staging before production"}.\n'


class _CapturedError(Exception):
    def __init__(self, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.args_saved, self.kwargs = args, kwargs


class _CachedReader:
    def __init__(self, observations: list[Observation] | None = None):
        self.observations = observations
        self.cache_reads = 0

    def ask(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        if stage == "source" and self.observations is not None:
            self.cache_reads += 1
            return self.observations
        raise _CapturedError((stage, *args), kwargs)


def _fixture(repeats: int) -> tuple[Input, dict[int, list[str]]]:
    sources = [
        Source(
            interaction_id=i,
            request_id=f"fixture-request-{i}",
            owner="fixture-owner",
            role="User",
            request_source="fixture",
            text=(SOURCE * 200)[:10_200],
        )
        for i in range(1, 6)
    ]
    return Input(
        owner="fixture-owner",
        sources=sources,
        existing=[
            {
                "user_playbook_id": 100 + i,
                "user_id": "fixture-owner",
                "content": "Verify staging before production.",
                "trigger": "Preparing a fictional release",
                "source_interaction_ids": [1000 + i],
                "source_span": f"Fictional provenance {i}:\n"
                + HISTORY * (weight * repeats),
                "rationale": "A synthetic prior instruction.",
                "expanded_terms": None,
                "status": None,
                "playbook_status": None,
            }
            for i, weight in enumerate((12, 5, 3, 3, 2))
        ],
        agent_context="Fictional release assistant",
        definition="Learn source-grounded reusable procedures.",
        tool_context="",
    ), {source.interaction_id: [source.text] for source in sources}


def _stage_input(
    data: Input, visible: dict[int, list[str]], reader: _CachedReader
) -> _CapturedError:
    with pytest.raises(_CapturedError) as found:
        workflow.run_pipeline(data, visible, cast(Any, reader))
    return found.value


def _capture_request(
    stage: transport.StageClient, call: _CapturedError
) -> dict[str, Any]:
    with pytest.raises(_CapturedError) as found:
        stage.ask(*call.args_saved, **call.kwargs)
    return copy.deepcopy(found.value.kwargs)


def _built(native: LiteLLMClient, request: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(request)
    request.pop("provider_request_guard", None)
    return native._build_completion_params(**request)[0]


def _full_schema_floor(params: dict[str, Any]) -> int:
    messages = litellm.token_counter(
        model=params["model"],
        messages=params["messages"],
    )
    schema = litellm.token_counter(
        model=params["model"],
        text=json.dumps(
            {"tools": params["tools"], "tool_choice": params.get("tool_choice")},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    )
    return math.ceil((messages + schema) * 1.2)


def test_large_provenance_keeps_complete_compiler_and_review_within_native_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "true")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    native = LiteLLMClient(
        LiteLLMConfig(model=MODEL, max_tokens=16_384, timeout=240, fallback_models=[])
    )
    captured_calls: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> Any:
        captured_calls.append(kwargs)
        raise _CapturedError((), kwargs)

    def no_dispatch(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("Input-budget regression must never dispatch a provider")

    monkeypatch.setattr(native, "generate_chat_response_with_provenance", capture)
    monkeypatch.setattr(native, "_completion_with_hard_timeout", no_dispatch)
    monkeypatch.setattr(litellm, "completion", no_dispatch)
    stage = transport.StageClient(native, model=MODEL)
    data, visible = _fixture(0)
    source_call = _stage_input(data, visible, _CachedReader())
    source_request = _capture_request(stage, source_call)
    source_params = _built(native, source_request)

    # At most ten provider-free token probes calibrate the synthetic fixture to
    # the same near-ceiling geometry without pinning a tokenizer's exact count.
    low, high = 0, 512
    for _ in range(10):
        middle = (low + high + 1) // 2
        candidate, candidate_visible = _fixture(middle)
        body = _stage_input(candidate, candidate_visible, _CachedReader()).args_saved[2]
        source_params["messages"][1]["content"] = json.dumps(body, ensure_ascii=False)
        if measure_request_input_tokens(source_params) <= 63_800:
            low = middle
        else:
            high = middle - 1
    data, visible = _fixture(low)
    source_call = _stage_input(data, visible, _CachedReader())
    source_params["messages"][1]["content"] = json.dumps(
        source_call.args_saved[2], ensure_ascii=False
    )
    source_tokens = check_request_input_budget(
        source_params, context_limits={MODEL: 1_000_000}
    )
    assert 62_000 <= source_tokens <= 63_800
    _, quotes, _ = catalog(data, visible)
    assert len(data.sources) == 5 and len(quotes) == 85
    cached_reader = _CachedReader(
        [
            Observation(
                topic="Fixture", evidence=key, observation="Read the original source."
            )
            for key in list(quotes)[:14]
        ]
    )
    compiler_call = _stage_input(data, visible, cached_reader)
    assert cached_reader.cache_reads == 1  # No second source model invocation.
    compiler_request = _capture_request(stage, compiler_call)
    compiler_params = _built(native, compiler_request)

    with monkeypatch.context() as old_wire:
        old_wire.setattr(
            transport,
            "_input_content",
            lambda _stage, body: json.dumps(body, ensure_ascii=False),
        )
        old_wire.setattr(transport, "_input_instructions", lambda _: "", raising=False)
        stage.single_call["maintain"] = False
        flat_params = _built(native, _capture_request(stage, compiler_call))
    stage.single_call["maintain"] = True
    limits = {MODEL: 1_000_000, "minimax/MiniMax-M3.1-Flash": 1_000_000}
    assert request_input_capacity(flat_params, context_limits=limits) == CAPACITY
    with pytest.raises(InputBudgetExceededError):
        check_request_input_budget(flat_params, context_limits=limits)
    assert (
        check_request_input_budget(compiler_params, context_limits=limits) <= CAPACITY
    )
    assert (
        yaml.safe_load(compiler_params["messages"][1]["content"])
        == compiler_call.args_saved[2]
    )
    assert len(compiler_params["tools"]) == 1

    incumbents = {f"P{i}": row for i, row in enumerate(data.existing, 1)}
    quote_id = next(iter(quotes))
    quote = quotes[quote_id].text
    old = incumbents["P1"]
    assert len(old["source_span"]) > 30_000
    changes = Maintenance(
        changes=[
            Change(
                group="revise",
                operation="upsert",
                consume=["P1"],
                document=Document(
                    name="fixture-revision",
                    pages=[
                        Page(path="SKILL.md", content=old["content"] + "\n\n" + quote)
                    ],
                ),
                trigger=old["trigger"],
                support=[quote_id],
            ),
            Change(
                group="add",
                operation="upsert",
                consume=[],
                document=Document(
                    name="fixture-addition",
                    pages=[Page(path="SKILL.md", content=quote)],
                ),
                trigger="A separate fictional procedure",
                support=[quote_id],
            ),
        ]
    )
    compiled = compile_changes(changes, incumbents, quotes, visible)
    assert old["source_span"] in compiled.records["W0"]["source_span"]
    payload = {
        "source": compiler_call.args_saved[2],
        "changes": compiled.changes,
        "proposed_records": compiled.records,
        "keep_existing": compiled.keep_existing,
        "retired_ids": compiled.retired_ids,
    }
    full_review = review.prepare(payload)
    unit_type = create_model(
        "FixtureResultVerdict",
        __base__=ResultVerdict,
        unit=(Text, Field(json_schema_extra={"enum": ["U0", "U1"]})),
    )
    review_call = _CapturedError(
        (
            "review",
            workflow._prompt("review"),
            full_review,
            {"review_result": unit_type},
            lambda value: value,
            lambda _: None,
        ),
        {},
    )
    review_params = _built(native, _capture_request(stage, review_call))
    decoded = yaml.safe_load(review_params["messages"][1]["content"])
    assert review._expand_provenance(decoded) == full_review
    assert check_request_input_budget(review_params, context_limits=limits) <= CAPACITY
    for params in (source_params, compiler_params, review_params):
        assert measure_request_input_tokens(params) >= _full_schema_floor(params)
    assert stage.physical_calls == 0
    assert len(captured_calls) == 4  # Reader once; compiler new/old; reviewer once.
