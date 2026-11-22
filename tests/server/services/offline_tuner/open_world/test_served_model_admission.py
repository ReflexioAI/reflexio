"""Served-model admission for the production open-world transport.

A provider resolves a floating alias to a dated snapshot, so ``gpt-4o`` may
answer as ``gpt-4o-2024-08-06``. That is the configured revision; a different
model (``gpt-4o-mini``, another family) is a substitution and is refused.
"""

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import json

import litellm
import pytest

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm._native_model_ref import (
    CLI_TIMEOUT_ENV,
    ENV_NATIVE_MODEL,
    PLUGIN_OPENCODE_EXECUTABLE_ENV,
)
from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig
from reflexio.server.services.offline_tuner.open_world.analyst import (
    ChatMessage,
    _is_provider_date_stamp,
    _LiteLLMOpenWorldModelClient,
    _native_cli_served_stamps,
    _served_model_admitted,
)


@pytest.mark.parametrize(
    ("configured", "served"),
    [
        ("gpt-4o", "gpt-4o"),
        ("gpt-4o", "gpt-4o-2024-08-06"),
        ("claude-x", "claude-x-20250101"),
        ("MiniMax-M3", "MiniMax-M3"),
    ],
)
def test_exact_and_dated_snapshots_are_admitted(configured, served):
    assert _served_model_admitted(configured_bare=configured, observed_bare=served)


@pytest.mark.parametrize(
    ("configured", "served"),
    [
        # A sibling model sharing the alias as a prefix.
        ("gpt-4o", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o-mini-2024-07-18"),
        # Partial or malformed date stamps.
        ("gpt-4o", "gpt-4o-2024-08"),
        ("gpt-4o", "gpt-4o-202408"),
        ("gpt-4o", "gpt-4o-2024-13-01"),
        ("gpt-4o", "gpt-4o-20241301"),
        ("gpt-4o", "gpt-4o-2024-08-06-preview"),
        ("gpt-4o", "gpt-4o2024-08-06"),
        ("gpt-4o", "gpt-4o-2024-08-06-2024-08-07"),
        # A different family entirely, with and without a stamp.
        ("gpt-4o", "claude-x-20250101"),
        ("claude-x", "gpt-4o"),
        # A dated configuration never admits the bare alias or another date.
        ("gpt-4o-2024-08-06", "gpt-4o"),
        ("gpt-4o-2024-08-06", "gpt-4o-2024-11-20"),
        # Case and emptiness are not normalised away.
        ("gpt-4o", "GPT-4o"),
        ("", "-2024-08-06"),
    ],
)
def test_near_misses_and_substitutions_are_refused(configured, served):
    assert not _served_model_admitted(configured_bare=configured, observed_bare=served)


class _Answer(StrictStructuredOutput):
    answer: str


def _compose(
    monkeypatch,
    *,
    model: str,
    served: str,
    hidden: dict[str, str] | None = None,
    with_usage: bool = True,
    dispatched: list[str] | None = None,
):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-synthetic-test")
    route = model.split("/", 1)[0] if "/" in model else "openai"
    if hidden is None:
        hidden = {"custom_llm_provider": route}

    def dispatch(self, params, deadline):
        del self, deadline
        if dispatched is not None:
            dispatched.append(params["model"])
        response = litellm.ModelResponse(
            model=served,
            choices=[
                {
                    "message": {"role": "assistant", "content": '{"answer": "ok"}'},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        response._hidden_params = dict(hidden)
        if not with_usage:
            setattr(response, "usage", None)  # noqa: B010 - untyped on ModelResponse
        return response

    # The lowest dispatch seam: the reviewed generate/completion methods that
    # the transport identity pins stay untouched.
    monkeypatch.setattr(LiteLLMClient, "_run_completion", dispatch)
    return _LiteLLMOpenWorldModelClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=512, fallback_models=[]))
    )


def _call(client, model: str):
    return client.complete(
        messages=[ChatMessage(role="user", content="synthetic evidence")],
        response_format=_Answer,
        model_revision=model,
        idempotency_key="synthetic-key",
    )


def _complete(monkeypatch, *, model: str, served: str, **options):
    client = _compose(monkeypatch, model=model, served=served, **options)
    return _call(client, model)


@pytest.mark.parametrize(
    ("model", "served", "bare_served"),
    [
        ("gpt-4o", "gpt-4o-2024-08-06", "gpt-4o-2024-08-06"),
        ("anthropic/claude-x", "claude-x-20250101", "claude-x-20250101"),
        ("anthropic/claude-x", "anthropic/claude-x-20250101", "claude-x-20250101"),
        ("gpt-4o", "gpt-4o", "gpt-4o"),
    ],
)
def test_alias_served_as_snapshot_records_configured_identity(
    monkeypatch, model, served, bare_served
):
    completion = _complete(monkeypatch, model=model, served=served)
    # Identity stays the configured id so qualification caches do not churn.
    assert completion.observed_model_revision == model
    assert completion.usage_json is not None
    assert json.loads(completion.usage_json)["served_model"] == bare_served


@pytest.mark.parametrize(
    ("model", "served"),
    [
        ("gpt-4o", "gpt-4o-mini"),
        ("gpt-4o", "gpt-4o-2024-08"),
        ("anthropic/claude-x", "claude-y-20250101"),
    ],
)
def test_substituted_model_is_refused_through_the_transport(monkeypatch, model, served):
    with pytest.raises(ValueError, match="served a different model"):
        _complete(monkeypatch, model=model, served=served)


def test_non_native_route_without_provider_stamp_still_raises(monkeypatch):
    with pytest.raises(ValueError, match="immutable observed provenance"):
        _complete(monkeypatch, model="gpt-4o", served="gpt-4o", hidden={})


def test_non_native_route_with_foreign_provider_still_raises(monkeypatch):
    with pytest.raises(ValueError, match="does not match the composed"):
        _complete(
            monkeypatch,
            model="gpt-4o",
            served="gpt-4o",
            hidden={"custom_llm_provider": "minimax"},
        )


def _native(monkeypatch, host: str) -> str:
    model = f"local-cli/{host}"
    # Own every input native selection reads, so a developer's own CLI setup
    # cannot fail a case before admission. The operator-configured native
    # target gate stays in force; satisfy it.
    monkeypatch.setenv(ENV_NATIVE_MODEL, model)
    for name in (
        "CLAUDE_SMART_USE_LOCAL_CLI",
        "CLAUDE_SMART_CLI_MODEL",
        "CLAUDE_SMART_CLI_PATH",
        "CLAUDE_SMART_CODEX_PATH",
        PLUGIN_OPENCODE_EXECUTABLE_ENV,
        CLI_TIMEOUT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return model


@pytest.mark.parametrize(
    ("host", "stamps", "expected"),
    [
        # claude-code with both stamps: a bridge backend's own labels.
        (
            "claude-code",
            {"reflexio_served_provider": "minimax", "reflexio_served_model": "M-3"},
            {"served_provider": "minimax", "served_model": "M-3"},
        ),
        # claude-code with a model only (no provider field in the stream).
        (
            "claude-code",
            {"reflexio_served_model": "claude-x-20250101"},
            {"served_model": "claude-x-20250101"},
        ),
        # codex and opencode stamp nothing.
        ("codex", {}, {}),
        ("opencode", {}, {}),
        # Blank stamps are missing, not values.
        (
            "claude-code",
            {"reflexio_served_provider": "  ", "reflexio_served_model": ""},
            {},
        ),
    ],
)
def test_native_cli_admits_whatever_the_cli_reports(
    monkeypatch, host, stamps, expected
):
    model = _native(monkeypatch, host)
    completion = _complete(
        monkeypatch,
        model=model,
        # Native routes echo the requested route here; it is never trusted.
        served=model,
        hidden={"reflexio_provider": "local-cli", **stamps},
    )
    assert completion.observed_model_revision == model
    assert completion.usage_json is not None
    usage = json.loads(completion.usage_json)
    assert {
        key: usage[key] for key in ("served_provider", "served_model") if key in usage
    } == expected


def test_native_cli_stamps_treat_blank_as_missing():
    assert _native_cli_served_stamps(provider=" minimax ", model=" M-3 ") == {
        "served_provider": "minimax",
        "served_model": "M-3",
    }
    assert _native_cli_served_stamps(provider="  ", model="") == {}
    assert _native_cli_served_stamps(provider=None, model=None) == {}
    assert _native_cli_served_stamps(provider=None, model="M-3") == {
        "served_model": "M-3"
    }


@pytest.mark.parametrize("native", [False, True])
def test_served_diagnostics_survive_a_response_without_usage(monkeypatch, native):
    if native:
        model = _native(monkeypatch, "claude-code")
        hidden = {"reflexio_provider": "local-cli", "reflexio_served_model": "M-3"}
        expected = "M-3"
    else:
        model, hidden, expected = "gpt-4o", None, "gpt-4o-2024-08-06"
    completion = _complete(
        monkeypatch,
        model=model,
        served=expected if not native else model,
        hidden=hidden,
        with_usage=False,
    )
    assert completion.usage_json is not None
    usage = json.loads(completion.usage_json)
    assert usage["served_model"] == expected
    assert usage["usage_observed_responses"] == 0


@pytest.mark.parametrize(
    ("served", "admitted"),
    [
        ("gpt-4o-2024-02-29", True),  # leap year
        ("gpt-4o-20240229", True),
        ("gpt-4o-2000-02-29", True),  # divisible by 400
        ("gpt-4o-2025-02-28", True),
        ("gpt-4o-2025-02-29", False),  # not a leap year
        ("gpt-4o-20250229", False),
        ("gpt-4o-1900-02-29", False),  # divisible by 100, not 400
        ("gpt-4o-2025-02-31", False),
        ("gpt-4o-20250231", False),
        ("gpt-4o-2025-04-31", False),
        ("gpt-4o-20250431", False),
        ("gpt-4o-2025-00-10", False),
        ("gpt-4o-2025-01-00", False),
        ("gpt-4o-0000-01-01", False),
        # Non-ASCII digits: Arabic-Indic and fullwidth, in both shapes.
        ("gpt-4o-\u0662\u0660\u0662\u0664-08-06", False),
        ("gpt-4o-\u0662\u0660\u0662\u0664\u0660\u0668\u0660\u0666", False),
        ("gpt-4o-\uff12\uff10\uff12\uff14-\uff10\uff18-\uff10\uff16", False),
        ("gpt-4o-\uff12\uff10\uff12\uff14\uff10\uff18\uff10\uff16", False),
    ],
)
def test_date_stamp_must_name_a_real_ascii_calendar_date(served, admitted):
    assert (
        _served_model_admitted(configured_bare="gpt-4o", observed_bare=served)
        is admitted
    )


def test_mixed_shape_stamp_is_refused():
    assert not _is_provider_date_stamp("-2024-0806")
    assert not _is_provider_date_stamp("-202408-06")


@pytest.mark.parametrize(
    ("configured", "served", "admitted"),
    [
        # A pinned snapshot needs an exact echo, in both stamp shapes.
        ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06", True),
        ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06-2024-11-20", False),
        ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06-20241120", False),
        ("claude-x-20250101", "claude-x-20250101", True),
        ("claude-x-20250101", "claude-x-20250101-20250601", False),
        ("claude-x-20250101", "claude-x-20250101-2025-06-01", False),
        # A trailing number that is not a real date is not a pin.
        ("model-2025-02-31", "model-2025-02-31-2025-03-01", True),
        ("model-99999999", "model-99999999-20250101", True),
    ],
)
def test_pinned_snapshot_requires_an_exact_echo(configured, served, admitted):
    assert (
        _served_model_admitted(configured_bare=configured, observed_bare=served)
        is admitted
    )


def test_native_launch_drift_after_composition_is_refused_before_dispatch(
    monkeypatch,
):
    model = _native(monkeypatch, "claude-code")
    dispatched: list[str] = []
    hidden = {"reflexio_provider": "local-cli", "reflexio_served_model": "M-3"}
    client = _compose(
        monkeypatch, model=model, served=model, hidden=hidden, dispatched=dispatched
    )
    # Unchanged launch policy: admitted.
    assert _call(client, model).observed_model_revision == model
    assert dispatched == [model]
    # A launch input the CLI host re-reads at dispatch changes underneath.
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    with pytest.raises(ValueError, match="transport configuration changed"):
        _call(client, model)
    assert dispatched == [model]


def test_non_native_transport_does_not_bind_native_launch_policy(monkeypatch):
    dispatched: list[str] = []
    client = _compose(
        monkeypatch, model="gpt-4o", served="gpt-4o", dispatched=dispatched
    )
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    assert _call(client, "gpt-4o").observed_model_revision == "gpt-4o"
    assert dispatched == ["gpt-4o"]
