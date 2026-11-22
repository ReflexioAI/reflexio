"""Served-model admission for the production open-world transport.

A provider resolves a floating alias to a dated snapshot, so ``gpt-4o`` may
answer as ``gpt-4o-2024-08-06``. That is the configured revision; a different
model (``gpt-4o-mini``, another family) is a substitution and is refused.
"""

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import json
import os

import litellm
import pytest

from reflexio.models.structured_output import StrictStructuredOutput
from reflexio.server.llm._native_model_ref import (
    CLI_TIMEOUT_ENV,
    ENV_NATIVE_MODEL,
)
from reflexio.server.llm._served_model_identity import (
    _is_provider_date_stamp,
)
from reflexio.server.llm._served_model_identity import (
    served_model_admitted as _served_model_admitted,
)
from reflexio.server.llm.litellm_client import (
    LiteLLMClient,
    LiteLLMConfig,
    ProviderRequestGuardError,
)
from reflexio.server.services.offline_tuner.open_world.analyst import (
    ChatMessage,
    _litellm_transport_identity,
    _LiteLLMOpenWorldModelClient,
    _native_cli_served_stamps,
    _native_pinned_model,
)
from tests.server.llm.test_native_launch_identity import (
    own_launch_environment,
    set_home,
    stub_executable,
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
    with pytest.raises(ProviderRequestGuardError, match="served a different model"):
        _complete(monkeypatch, model=model, served=served)


def test_non_native_route_without_provider_stamp_still_raises(monkeypatch):
    with pytest.raises(
        ProviderRequestGuardError, match="immutable observed provenance"
    ):
        _complete(monkeypatch, model="gpt-4o", served="gpt-4o", hidden={})


def test_non_native_route_with_foreign_provider_still_raises(monkeypatch):
    with pytest.raises(ProviderRequestGuardError, match="does not match the composed"):
        _complete(
            monkeypatch,
            model="gpt-4o",
            served="gpt-4o",
            hidden={"custom_llm_provider": "minimax"},
        )


def _native(monkeypatch, host: str, pin: str | None = None) -> str:
    model = f"local-cli/{host}" if pin is None else f"local-cli/{host}/{pin}"
    # Own every input native selection reads, so a developer's own CLI setup
    # cannot fail a case before admission. The operator-configured native
    # target gate stays in force; satisfy it.
    own_launch_environment(monkeypatch)
    monkeypatch.setenv(ENV_NATIVE_MODEL, model)
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


def _native_client_with_invalid_first_answer(monkeypatch, *, change_launch: bool):
    """Compose a native client whose first answer fails structured validation."""
    model = _native(monkeypatch, "claude-code")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test")
    dispatched: list[list[dict[str, str]]] = []

    def dispatch(self, params, deadline):
        del self, deadline
        dispatched.append(params["messages"])
        content = '{"answer": "ok"}'
        if len(dispatched) == 1:
            content = '{"unexpected": 1}'
            if change_launch:
                # The CLI host re-reads this at its next dispatch.
                monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
        response = litellm.ModelResponse(
            model=model,
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        response._hidden_params = {
            "reflexio_provider": "local-cli",
            "reflexio_served_model": "M-3",
        }
        return response

    monkeypatch.setattr(LiteLLMClient, "_run_completion", dispatch)
    client = _LiteLLMOpenWorldModelClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=512, fallback_models=[]))
    )
    return client, model, dispatched


def test_corrective_retry_after_unchanged_launch_is_admitted(monkeypatch):
    client, model, dispatched = _native_client_with_invalid_first_answer(
        monkeypatch, change_launch=False
    )
    assert _call(client, model).observed_model_revision == model
    # The invalid first answer really was followed by a second dispatch.
    assert len(dispatched) == 2


def test_launch_drift_before_a_retry_is_refused_without_dispatch(monkeypatch):
    client, model, dispatched = _native_client_with_invalid_first_answer(
        monkeypatch, change_launch=True
    )
    with pytest.raises(
        ProviderRequestGuardError, match="transport configuration changed"
    ):
        _call(client, model)
    assert len(dispatched) == 1


@pytest.mark.parametrize(
    ("reference", "pin"),
    [
        ("local-cli/claude-code", None),
        ("local-cli/codex", None),
        ("local-cli/opencode", None),
        ("local-cli/claude-code/claude-x", "claude-x"),
        ("local-cli/codex/synthetic-model", "synthetic-model"),
        # OpenCode's `#variant` is not part of the model id.
        ("local-cli/opencode/provider/model#high", "provider/model"),
        ("local-cli/opencode/#high", None),
        # Only OpenCode treats `#` as a variant separator.
        ("local-cli/claude-code/claude-x#y", "claude-x#y"),
    ],
)
def test_native_pinned_model(reference, pin):
    assert _native_pinned_model(reference) == pin


@pytest.mark.parametrize(
    ("host", "pin", "stamp", "admitted"),
    [
        ("claude-code", "claude-x", "claude-x", True),
        ("claude-code", "claude-x", "claude-x-20250101", True),
        ("claude-code", "claude-x", "claude-x-2025-01-01", True),
        ("claude-code", "claude-x", None, True),
        ("claude-code", "claude-x", "claude-y", False),
        ("claude-code", "claude-x", "claude-x-mini", False),
        # A pinned snapshot needs an exact echo.
        ("claude-code", "claude-x-20250101", "claude-x-20250101", True),
        ("claude-code", "claude-x-20250101", "claude-x", False),
        ("claude-code", "claude-x-20250101", "claude-x-20250101-20250601", False),
        # Hosts that stamp nothing are admitted on an explicit pin.
        ("codex", "synthetic-model", None, True),
        ("opencode", "provider/model#high", None, True),
        # A delegated default admits anything.
        ("claude-code", None, "anything-at-all", True),
    ],
)
def test_explicit_native_pin_refuses_a_contradicting_stamp(
    monkeypatch, host, pin, stamp, admitted
):
    model = _native(monkeypatch, host, pin)
    hidden = {"reflexio_provider": "local-cli"}
    if stamp is not None:
        hidden["reflexio_served_model"] = stamp
    if not admitted:
        with pytest.raises(ProviderRequestGuardError, match="served a different model"):
            _complete(monkeypatch, model=model, served=model, hidden=hidden)
        return
    completion = _complete(monkeypatch, model=model, served=model, hidden=hidden)
    assert completion.observed_model_revision == model


def _provider(monkeypatch, model: str) -> str:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-synthetic-test")
    client = LiteLLMClient(
        LiteLLMConfig(model=model, max_tokens=512, fallback_models=[])
    )
    return _litellm_transport_identity(client).provider


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        # Captured by running the same composition against origin/main's
        # analyst: binding the native launch policy must not move any
        # non-native persisted identity (no qualification cache churn).
        (
            "gpt-4o",
            "litellm:openai-default:"
            "34feb85be62f049b2c162705dbf7c70ef12d912630ea8bbf878f82924394fd8c",
        ),
        (
            "anthropic/claude-x-20250101",
            "litellm:anthropic:"
            "e5733a4bd8f9936419421c37c04d5d09a01ab4bcaafc06330b5332cc3350e2fa",
        ),
    ],
)
def test_non_native_identity_is_unchanged(monkeypatch, model, provider):
    for name in (
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_BASE",
        "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    # A native launch input must not leak into a non-native identity.
    monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    assert _provider(monkeypatch, model) == provider


def _claude_on_path(monkeypatch, directory) -> None:
    """Put an owned executable `claude` first and alone on PATH."""
    stub_executable(directory)
    monkeypatch.setenv("PATH", str(directory))


@pytest.mark.parametrize(
    ("change", "moves"),
    [
        ("cli-path-override", True),
        ("claude-config-dir", True),
        ("home", True),
        ("other-binary-on-path", True),
        ("cwd", False),
        ("path-same-binary", False),
        ("timeout", False),
    ],
)
def test_native_launch_identity_is_part_of_the_persisted_identity(
    monkeypatch, tmp_path, change, moves
):
    model = _native(monkeypatch, "claude-code")
    set_home(monkeypatch, tmp_path / "home")
    monkeypatch.chdir(tmp_path)
    _claude_on_path(monkeypatch, tmp_path / "bin")
    before = _provider(monkeypatch, model)
    assert _provider(monkeypatch, model) == before, "recomposition is stable"
    if change == "cli-path-override":
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", "/synthetic/other/claude")
    elif change == "claude-config-dir":
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "account-b"))
    elif change == "home":
        set_home(monkeypatch, tmp_path / "other-home")
    elif change == "other-binary-on-path":
        _claude_on_path(monkeypatch, tmp_path / "other-bin")
    elif change == "cwd":
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
    elif change == "path-same-binary":
        (tmp_path / "empty").mkdir()
        monkeypatch.setenv(
            "PATH", os.pathsep.join([str(tmp_path / "empty"), str(tmp_path / "bin")])
        )
    else:
        monkeypatch.setenv(CLI_TIMEOUT_ENV, "30")
    assert (_provider(monkeypatch, model) != before) is moves


def test_a_long_running_client_keeps_answering_across_an_inert_env_change(
    monkeypatch, tmp_path
):
    # The in-process per-dispatch check compares the same narrow identity, so
    # a daemon whose PATH or cwd changes but still resolves the same binary
    # keeps answering instead of refusing every call.
    model = _native(monkeypatch, "claude-code")
    monkeypatch.chdir(tmp_path)
    _claude_on_path(monkeypatch, tmp_path / "bin")
    dispatched: list[str] = []
    hidden = {"reflexio_provider": "local-cli"}
    client = _compose(
        monkeypatch, model=model, served=model, hidden=hidden, dispatched=dispatched
    )
    assert _call(client, model).observed_model_revision == model
    (tmp_path / "empty").mkdir()
    monkeypatch.setenv(
        "PATH", os.pathsep.join([str(tmp_path / "empty"), str(tmp_path / "bin")])
    )
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path / "elsewhere")
    monkeypatch.setenv(CLI_TIMEOUT_ENV, "30")
    assert _call(client, model).observed_model_revision == model
    assert len(dispatched) == 2


@pytest.mark.parametrize(
    ("pin", "stamp", "admitted"),
    [
        ("sonnet", "claude-sonnet-4-5-20250929", True),
        ("Sonnet", "claude-sonnet-4-5-20250929", True),
        ("OPUS", "claude-opus-4-1-20250805", True),
        ("haiku", "claude-haiku-4-5", True),
        ("sonnet", None, True),
        # Another family, or not a Claude id at all.
        ("sonnet", "claude-opus-4-1-20250805", False),
        ("opus", "claude-sonnet-4-5-20250929", False),
        ("haiku", "claude-sonnet-4-5", False),
        ("sonnet", "sonnet", False),
        ("sonnet", "gpt-4o", False),
        ("sonnet", "claude-sonnetx-1", False),
        # An explicit `/default` is a literal pin, not delegation.
        ("default", "claude-haiku-4-5", False),
        # A full-id pin keeps the strict rule.
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929", True),
        ("claude-sonnet-4-5", "claude-sonnet-4-6", False),
    ],
)
def test_claude_code_alias_pin_matches_its_family(monkeypatch, pin, stamp, admitted):
    model = _native(monkeypatch, "claude-code", pin)
    hidden = {"reflexio_provider": "local-cli"}
    if stamp is not None:
        hidden["reflexio_served_model"] = stamp
    if not admitted:
        with pytest.raises(ProviderRequestGuardError, match="served a different model"):
            _complete(monkeypatch, model=model, served=model, hidden=hidden)
        return
    completion = _complete(monkeypatch, model=model, served=model, hidden=hidden)
    assert completion.observed_model_revision == model


def test_alias_handling_is_claude_code_only(monkeypatch):
    from reflexio.server.llm._served_model_identity import (
        native_served_model_admitted as _native_served_model_admitted,
    )

    own_launch_environment(monkeypatch)

    assert _native_served_model_admitted(
        host="claude-code", pin="sonnet", served="claude-sonnet-4-5"
    )
    assert not _native_served_model_admitted(
        host="codex", pin="sonnet", served="claude-sonnet-4-5"
    )
    assert not _native_served_model_admitted(
        host="opencode", pin="sonnet", served="claude-sonnet-4-5"
    )


def _client_answering(monkeypatch, model: str, answers):
    """Compose a client whose physical responses are ``answers``, in order.

    Each answer is ``(served, hidden, content, with_usage)``.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-synthetic-test")
    dispatched: list[str] = []

    def dispatch(self, params, deadline):
        del self, deadline
        served, hidden, content, with_usage = answers[len(dispatched)]
        dispatched.append(params["model"])
        response = litellm.ModelResponse(
            model=served,
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            usage={"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        )
        response._hidden_params = dict(hidden)
        if not with_usage:
            setattr(response, "usage", None)  # noqa: B010 - untyped on ModelResponse
        return response

    monkeypatch.setattr(LiteLLMClient, "_run_completion", dispatch)
    client = _LiteLLMOpenWorldModelClient(
        LiteLLMClient(LiteLLMConfig(model=model, max_tokens=512, fallback_models=[]))
    )
    return client, dispatched


_INVALID = '{"unexpected": 1}'
_VALID = '{"answer": "ok"}'


@pytest.mark.parametrize("with_usage", [True, False])
def test_a_contradicting_native_stamp_is_refused_before_any_retry(
    monkeypatch, with_usage
):
    model = _native(monkeypatch, "claude-code", "claude-x")
    native = {"reflexio_provider": "local-cli"}
    client, dispatched = _client_answering(
        monkeypatch,
        model,
        [
            # Contradicts the pin AND fails validation, which would otherwise
            # earn a corrective turn ...
            (
                model,
                {**native, "reflexio_served_model": "claude-y"},
                _INVALID,
                with_usage,
            ),
            # ... whose matching answer must not be what gets accepted.
            (model, {**native, "reflexio_served_model": "claude-x"}, _VALID, True),
        ],
    )
    with pytest.raises(ProviderRequestGuardError, match="served a different model"):
        _call(client, model)
    assert len(dispatched) == 1


def test_a_substituted_http_model_is_refused_before_any_retry(monkeypatch):
    hidden = {"custom_llm_provider": "openai"}
    client, dispatched = _client_answering(
        monkeypatch,
        "gpt-4o",
        [
            ("gpt-4o-mini", hidden, _INVALID, False),
            ("gpt-4o", hidden, _VALID, True),
        ],
    )
    with pytest.raises(ProviderRequestGuardError, match="served a different model"):
        _call(client, "gpt-4o")
    assert len(dispatched) == 1


def test_an_admitted_invalid_answer_still_earns_its_retry(monkeypatch):
    model = _native(monkeypatch, "claude-code", "claude-x")
    native = {"reflexio_provider": "local-cli", "reflexio_served_model": "claude-x"}
    client, dispatched = _client_answering(
        monkeypatch,
        model,
        [(model, native, _INVALID, False), (model, native, _VALID, True)],
    )
    assert _call(client, model).observed_model_revision == model
    assert len(dispatched) == 2


@pytest.mark.parametrize(
    ("host", "pin", "stamp", "admitted"),
    [
        # `[1m]` selects the 1M-context variant of the same model.
        ("claude-code", "sonnet[1m]", "claude-sonnet-4-5-20250929", True),
        ("claude-code", "SONNET[1M]", "claude-sonnet-4-5-20250929", True),
        ("claude-code", "opus[1m]", "claude-opus-4-1-20250805", True),
        ("claude-code", "sonnet[1m]", "claude-opus-4-1-20250805", False),
        ("claude-code", "claude-sonnet-4-6[1m]", "claude-sonnet-4-6", True),
        ("claude-code", "claude-sonnet-4-6[1m]", "claude-sonnet-4-5", False),
        # `best` uses the same model as `opus`.
        ("claude-code", "best", "claude-opus-4-1-20250805", True),
        ("claude-code", "best", "claude-sonnet-4-5", False),
        ("claude-code", "fable", "claude-fable-5-2", True),
        ("claude-code", "fable", "claude-opus-5-5", False),
        # `opusplan` runs opus in plan mode and sonnet otherwise.
        ("claude-code", "opusplan", "claude-opus-4-1", True),
        ("claude-code", "opusplan", "claude-sonnet-4-5", True),
        ("claude-code", "opusplan[1m]", "claude-sonnet-4-5", True),
        ("claude-code", "opusplan", "claude-haiku-4-5", False),
        # An explicit `/default` is the literal model ID `default`, held to
        # the explicit-pin rule; only a host-only reference delegates.
        ("claude-code", "default", "claude-haiku-4-5", False),
        ("claude-code", "DEFAULT", "claude-haiku-4-5", False),
        ("claude-code", "default[1m]", "claude-sonnet-4-5", False),
        ("claude-code", "default", "default", True),
        # Only Claude Code has these aliases.
        ("codex", "sonnet[1m]", "claude-sonnet-4-5", False),
        ("codex", "default", "synthetic-model", False),
    ],
)
def test_claude_code_documented_aliases(monkeypatch, host, pin, stamp, admitted):
    from reflexio.server.llm._served_model_identity import (
        native_served_model_admitted as _native_served_model_admitted,
    )

    own_launch_environment(monkeypatch)
    assert _native_served_model_admitted(host=host, pin=pin, served=stamp) is admitted


@pytest.mark.parametrize(
    ("pin", "stamp", "admitted"),
    [
        # Bedrock (with or without a cross-region profile) and Vertex ids.
        ("sonnet", "us.anthropic.claude-sonnet-4-6", True),
        ("sonnet", "global.anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("sonnet", "anthropic.claude-sonnet-4-5-20250929-v1:0", True),
        ("sonnet", "claude-sonnet-4-6@20250929", True),
        ("opusplan", "eu.anthropic.claude-opus-4-1", True),
        # Qualification never relaxes the family.
        ("sonnet", "us.anthropic.claude-opus-4-1", False),
        ("haiku", "anthropic.claude-sonnet-4-5", False),
        ("sonnet", "claude-opus-4-1@20250805", False),
        ("sonnet", "us.anthropic.claude-sonnetx-1", False),
        ("sonnet", "us.notanthropic.claude-sonnet-4-6", False),
        ("sonnet", "synthetic.claude-sonnet-4-6", False),
    ],
)
def test_claude_code_alias_admits_cloud_qualified_ids(
    monkeypatch, pin, stamp, admitted
):
    from reflexio.server.llm._served_model_identity import (
        native_served_model_admitted as _native_served_model_admitted,
    )

    own_launch_environment(monkeypatch)
    assert (
        _native_served_model_admitted(host="claude-code", pin=pin, served=stamp)
        is admitted
    )


@pytest.mark.parametrize(
    ("family", "pin", "configured", "stamp", "admitted"),
    [
        # The remapped id is exactly what the alias launches.
        ("SONNET", "sonnet", "synthetic-model-a", "synthetic-model-a", True),
        ("SONNET", "sonnet", "synthetic-model-a", "synthetic-model-a-20250929", True),
        ("SONNET", "sonnet[1m]", "synthetic-model-a[1m]", "synthetic-model-a", True),
        (
            "OPUS",
            "best",
            "us.anthropic.claude-opus-4-1",
            "us.anthropic.claude-opus-4-1",
            True,
        ),
        # Once remapped, the family's other ids are not that configuration.
        ("SONNET", "sonnet", "synthetic-model-a", "claude-sonnet-4-5", False),
        ("SONNET", "sonnet", "synthetic-model-a", "synthetic-model-b", False),
        ("HAIKU", "haiku", "claude-haiku-4-5", "us.anthropic.claude-haiku-4-5", False),
        # A remap of one family leaves the others on their family rule.
        ("SONNET", "opus", "synthetic-model-a", "claude-opus-4-1", True),
        ("SONNET", "opus", "synthetic-model-a", "synthetic-model-a", False),
        # opusplan: each family its own rule.
        ("SONNET", "opusplan", "synthetic-model-a", "synthetic-model-a", True),
        ("SONNET", "opusplan", "synthetic-model-a", "claude-opus-4-1", True),
        ("SONNET", "opusplan", "synthetic-model-a", "claude-sonnet-4-5", False),
    ],
)
def test_a_remapped_alias_admits_exactly_the_configured_id(
    monkeypatch, family, pin, configured, stamp, admitted
):
    from reflexio.server.llm._served_model_identity import (
        native_served_model_admitted as _native_served_model_admitted,
    )

    own_launch_environment(monkeypatch)
    monkeypatch.setenv(f"ANTHROPIC_DEFAULT_{family}_MODEL", configured)
    assert (
        _native_served_model_admitted(host="claude-code", pin=pin, served=stamp)
        is admitted
    )


def test_a_blank_remap_keeps_the_family_rule(monkeypatch):
    from reflexio.server.llm._served_model_identity import (
        native_served_model_admitted as _native_served_model_admitted,
    )

    own_launch_environment(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_DEFAULT_SONNET_MODEL", "  ")
    assert _native_served_model_admitted(
        host="claude-code", pin="sonnet", served="claude-sonnet-4-5"
    )
