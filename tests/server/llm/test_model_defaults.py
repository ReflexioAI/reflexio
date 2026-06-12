"""Tests for model_defaults module — auto-detection and resolution of LLM models."""

from __future__ import annotations

from typing import Any

import pytest

from reflexio.models.config_schema import (
    AnthropicConfig,
    OpenAIConfig,
)
from reflexio.server.llm.model_defaults import (
    _PROVIDER_DEFAULTS,
    ModelRole,
    detect_available_providers,
    resolve_model_name,
    validate_llm_availability,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all LLM API key env vars to isolate each test."""
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "DEEPSEEK_API_KEY",
        "OPENROUTER_API_KEY",
        "MINIMAX_API_KEY",
        "DASHSCOPE_API_KEY",
        "XAI_API_KEY",
        "MOONSHOT_API_KEY",
        "ZAI_API_KEY",
        "CLAUDE_SMART_USE_LOCAL_CLI",
        "CLAUDE_SMART_CLI_PATH",
        "CLAUDE_SMART_CLI_TIMEOUT",
        "CLAUDE_SMART_USE_LOCAL_EMBEDDING",
    ]:
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# detect_available_providers
# ---------------------------------------------------------------------------


class TestDetectAvailableProviders:
    def test_no_keys(self) -> None:
        assert detect_available_providers() == []

    def test_single_provider_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert detect_available_providers() == ["openai"]

    def test_multiple_providers_priority_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        providers = detect_available_providers()
        assert providers[0] == "deepseek"
        assert "openai" in providers

    def test_empty_env_var_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "")
        assert detect_available_providers() == []

    def test_api_key_config_detected(self) -> None:
        from reflexio.models.config_schema import APIKeyConfig

        config = APIKeyConfig(anthropic=AnthropicConfig(api_key="ant-test"))
        providers = detect_available_providers(config)
        assert providers == ["anthropic"]

    def test_api_key_config_plus_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from reflexio.models.config_schema import APIKeyConfig

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        config = APIKeyConfig(anthropic=AnthropicConfig(api_key="ant-test"))
        providers = detect_available_providers(config)
        assert providers[0] == "anthropic"
        assert "openai" in providers

    def test_claude_code_needs_env_and_cli(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """claude-code requires both the env var AND the `claude` binary."""
        from reflexio.server.llm.providers import claude_code_provider

        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setattr(claude_code_provider.shutil, "which", lambda _: None)
        assert "claude-code" not in detect_available_providers()

        monkeypatch.setattr(
            claude_code_provider.shutil, "which", lambda _: "/usr/local/bin/claude"
        )
        assert detect_available_providers() == ["claude-code"]

    def test_claude_code_not_detected_without_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without CLAUDE_SMART_USE_LOCAL_CLI=1, the CLI alone is not enough."""
        from reflexio.server.llm.providers import claude_code_provider

        monkeypatch.setattr(
            claude_code_provider.shutil, "which", lambda _: "/usr/local/bin/claude"
        )
        assert detect_available_providers() == []

    def test_claude_code_takes_priority_over_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reflexio.server.llm.providers import claude_code_provider

        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(
            claude_code_provider.shutil, "which", lambda _: "/usr/local/bin/claude"
        )
        providers = detect_available_providers()
        assert providers[0] == "claude-code"
        assert "anthropic" in providers

    def test_claude_code_respects_cli_path_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
    ) -> None:
        """CLAUDE_SMART_CLI_PATH should be honoured when `claude` is not on PATH."""
        from reflexio.server.llm.providers import claude_code_provider

        fake_cli = tmp_path / "claude"
        fake_cli.write_text("#!/bin/sh\n")
        fake_cli.chmod(0o755)
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_CLI", "1")
        monkeypatch.setenv("CLAUDE_SMART_CLI_PATH", str(fake_cli))
        monkeypatch.setattr(claude_code_provider.shutil, "which", lambda _: None)
        assert "claude-code" in detect_available_providers()


# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------


class TestResolveModelName:
    def test_config_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value="minimax/MiniMax-M2.5",
            config_override="custom/my-model",
        )
        assert result == "custom/my-model"

    def test_site_var_wins_over_auto_detect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value="minimax/MiniMax-M2.5",
        )
        assert result == "minimax/MiniMax-M2.5"

    def test_empty_site_var_falls_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = resolve_model_name(
            ModelRole.GENERATION,
            site_var_value="",
        )
        assert result == _PROVIDER_DEFAULTS["openai"].generation

    def test_none_site_var_falls_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = resolve_model_name(ModelRole.GENERATION)
        assert result == _PROVIDER_DEFAULTS["openai"].generation

    def test_auto_detect_openai(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        for role in ModelRole:
            result = resolve_model_name(role)
            expected = (
                _PROVIDER_DEFAULTS["local"].embedding
                if role == ModelRole.EMBEDDING
                else getattr(_PROVIDER_DEFAULTS["openai"], role.value)
            )
            assert result == expected, f"Mismatch for {role}"

    def test_auto_detect_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        # Generation should use anthropic
        result = resolve_model_name(ModelRole.GENERATION)
        assert result == _PROVIDER_DEFAULTS["anthropic"].generation
        # Embedding falls back to local when chromadb is importable.
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        assert (
            resolve_model_name(ModelRole.EMBEDDING)
            == _PROVIDER_DEFAULTS["local"].embedding
        )

    def test_no_keys_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            resolve_model_name(ModelRole.GENERATION)

    def test_embedding_cross_provider_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic primary for generation, OpenAI for embeddings."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        # Anthropic > OpenAI in priority, so anthropic is primary for generation
        result = resolve_model_name(ModelRole.GENERATION)
        assert result == _PROVIDER_DEFAULTS["anthropic"].generation

    def test_embedding_config_default_ignores_cloud_api_keys(self) -> None:
        """Embedding defaults to the OSS local model unless config overrides it."""
        from reflexio.models.config_schema import APIKeyConfig

        config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="ant-test"),
            openai=OpenAIConfig(api_key="sk-test"),
        )
        result = resolve_model_name(
            ModelRole.EMBEDDING,
            api_key_config=config,
        )
        assert result == _PROVIDER_DEFAULTS["local"].embedding

    def test_gemini_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == _PROVIDER_DEFAULTS["local"].embedding

    def test_embedding_fallback_to_local_when_no_cloud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic key + chromadb available → fall back to local embedder."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == _PROVIDER_DEFAULTS["local"].embedding

    def test_embedding_fallback_skipped_when_cloud_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Local default still wins when cloud embedding keys are available."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == _PROVIDER_DEFAULTS["local"].embedding

    def test_embedding_explicit_opt_in_beats_cloud(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLAUDE_SMART_USE_LOCAL_EMBEDDING=1 forces local even with cloud keys."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == _PROVIDER_DEFAULTS["local"].embedding

    def test_embedding_default_does_not_probe_chromadb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Saved config owns overrides; default model selection does not inspect imports."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert (
            resolve_model_name(ModelRole.EMBEDDING)
            == _PROVIDER_DEFAULTS["local"].embedding
        )


# ---------------------------------------------------------------------------
# validate_llm_availability
# ---------------------------------------------------------------------------


class TestValidateLlmAvailability:
    def test_no_keys_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            validate_llm_availability()

    def test_no_embedding_provider_falls_back_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic key + chromadb importable → local fallback, no raise."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        validate_llm_availability()  # should not raise

    def test_no_embedding_provider_no_chromadb_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Anthropic key + chromadb missing → raise with install hint."""
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(RuntimeError, match="chromadb"):
            validate_llm_availability()

    def test_openai_only_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_llm_availability()  # should not raise

    def test_anthropic_plus_openai_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        validate_llm_availability()

    def test_api_key_config_passes(self) -> None:
        from reflexio.models.config_schema import APIKeyConfig

        config = APIKeyConfig(openai=OpenAIConfig(api_key="sk-test"))
        validate_llm_availability(config)

    def test_gemini_only_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        validate_llm_availability()

    def test_embedding_only_provider_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``providers == ["local"]`` (embedder-only, no LLM key) must raise.

        The local ONNX embedder satisfies the embedding role but has no
        generation model, so any extraction call would crash inside the
        role resolver. Failing fast at startup keeps the user out of
        that footgun.
        """
        from reflexio.server.llm.providers import local_embedding_provider as lep

        # No LLM env vars; opt into the local embedder so ``local`` shows
        # up in ``detect_available_providers`` results.
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        with pytest.raises(
            RuntimeError, match="No generation-capable LLM provider available"
        ):
            validate_llm_availability()


# ---------------------------------------------------------------------------
# All providers have defaults defined
# ---------------------------------------------------------------------------


class TestProviderDefaults:
    def test_all_priority_providers_have_defaults(self) -> None:
        from reflexio.server.llm.model_defaults import _PROVIDER_PRIORITY

        for provider in _PROVIDER_PRIORITY:
            assert provider in _PROVIDER_DEFAULTS, f"Missing defaults for {provider}"

    def test_all_roles_have_values(self) -> None:
        """Every provider must support either generation+evaluation or embedding.

        Embedding-only providers (e.g. ``local``) have None for the
        generation/evaluation/should_run/pre_retrieval slots; the role
        resolver falls through to the next provider in priority order.
        Generation-only providers (e.g. ``claude-code``) have None for
        embedding and fall back to an embedding-capable provider.
        """
        for provider, defaults in _PROVIDER_DEFAULTS.items():
            has_generation = defaults.generation is not None
            has_embedding = defaults.embedding is not None
            assert has_generation or has_embedding, (
                f"{provider} supports neither generation nor embedding"
            )
            if has_generation:
                # A provider that advertises generation must advertise
                # every generation-family role.
                for role in (
                    ModelRole.GENERATION,
                    ModelRole.EVALUATION,
                    ModelRole.SHOULD_RUN,
                    ModelRole.PRE_RETRIEVAL,
                ):
                    value = getattr(defaults, role.value)
                    assert value, f"{provider}.{role.value} is empty"


# ---------------------------------------------------------------------------
# EXTRACTION_AGENT role (drives the always-on resumable extraction loop)
# ---------------------------------------------------------------------------


class TestExtractionAgentRole:
    def test_extraction_agent_role_exists(self) -> None:
        assert ModelRole.EXTRACTION_AGENT.value == "extraction_agent"

    def test_anthropic_defaults_map_to_sonnet(self) -> None:
        anthropic = _PROVIDER_DEFAULTS["anthropic"]
        assert anthropic.extraction_agent is not None
        assert "sonnet" in anthropic.extraction_agent.lower()

    def test_openai_defaults_map_to_gpt5(self) -> None:
        openai = _PROVIDER_DEFAULTS["openai"]
        assert openai.extraction_agent == "gpt-5.5"

    def test_claude_code_defaults_cover_extraction_agent(self) -> None:
        cc = _PROVIDER_DEFAULTS["claude-code"]
        assert cc.extraction_agent == "claude-code/default"

    def test_unpopulated_providers_default_to_none(self) -> None:
        """Providers without an extraction_agent fall through to next priority provider."""
        local = _PROVIDER_DEFAULTS["local"]
        assert local.extraction_agent is None

    def test_resolve_extraction_agent_with_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        name = resolve_model_name(role=ModelRole.EXTRACTION_AGENT)
        assert "sonnet" in name.lower()


class TestMinimaxExtractionAgentRole:
    """MiniMax must expose extraction_agent so the resumable extraction loop
    works for MiniMax-only configurations.

    Surfaced by an e2e run where publish on a MiniMax-only VPS emitted
    'No provider in [\'minimax\'] supports role=extraction_agent' and
    silently skipped profile creation.
    """

    def test_minimax_has_extraction_agent(self):
        from reflexio.server.llm.model_defaults import _PROVIDER_DEFAULTS

        assert _PROVIDER_DEFAULTS["minimax"].extraction_agent is not None
        assert _PROVIDER_DEFAULTS["minimax"].extraction_agent.startswith("minimax/")

    def test_minimax_only_resolves_extraction_agent(self):
        """Auto-detect must return a MiniMax model when only MINIMAX_API_KEY
        is configured and the role is extraction_agent."""
        from reflexio.server.llm.model_defaults import (
            ModelRole,
            _auto_detect_model,
        )

        result = _auto_detect_model(ModelRole.EXTRACTION_AGENT, providers=["minimax"])
        assert result == "minimax/MiniMax-M3"


# ---------------------------------------------------------------------------
# Regression: MiniMax-only env must NOT silently resolve to OpenAI defaults
# ---------------------------------------------------------------------------


class TestMinimaxOnlyEnvRegression:
    """Reproduces the e2e regression where a fresh setup-init with only
    MINIMAX_API_KEY in env saw the extractor pick gpt-5.4-mini at runtime.

    The contract this class locks in: when the only LLM env var in scope
    is MINIMAX_API_KEY, every generation-family role (the slots that
    profile_extractor / playbook_extractor / agent_success_evaluator
    resolve at construction time) must resolve to ``minimax/MiniMax-M3``,
    not to any OpenAI default. This is the runtime expectation the
    setup-init wizard documents to MiniMax-only users.

    Failure mode this guards against: a future provider-priority edit
    that drops MiniMax below OpenAI in ``_PROVIDER_PRIORITY``, or a
    rename of ``MINIMAX_API_KEY`` in ``_ENV_TO_PROVIDER`` that silently
    falls through to the empty-key OpenAI default.
    """

    def test_generation_role_resolves_to_minimax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``default_generation_model_name`` must be the MiniMax model."""
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        result = resolve_model_name(ModelRole.GENERATION)
        assert result == "minimax/MiniMax-M3", (
            f"Expected minimax/MiniMax-M3, got {result!r}. If this fails, "
            "MiniMax-only users will hit OpenAI auth errors at extraction time."
        )

    def test_should_run_role_resolves_to_minimax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``should_run_model_name`` must also resolve to the MiniMax model."""
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        result = resolve_model_name(ModelRole.SHOULD_RUN)
        assert result == "minimax/MiniMax-M3"

    def test_all_generation_roles_resolve_to_minimax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every non-embedding role must resolve to a MiniMax model.

        Catches regressions where the ``extraction_agent`` role gets
        re-introduced without MiniMax coverage — the exact bug PR #51 fixed.
        """
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        for role in (
            ModelRole.GENERATION,
            ModelRole.EVALUATION,
            ModelRole.SHOULD_RUN,
            ModelRole.PRE_RETRIEVAL,
            ModelRole.EXTRACTION_AGENT,
        ):
            result = resolve_model_name(role)
            assert result.startswith("minimax/"), (
                f"Role {role.value} resolved to {result!r}, expected a "
                "minimax/ model. MiniMax-only users would hit a "
                "provider-mismatch error for this role."
            )

    def test_minimax_with_empty_openai_key_still_resolves_to_minimax(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty ``OPENAI_API_KEY=`` placeholder line must NOT promote OpenAI.

        A bundled ``.env`` template may ship ``OPENAI_API_KEY=`` (no
        value); ``load_dotenv`` interprets this as
        ``os.environ['OPENAI_API_KEY'] = ''``. ``detect_available_providers``
        relies on truthiness, so an empty string must be treated as
        "key not set" and the OpenAI provider must NOT appear in the
        priority list.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "")
        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        providers = detect_available_providers()
        assert "openai" not in providers, (
            f"Empty OPENAI_API_KEY must not promote OpenAI; got {providers}"
        )
        assert resolve_model_name(ModelRole.GENERATION) == "minimax/MiniMax-M3"

    def test_minimax_with_chromadb_resolves_embedding_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mirror of the e2e Mode A path: MiniMax + chromadb → local embedder.

        MiniMax has no embedding endpoint, so the embedding role must
        fall through to the local ONNX embedder via Path 3 of
        ``_auto_detect_model``. This is the configuration the e2e
        reproducer uses.
        """
        from reflexio.server.llm.providers import local_embedding_provider as lep

        monkeypatch.setenv("MINIMAX_API_KEY", "mm-test")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: object())
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == "local/minilm-l6-v2"
