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
            expected = getattr(_PROVIDER_DEFAULTS["openai"], role.value)
            assert result == expected, f"Mismatch for {role}"

    def test_auto_detect_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        # Generation should use anthropic
        result = resolve_model_name(ModelRole.GENERATION)
        assert result == _PROVIDER_DEFAULTS["anthropic"].generation
        # Embedding should fail (no embedding-capable provider)
        with pytest.raises(RuntimeError, match="embedding-capable"):
            resolve_model_name(ModelRole.EMBEDDING)

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

    def test_embedding_cross_provider_anthropic_primary(self) -> None:
        """When only Anthropic key is in APIKeyConfig, embedding falls back to OpenAI via env."""
        from reflexio.models.config_schema import APIKeyConfig

        config = APIKeyConfig(
            anthropic=AnthropicConfig(api_key="ant-test"),
            openai=OpenAIConfig(api_key="sk-test"),
        )
        result = resolve_model_name(
            ModelRole.EMBEDDING,
            api_key_config=config,
        )
        assert result == _PROVIDER_DEFAULTS["openai"].embedding

    def test_gemini_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
        result = resolve_model_name(ModelRole.EMBEDDING)
        assert result == _PROVIDER_DEFAULTS["gemini"].embedding


# ---------------------------------------------------------------------------
# validate_llm_availability
# ---------------------------------------------------------------------------


class TestValidateLlmAvailability:
    def test_no_keys_raises(self) -> None:
        with pytest.raises(RuntimeError, match="No LLM provider available"):
            validate_llm_availability()

    def test_no_embedding_provider_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        with pytest.raises(RuntimeError, match="embedding-capable"):
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
# EXTRACTION_AGENT and SEARCH_AGENT roles
# ---------------------------------------------------------------------------


class TestAgenticV2Roles:
    def test_extraction_agent_role_exists(self) -> None:
        assert ModelRole.EXTRACTION_AGENT.value == "extraction_agent"

    def test_search_agent_role_exists(self) -> None:
        assert ModelRole.SEARCH_AGENT.value == "search_agent"

    def test_anthropic_defaults_map_to_sonnet(self) -> None:
        anthropic = _PROVIDER_DEFAULTS["anthropic"]
        assert anthropic.extraction_agent is not None
        assert "sonnet" in anthropic.extraction_agent.lower()
        assert anthropic.search_agent is not None
        assert "sonnet" in anthropic.search_agent.lower()

    def test_openai_defaults_map_to_gpt5_mini(self) -> None:
        openai = _PROVIDER_DEFAULTS["openai"]
        assert openai.extraction_agent == "gpt-5-mini"
        assert openai.search_agent == "gpt-5-mini"

    def test_claude_code_defaults_cover_new_roles(self) -> None:
        cc = _PROVIDER_DEFAULTS["claude-code"]
        assert cc.extraction_agent == "claude-code/default"
        assert cc.search_agent == "claude-code/default"

    def test_unpopulated_providers_default_to_none(self) -> None:
        """Providers that haven't opted into agentic-v2 fall through to next priority provider."""
        local = _PROVIDER_DEFAULTS["local"]
        assert local.extraction_agent is None
        assert local.search_agent is None

    def test_resolve_extraction_agent_with_anthropic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-test")
        name = resolve_model_name(role=ModelRole.EXTRACTION_AGENT)
        assert "sonnet" in name.lower()

    def test_resolve_search_agent_with_openai(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        name = resolve_model_name(role=ModelRole.SEARCH_AGENT)
        assert name == "gpt-5-mini"
