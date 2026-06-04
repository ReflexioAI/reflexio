"""Unit tests for the OSS billing-signals helper."""

from reflexio.models.config_schema import APIKeyConfig, Config, OpenAIConfig
from reflexio.server.billing_signals import platform_llm_from_config


def test_platform_llm_true_when_no_api_key_config():
    """Config with no api_key_config → platform supplies the LLM."""
    assert platform_llm_from_config(Config(storage_config=None)) is True


def test_platform_llm_false_when_byo_openai_key():
    """Config with a populated OpenAI sub-config → customer BYO-LLM."""
    cfg = Config(
        storage_config=None,
        api_key_config=APIKeyConfig(openai=OpenAIConfig(api_key="sk-x")),
    )
    assert platform_llm_from_config(cfg) is False


def test_platform_llm_true_for_none_config():
    """None config (missing entirely) defaults to platform-supplied LLM."""
    assert platform_llm_from_config(None) is True


def test_platform_llm_true_for_empty_api_key_config():
    """APIKeyConfig with all providers None → no BYO key → platform LLM."""
    cfg = Config(storage_config=None, api_key_config=APIKeyConfig())
    assert platform_llm_from_config(cfg) is True
