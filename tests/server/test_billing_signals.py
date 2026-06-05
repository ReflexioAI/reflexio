"""Unit tests for the OSS billing-signals helper."""

from reflexio.models.config_schema import APIKeyConfig, Config, OpenAIConfig
from reflexio.server.billing_signals import count_input_tokens, platform_llm_from_config


def test_count_input_tokens_empty_string_is_zero():
    """Empty input short-circuits to 0 without invoking the encoder."""
    assert count_input_tokens("") == 0


def test_count_input_tokens_known_string_is_stable():
    """A fixed string yields a stable cl100k_base count (regression guard vs. encoding drift)."""
    # "hello world" tokenizes to exactly 2 tokens under cl100k_base. Asserting the
    # literal here makes any silent change to the canonical encoding fail loudly.
    assert count_input_tokens("hello world") == 2


def test_count_input_tokens_handles_special_token_text_without_raising():
    """Text containing special-token markup is tokenized literally, not rejected.

    ``disallowed_special=()`` means a string like ``<|endoftext|>`` is encoded as
    ordinary text rather than raising ``ValueError``.
    """
    count = count_input_tokens("hello <|endoftext|> world")
    assert count == 8


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
