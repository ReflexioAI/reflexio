"""Unit tests for the opt-in determinism cache in LiteLLMClient.

The cache is enabled via REFLEXIO_LLM_CACHE_DIR (+ REFLEXIO_LLM_SEED) and
makes benchmark extraction reruns byte-identical even on reasoning models
that ignore `temperature` / `seed`. These tests pin the cache-key stability
contract, the session-header normalization, and the roundtrip behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexio.server.llm.litellm_client import (
    _llm_cache_key,
    _read_llm_cache,
    _write_llm_cache,
)


def _base_params() -> dict:
    """Build a minimal completion-params dict for cache tests.

    Returns:
        dict: A params dict with stable defaults suitable for hashing.
    """
    return {
        "model": "openai/gpt-5-mini",
        "messages": [
            {"role": "system", "content": "you are a solution archivist"},
            {"role": "user", "content": "trajectory body here"},
        ],
        "temperature": 0.0,
        "timeout": 120,
        "num_retries": 0,
        "api_key": "sk-test-secret",
    }


def test_llm_cache_key_stable_under_param_reorder() -> None:
    """Two params dicts with identical content but different key order
    must hash to the same cache key.

    If this regresses, cache hits become order-sensitive and the whole
    determinism story collapses — each rerun would silently miss.
    """
    params_a = _base_params()
    params_b = {
        "api_key": params_a["api_key"],
        "num_retries": params_a["num_retries"],
        "timeout": params_a["timeout"],
        "temperature": params_a["temperature"],
        "messages": params_a["messages"],
        "model": params_a["model"],
    }
    assert _llm_cache_key(params_a) == _llm_cache_key(params_b)


def test_llm_cache_key_ignores_volatile_fields() -> None:
    """`api_key`, `timeout`, `num_retries`, `api_base`, `api_version`, and
    `metadata` must not influence the cache key — they change per run but
    don't affect the LLM's response."""
    base = _base_params()
    variant = {
        **base,
        "api_key": "sk-different",
        "api_base": "https://other.example",
        "api_version": "2024-10-01",
        "timeout": 300,
        "num_retries": 3,
        "metadata": {"trace_id": "abc"},
    }
    assert _llm_cache_key(base) == _llm_cache_key(variant)


def test_llm_cache_key_normalizes_session_header() -> None:
    """Two messages differing only in their `=== Session: ... ===` header
    must hash identically — publish_interaction labels each run with a
    fresh session ID, so without normalization every rerun would miss."""
    base = _base_params()
    a = {
        **base,
        "messages": [
            {"role": "system", "content": "you are a solution archivist"},
            {
                "role": "user",
                "content": "=== Session: abc123 ===\nhere is the trajectory",
            },
        ],
    }
    b = {
        **base,
        "messages": [
            {"role": "system", "content": "you are a solution archivist"},
            {
                "role": "user",
                "content": "=== Session: zzz999 ===\nhere is the trajectory",
            },
        ],
    }
    assert _llm_cache_key(a) == _llm_cache_key(b)


def test_llm_cache_key_differs_on_real_content_change() -> None:
    """Anything that *does* change the semantic input must change the key."""
    base = _base_params()
    changed = {
        **base,
        "messages": [
            {"role": "system", "content": "you are a solution archivist"},
            {"role": "user", "content": "a different trajectory body"},
        ],
    }
    assert _llm_cache_key(base) != _llm_cache_key(changed)


def test_llm_cache_roundtrip_respects_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With REFLEXIO_LLM_CACHE_DIR set, write→read returns the same content.
    With it unset, both helpers no-op (read returns None, write is a no-op)."""
    monkeypatch.setenv("REFLEXIO_LLM_CACHE_DIR", str(tmp_path))
    params = _base_params()

    assert _read_llm_cache(params) is None  # cold cache
    _write_llm_cache(params, "cached response body")
    assert _read_llm_cache(params) == "cached response body"

    # Verify the on-disk artifact is a well-formed JSON with the expected
    # keys — future maintainers should see this pinned.
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    obj = json.loads(files[0].read_text())
    assert obj == {"model": params["model"], "content": "cached response body"}

    # Now disable the cache — both helpers must no-op.
    monkeypatch.delenv("REFLEXIO_LLM_CACHE_DIR", raising=False)
    assert _read_llm_cache(params) is None
    _write_llm_cache(params, "should not be written")
    # The file count from the earlier write is preserved but no new file
    # was added under a dir the env var no longer points at.
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_llm_cache_write_noop_for_non_string_content(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only string content is cached — BaseModel / None / list etc. must
    no-op rather than crash or serialize garbage."""
    monkeypatch.setenv("REFLEXIO_LLM_CACHE_DIR", str(tmp_path))
    params = _base_params()

    _write_llm_cache(params, None)
    _write_llm_cache(params, {"already": "parsed"})
    _write_llm_cache(params, 42)

    assert list(tmp_path.glob("*.json")) == []
