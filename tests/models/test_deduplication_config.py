"""Tests for ``DeduplicationConfig`` (consolidation search + complexity budget)."""

import pytest
from pydantic import ValidationError

from reflexio.models.config_schema import DeduplicationConfig


def test_dedup_config_defaults():
    cfg = DeduplicationConfig()
    assert cfg.search_threshold == 0.4
    assert cfg.search_top_k == 5
    assert cfg.max_unified_content_chars == 1200


def test_max_unified_content_chars_must_be_positive():
    with pytest.raises(ValidationError):
        DeduplicationConfig(max_unified_content_chars=0)


def test_max_unified_content_chars_rejects_negative():
    with pytest.raises(ValidationError):
        DeduplicationConfig(max_unified_content_chars=-100)


def test_max_unified_content_chars_override():
    cfg = DeduplicationConfig(max_unified_content_chars=500)
    assert cfg.max_unified_content_chars == 500
