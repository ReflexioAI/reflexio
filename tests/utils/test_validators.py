"""Tests for reflexio_commons api_schema validators."""

from datetime import datetime

import pytest
from pydantic import BaseModel, ValidationError

from reflexio.models.api_schema.validators import (
    EMBEDDING_DIMENSIONS,
    NonEmptyStr,
    OptionalNonEmptyStr,
    SanitizedNonEmptyStr,
    SanitizedStr,
    TimeRangeValidatorMixin,
    _check_embedding_dimensions,
    _check_non_empty_str,
    _check_optional_non_empty_str,
    _check_safe_url,
    _is_strict_mode,
    _strip_control_chars,
)

# ===============================
# Data Integrity Validators
# ===============================


class TestNonEmptyStr:
    """Tests for NonEmptyStr validator."""

    def test_valid_string(self):
        result = _check_non_empty_str("hello")
        assert result == "hello"

    def test_strips_whitespace(self):
        result = _check_non_empty_str("  hello  ")
        assert result == "hello"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            _check_non_empty_str("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            _check_non_empty_str("   ")

    def test_tab_only_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            _check_non_empty_str("\t\n")


class TestOptionalNonEmptyStr:
    """Tests for OptionalNonEmptyStr validator."""

    def test_none_returns_none(self):
        assert _check_optional_non_empty_str(None) is None

    def test_valid_string(self):
        result = _check_optional_non_empty_str("hello")
        assert result == "hello"

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            _check_optional_non_empty_str("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="not be empty"):
            _check_optional_non_empty_str("   ")


class TestEmbeddingVector:
    """Tests for EmbeddingVector validator."""

    def test_empty_vector_valid(self):
        result = _check_embedding_dimensions([])
        assert result == []

    def test_correct_dimensions_valid(self):
        vec = [0.1] * EMBEDDING_DIMENSIONS
        result = _check_embedding_dimensions(vec)
        assert len(result) == EMBEDDING_DIMENSIONS

    def test_wrong_dimensions_raises(self):
        vec = [0.1] * 10
        with pytest.raises(ValueError, match=f"exactly {EMBEDDING_DIMENSIONS}"):
            _check_embedding_dimensions(vec)


# ===============================
# Security Validators - SSRF Prevention
# ===============================


class TestIsStrictMode:
    """Tests for _is_strict_mode function."""

    def test_strict_mode_true(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")
            assert _is_strict_mode() is True

    def test_strict_mode_1(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "1")
            assert _is_strict_mode() is True

    def test_strict_mode_yes(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "yes")
            assert _is_strict_mode() is True

    def test_strict_mode_off(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "false")
            assert _is_strict_mode() is False

    def test_strict_mode_not_set(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.delenv("REFLEXIO_BLOCK_PRIVATE_URLS", raising=False)
            assert _is_strict_mode() is False


class TestCheckSafeUrl:
    """Tests for _check_safe_url validator."""

    def test_public_url_allowed(self):
        result = _check_safe_url("https://api.example.com/v1")
        assert str(result) == "https://api.example.com/v1"

    def test_metadata_host_blocked(self):
        with pytest.raises(ValueError, match="cloud metadata"):
            _check_safe_url("http://metadata.google.internal/v1")

    def test_metadata_ip_blocked(self):
        with pytest.raises(ValueError, match="cloud metadata"):
            _check_safe_url("http://169.254.169.254/latest/meta-data")

    def test_localhost_allowed_in_non_strict(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "false")
            result = _check_safe_url("http://localhost:8080/api")
            assert "localhost" in str(result)

    def test_localhost_blocked_in_strict(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")
            with pytest.raises(ValueError, match="targets"):
                _check_safe_url("http://localhost:8080/api")

    def test_private_ip_blocked_in_strict(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")
            with pytest.raises(ValueError, match="private"):
                _check_safe_url("http://192.168.1.1/api")

    def test_private_ip_allowed_in_non_strict(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "false")
            result = _check_safe_url("http://192.168.1.1/api")
            assert "192.168.1.1" in str(result)

    def test_zero_addr_blocked_in_strict(self):
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("REFLEXIO_BLOCK_PRIVATE_URLS", "true")
            with pytest.raises(ValueError, match="targets"):
                _check_safe_url("http://0.0.0.0:8080/api")


# ===============================
# Security Validators - Prompt Injection
# ===============================


class TestStripControlChars:
    """Tests for _strip_control_chars function."""

    def test_normal_string_unchanged(self):
        assert _strip_control_chars("hello world") == "hello world"

    def test_tabs_and_newlines_preserved(self):
        assert _strip_control_chars("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_null_byte_stripped(self):
        assert _strip_control_chars("hello\x00world") == "helloworld"

    def test_bell_stripped(self):
        assert _strip_control_chars("alert\x07bell") == "alertbell"

    def test_escape_sequence_stripped(self):
        assert _strip_control_chars("escape\x1bsequence") == "escapesequence"

    def test_carriage_return_preserved(self):
        assert _strip_control_chars("line1\r\nline2") == "line1\r\nline2"


# ===============================
# TimeRangeValidatorMixin
# ===============================


class TestTimeRangeValidatorMixin:
    """Tests for TimeRangeValidatorMixin."""

    def test_valid_range_no_error(self):
        TimeRangeValidatorMixin.validate_time_range(
            start_time=datetime(2024, 1, 1),
            end_time=datetime(2024, 1, 2),
        )

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            TimeRangeValidatorMixin.validate_time_range(
                start_time=datetime(2024, 1, 2),
                end_time=datetime(2024, 1, 1),
            )

    def test_equal_times_raises(self):
        dt = datetime(2024, 1, 1)
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            TimeRangeValidatorMixin.validate_time_range(
                start_time=dt,
                end_time=dt,
            )

    def test_none_start_no_error(self):
        TimeRangeValidatorMixin.validate_time_range(
            start_time=None,
            end_time=datetime(2024, 1, 1),
        )

    def test_none_end_no_error(self):
        TimeRangeValidatorMixin.validate_time_range(
            start_time=datetime(2024, 1, 1),
            end_time=None,
        )

    def test_both_none_no_error(self):
        TimeRangeValidatorMixin.validate_time_range(
            start_time=None,
            end_time=None,
        )


# ===============================
# Pydantic Annotated types integration
# ===============================


class TestAnnotatedTypes:
    """Integration tests for Pydantic Annotated types."""

    def test_non_empty_str_in_model(self):
        class MyModel(BaseModel):
            name: NonEmptyStr

        m = MyModel(name="  hello  ")
        assert m.name == "hello"

    def test_non_empty_str_empty_fails(self):
        class MyModel(BaseModel):
            name: NonEmptyStr

        with pytest.raises(ValidationError):
            MyModel(name="")

    def test_optional_non_empty_str_none(self):
        class MyModel(BaseModel):
            name: OptionalNonEmptyStr = None

        m = MyModel()
        assert m.name is None

    def test_sanitized_str_strips_control_chars(self):
        class MyModel(BaseModel):
            text: SanitizedStr

        m = MyModel(text="hello\x00\x07world")
        assert m.text == "helloworld"

    def test_sanitized_non_empty_str(self):
        class MyModel(BaseModel):
            text: SanitizedNonEmptyStr

        m = MyModel(text="  valid\x00text  ")
        assert m.text == "validtext"

    def test_sanitized_non_empty_str_empty_fails(self):
        class MyModel(BaseModel):
            text: SanitizedNonEmptyStr

        with pytest.raises(ValidationError):
            MyModel(text="\x00\x07")
