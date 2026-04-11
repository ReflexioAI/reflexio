"""
Reusable Pydantic v2 validator types for reflexio.models.

This module provides:
1. **Data Integrity Validators** - NonEmptyStr, EmbeddingVector, numeric constraints
2. **Security Validators** - SafeHttpUrl (SSRF prevention), SanitizedStr (prompt injection mitigation)
3. **Mixins** - TimeRangeValidatorMixin for models with start_time/end_time

Usage:
    from reflexio.models.api_schema.validators import (
        NonEmptyStr, SafeHttpUrl, SanitizedNonEmptyStr, EmbeddingVector, ...
    )
"""

import ipaddress
import os
import re
from typing import Annotated, Any
from urllib.parse import urlparse

from pydantic import AfterValidator, HttpUrl

# Embedding vector dimensions — must match config_schema.EMBEDDING_DIMENSIONS.
# Duplicated here to avoid circular imports (config_schema imports from this module).
EMBEDDING_DIMENSIONS = 512


# =============================================================================
# Data Integrity Validators
# =============================================================================


def _check_non_empty_str(v: str) -> str:
    """Validate that a string is not empty or whitespace-only after stripping.

    Args:
        v (str): The string to validate

    Returns:
        str: The stripped string

    Raises:
        ValueError: If the string is empty or whitespace-only
    """
    stripped = v.strip()
    if not stripped:
        raise ValueError("String must not be empty or whitespace-only")
    return stripped


def _check_optional_non_empty_str(v: str | None) -> str | None:
    """Validate that an optional string, if provided, is not empty or whitespace-only.

    Args:
        v (Optional[str]): The string to validate, or None

    Returns:
        Optional[str]: The stripped string, or None

    Raises:
        ValueError: If the string is provided but empty or whitespace-only
    """
    if v is None:
        return v
    stripped = v.strip()
    if not stripped:
        raise ValueError("String must not be empty or whitespace-only")
    return stripped


def _check_embedding_dimensions(v: list[float]) -> list[float]:
    """Validate that an embedding vector is either empty or has the correct dimensions.

    Args:
        v (list[float]): The embedding vector

    Returns:
        list[float]: The validated embedding vector

    Raises:
        ValueError: If the embedding has wrong dimensions (not empty and not EMBEDDING_DIMENSIONS)
    """
    if len(v) > 0 and len(v) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Embedding must be empty or have exactly {EMBEDDING_DIMENSIONS} dimensions, "
            f"got {len(v)}"
        )
    return v


# Reusable Annotated types for data integrity
NonEmptyStr = Annotated[str, AfterValidator(_check_non_empty_str)]
"""String that rejects empty/whitespace-only values. Strips leading/trailing whitespace."""

OptionalNonEmptyStr = Annotated[
    str | None, AfterValidator(_check_optional_non_empty_str)
]
"""Optional string that, if provided, rejects empty/whitespace-only values."""

EmbeddingVector = Annotated[list[float], AfterValidator(_check_embedding_dimensions)]
"""Embedding vector that must be either empty or exactly EMBEDDING_DIMENSIONS (512) floats."""


# =============================================================================
# Security Validators — SSRF Prevention
# =============================================================================

# Cloud metadata endpoints — ALWAYS blocked (no legitimate use case)
METADATA_HOSTS = {"metadata.google.internal"}
METADATA_IPS = {"169.254.169.254", "fd00:ec2::254"}  # AWS/GCP/Azure


def _is_strict_mode() -> bool:
    """Check if strict URL validation is enabled (for production).

    Returns:
        bool: True if REFLEXIO_BLOCK_PRIVATE_URLS env var is set to true/1/yes
    """
    return os.environ.get("REFLEXIO_BLOCK_PRIVATE_URLS", "").lower() in (
        "true",
        "1",
        "yes",
    )


def _check_safe_url(v: Any) -> Any:
    """Validate URL is not targeting dangerous resources.

    Always blocks: cloud metadata endpoints, non-http schemes.
    In strict mode (REFLEXIO_BLOCK_PRIVATE_URLS=true): also blocks private IPs and localhost.

    Args:
        v: The URL value (HttpUrl or string)

    Returns:
        The validated URL value

    Raises:
        ValueError: If the URL targets a dangerous resource
    """
    url_str = str(v)
    parsed = urlparse(url_str)
    host = (parsed.hostname or "").lower()

    # ALWAYS block cloud metadata (never legitimate)
    if host in METADATA_HOSTS:
        raise ValueError(f"URL must not target cloud metadata: {host}")

    try:
        ip = ipaddress.ip_address(host)
        if str(ip) in METADATA_IPS:
            raise ValueError(f"URL must not target cloud metadata endpoint: {host}")

        # In strict mode, also block private/localhost
        if _is_strict_mode() and (
            ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local
        ):
            raise ValueError(
                f"URL targets private/reserved IP '{host}'. "
                f"If running locally, unset REFLEXIO_BLOCK_PRIVATE_URLS."
            )
    except ValueError as e:
        if "must not target" in str(e) or "targets private" in str(e):
            raise
        # Not an IP (hostname) — check localhost in strict mode
        if _is_strict_mode() and host in ("localhost", "0.0.0.0"):  # noqa: S104
            raise ValueError(
                f"URL targets '{host}'. "
                f"If running locally, unset REFLEXIO_BLOCK_PRIVATE_URLS."
            ) from None

    return v


SafeHttpUrl = Annotated[HttpUrl, AfterValidator(_check_safe_url)]
"""HTTP URL that blocks cloud metadata endpoints (always) and private IPs (in strict mode).

Use for: api_base, endpoint — anywhere the server fetches a user-provided URL.

Local dev:  works out of the box (localhost allowed)
Production: set REFLEXIO_BLOCK_PRIVATE_URLS=true in env/Dockerfile for full protection
"""


# =============================================================================
# Security Validators — Prompt Injection Mitigation
# =============================================================================

# Control characters that can manipulate prompt rendering or terminal output.
# Excludes tab (\x09), newline (\x0a), carriage return (\x0d) which are legitimate.
_CONTROL_CHAR_PATTERN = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"  # C0 control chars except \t \n \r
)


def _strip_control_chars(v: str) -> str:
    """Remove invisible control characters that could manipulate prompt rendering.

    Strips C0 control characters (NULL, bell, backspace, escape sequences, etc.)
    while preserving tabs, newlines, and carriage returns.

    Args:
        v (str): The string to sanitize

    Returns:
        str: The sanitized string with control characters removed
    """
    return _CONTROL_CHAR_PATTERN.sub("", v)


SanitizedStr = Annotated[str, AfterValidator(_strip_control_chars)]
"""String with control characters stripped. Use for user content flowing into LLM prompts."""

SanitizedNonEmptyStr = Annotated[
    str,
    AfterValidator(_strip_control_chars),
    AfterValidator(_check_non_empty_str),
]
"""String that is sanitized (control chars stripped) AND non-empty. Use for config prompts."""


# =============================================================================
# Time Range Validator Mixin
# =============================================================================


def _validate_image_url(v: str) -> str:
    """SSRF prevention: if URL is provided, must be safe http(s) or data URI.

    Args:
        v (str): The image URL to validate

    Returns:
        str: The validated URL

    Raises:
        ValueError: If the URL scheme is not http, https, or data, or if it targets unsafe resources
    """
    if not v:
        return v
    parsed = urlparse(v)
    if parsed.scheme not in ("http", "https", "data"):
        raise ValueError(
            f"Image URL scheme must be http, https, or data — got '{parsed.scheme}'"
        )
    if parsed.scheme in ("http", "https"):
        _check_safe_url(v)
    return v


class TimeRangeValidatorMixin:
    """Mixin for models with optional start_time/end_time fields.

    Validates that end_time is after start_time when both are provided.
    Add to models by including in the class bases alongside BaseModel.

    Usage:
        class MyRequest(TimeRangeValidatorMixin, BaseModel):
            start_time: Optional[datetime] = None
            end_time: Optional[datetime] = None
    """

    # Note: This is implemented as a classmethod to be called from
    # @model_validator(mode='after') in each model, rather than using
    # __init_subclass__ magic that could conflict with Pydantic's metaclass.
    @staticmethod
    def validate_time_range(start_time: Any, end_time: Any) -> None:
        """Validate that end_time is after start_time.

        Args:
            start_time: The start time value
            end_time: The end time value

        Raises:
            ValueError: If end_time is not after start_time
        """
        if start_time is not None and end_time is not None and end_time <= start_time:
            raise ValueError("end_time must be after start_time")
