"""The fixed size policy for an open-world evidence payload."""

from __future__ import annotations

from reflexio.server.metering_signals import count_input_tokens
from reflexio.server.services.offline_tuner.evidence.open_world_policy import (
    analyst_payload_within_limits,
    bundle_within_analyst_payload_budget,
)


def _counts(text: str) -> dict[str, int]:
    return {
        "content_byte_count": len(text.encode("utf-8")),
        "content_token_count": count_input_tokens(text),
    }


def test_session_over_by_storage_row_but_under_by_content_is_admitted() -> None:
    assert analyst_payload_within_limits(
        content_byte_count=21_000, content_token_count=4_980
    )


def test_whitespace_padded_content_is_still_refused() -> None:
    """60,000 bytes of spaces is ~480 content tokens and must not pass."""
    assert not analyst_payload_within_limits(
        content_byte_count=60_000, content_token_count=480
    )


def test_whitespace_padding_is_refused_at_its_measured_token_count() -> None:
    """The same guard, against the real tokenizer rather than an asserted count.

    A hand-written token count is a check that cannot fail if the density the
    cap is sized against is wrong.
    """
    assert not analyst_payload_within_limits(**_counts(" " * 60_000))


def test_real_prose_at_the_full_budget_is_admitted() -> None:
    """The cap must bind on whitespace and on nothing else."""
    counts = _counts("The quick brown fox jumps over the lazy dog. " * 700)

    assert counts["content_token_count"] <= 8_000
    assert analyst_payload_within_limits(**counts)


def test_the_token_budget_boundary_is_exact() -> None:
    """The token half still binds; the density cap did not replace it."""
    assert analyst_payload_within_limits(
        content_byte_count=35_000, content_token_count=8_000
    )
    assert not analyst_payload_within_limits(
        content_byte_count=35_000, content_token_count=8_001
    )


def test_the_density_boundary_is_exact() -> None:
    assert analyst_payload_within_limits(
        content_byte_count=8_000, content_token_count=1_000
    )
    assert not analyst_payload_within_limits(
        content_byte_count=8_001, content_token_count=1_000
    )


def test_empty_content_is_admitted() -> None:
    assert analyst_payload_within_limits(content_byte_count=0, content_token_count=0)


def test_the_selected_bundle_render_budget_boundary_is_exact() -> None:
    """Pin the local admission backstop independently of per-session density."""
    assert bundle_within_analyst_payload_budget(render_token_count=250_000)
    assert not bundle_within_analyst_payload_budget(render_token_count=250_001)
