"""Cross-version tolerance of the persisted window payload.

``token_totals`` is persisted JSON that a *different image* may decode: during a
rolling deploy, a window written by a new task can be picked up by an old one.
``decode_plan`` used to splat it straight into ``RunTokenTotals`` kwargs, so
adding a field raised ``TypeError`` inside ``window_executor.execute`` — on a
durable-learning path, unguarded.

Both halves of the fix are pinned here, because each covers a direction the
other cannot:

* decode ignores keys it does not know → a NEW window decodes on an old-enough
  build, and every future field addition is safe;
* encode writes only the pinned field set → the image already deployed, which
  has no tolerance of its own, never receives a key it cannot handle.
"""

import time

from reflexio.server.llm.token_accounting import RunTokenTotals
from reflexio.server.services.base_generation_service import PreparedGenerationRun
from reflexio.server.services.deferred_learning_plan import GenerationComputePlan
from reflexio.server.services.durable_learning.window_codec import (
    _PERSISTED_TOKEN_FIELDS,
    _decode_token_totals,
    _encode_token_totals,
    encode_plan,
)


class _FakeService:
    _last_model_provenance = None
    _last_extractor_run_stats = {"total": 1, "failed": 0, "timed_out": 0}


def _plan(totals: RunTokenTotals | None) -> GenerationComputePlan:
    return GenerationComputePlan(
        prepared=PreparedGenerationRun(None, "fake-extractor", "user1"),
        generated_count=1,
        billable_count=1,
        write_plan=None,
        bookmark_advance=None,
        generation_start=time.perf_counter(),
        extraction_run_ids=[],
        token_totals=totals,
    )


def test_decode_ignores_a_field_this_build_does_not_know() -> None:
    """A newer image's key must not fail the whole window execution."""
    totals = _decode_token_totals(
        {
            "prompt_tokens": 12,
            "completion_tokens": 3,
            "a_field_from_a_newer_image": 99,
        }
    )
    assert totals is not None
    assert totals.prompt_tokens == 12
    assert totals.completion_tokens == 3


def test_decode_round_trips_the_fields_it_does_know() -> None:
    assert _decode_token_totals({"prompt_tokens": 5}) == RunTokenTotals(prompt_tokens=5)
    assert _decode_token_totals(None) is None
    assert _decode_token_totals({}) is None


def test_encode_writes_only_the_pinned_compatibility_floor() -> None:
    """The cache fields exist on the dataclass but are NOT persisted yet.

    This is the half that protects the image already running, which has no
    unknown-key tolerance. Widening ``_PERSISTED_TOKEN_FIELDS`` is a deliberate
    later step, not something to do by regenerating it from the dataclass.
    """
    totals = RunTokenTotals(
        prompt_tokens=100,
        completion_tokens=20,
        cache_read_input_tokens=80,
        cache_write_input_tokens=10,
    )
    encoded = _encode_token_totals(totals)
    assert encoded is not None
    assert encoded == {"prompt_tokens": 100, "completion_tokens": 20}
    assert set(encoded) == set(_PERSISTED_TOKEN_FIELDS)
    assert _encode_token_totals(None) is None


def test_encode_plan_routes_token_totals_through_the_filter() -> None:
    """Pin the wiring, not just the helper — an unrouted `encode_plan` would
    reintroduce the break while every helper test stayed green."""
    encoded = encode_plan(
        _plan(
            RunTokenTotals(
                prompt_tokens=7,
                completion_tokens=1,
                cache_read_input_tokens=5,
            )
        ),
        _FakeService(),
    )
    assert encoded["token_totals"] == {"prompt_tokens": 7, "completion_tokens": 1}


def test_a_new_windows_payload_survives_this_builds_decoder() -> None:
    """End to end: the shape a future image writes still decodes here."""
    future_payload = {
        "prompt_tokens": 42,
        "completion_tokens": 9,
        "cache_read_input_tokens": 30,
        "cache_write_input_tokens": 4,
        "some_meter_invented_later": 1,
    }
    totals = _decode_token_totals(future_payload)
    assert totals is not None
    assert totals.prompt_tokens == 42
    assert totals.cache_read_input_tokens == 30
