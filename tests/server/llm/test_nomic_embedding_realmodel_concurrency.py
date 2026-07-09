"""Real-model concurrency proof for the Step A encode-lock fix (design §6 row 6ii).

The fake-model concurrency tests
(``test_embedding_service_concurrency.py``,
``test_nomic_embedding_provider.py::_ReentrancyDetectingModel``) prove that
``NomicEmbedder._model_lock`` is *acquired* — they serialize a stub ``encode()``.
They cannot prove it *fixes the real race*, because a stub does not run the
nomic-bert rotary cache (``_cos_cached`` / ``_sin_cached``) whose mid-forward
resize is what actually corrupts concurrent encodes in prod. This test closes
that gap with the REAL ``nomic-embed-text-v1.5`` model.

Invariant under test (positive only)
------------------------------------
With ``_model_lock`` intact, N threads embedding a mixed-length burst — with the
rotary cache reset to COLD before every round, reproducing the prod boot-window
state where ``_update_cos_sin_cache`` must rebuild the shared buffers — produce
ZERO exceptions and vectors byte-for-byte close (``atol=1e-4``) to a single-thread
serial baseline. The cold-cache reset is essential: pre-warming the cache to the
max length hides the race, because the resize (the window a concurrent reader
corrupts) never fires again.

Why the negative control is NOT asserted here
----------------------------------------------
Neutralizing ``_model_lock`` (swap it for ``contextlib.nullcontext()``) and
re-running this exact harness reproduces the prod signature
``RuntimeError: The size of tensor a (N) must match the size of tensor b (M)``.
Verified 2026-07-09 via the standalone proof harness: 12 workers × 40 rounds gave
0 errors locked vs 67 errors unlocked over 480 concurrent embeds. That control is
deliberately left OUT of the committed assertions — its timing/torch-version
fragility makes it a flaky gate — but a future reader can re-confirm the test
genuinely bites by monkeypatching ``embedder._model_lock = contextlib.nullcontext()``
before the concurrent phase and watching it go red.

This test is heavy (loads torch + the ~550 MB real model) and keyless (local
embeddings, no paid API), so it is opt-in: ``@skip_low_priority`` keeps it out of
the default/fast/unit tier (run it with ``RUN_LOW_PRIORITY=1``), and
``@skip_in_precommit`` also excludes it from the pre-commit hook — reusing the
repo's heaviest-opt-in-test convention even though nothing here costs money.
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any

import numpy as np
import pytest
import torch

from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder
from tests.server.test_utils import skip_in_precommit, skip_low_priority

_N_WORKERS = 12
_ROUNDS = 20
_ATOL = 1e-4  # inference is deterministic; a race corrupts grossly, not by 1e-4

# Mixed lengths are the trigger: the rotary cache is (re)sized whenever a longer
# sequence than any seen so far arrives, and that resize is the window a concurrent
# reader corrupts. A length-ASCENDING burst makes several threads hit the resize
# at once.
_LONG = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)  # ~120 tok
_TEXTS = [
    "hi",
    "short phrase here now",
    "a medium length sentence that carries a bit more token weight than its peers",
    "user asked about billing and the agent explained the invoice line items here " * 2,
    _LONG,
    _LONG[: len(_LONG) * 2 // 3],
]


def _reset_rotary(model: Any) -> int:
    """Reset every rotary submodule's cache to cold.

    Reproduces the prod boot-window state where the next forwards must rebuild
    ``_cos_cached`` / ``_sin_cached`` — the racy resize the lock must serialize.

    Args:
        model (Any): The loaded sentence-transformers model.

    Returns:
        int: How many rotary modules were reset. Zero means the cache attribute
            was renamed upstream and this reset is a silent no-op — the caller
            MUST assert this is non-zero, or the test passes on a warm cache that
            never opens the race window (a false green).
    """
    reset = 0
    for module in model.modules():
        if hasattr(module, "_seq_len_cached"):
            module._seq_len_cached = 0
            module._cos_cached = None
            module._sin_cached = None
            reset += 1
    return reset


def _embed_one(embedder: NomicEmbedder, text: str) -> np.ndarray:
    """Embed a single text and return its vector as float64 for comparison.

    Args:
        embedder (NomicEmbedder): The embedder under test.
        text (str): Input to encode.

    Returns:
        np.ndarray: The embedding vector as ``float64``.
    """
    return np.asarray(embedder.embed([text])[0], dtype=np.float64)


@pytest.fixture
def real_embedder() -> NomicEmbedder:
    """Return a fresh ``NomicEmbedder`` with the real model loaded.

    Uses a fresh instance (not the process-wide singleton) so cache/state does
    not bleed across tests. Skips — rather than fails — when the model cannot be
    loaded (no network on a cold cache, or ``sentence-transformers`` missing).

    Returns:
        NomicEmbedder: An embedder whose real model is loaded and ready.
    """
    pytest.importorskip(
        "sentence_transformers",
        reason="sentence-transformers is required for the real-model concurrency test",
    )
    embedder = NomicEmbedder()
    try:
        embedder._load()
    except Exception as exc:  # noqa: BLE001 — any load failure is a skip, not a fail
        pytest.skip(f"real Nomic model unavailable (no network / not cached): {exc}")
    return embedder


@skip_in_precommit
@skip_low_priority
def test_model_lock_holds_under_real_concurrency(real_embedder: NomicEmbedder) -> None:
    """The Step A ``_model_lock`` keeps real concurrent encodes uncorrupted.

    For ``_ROUNDS`` rounds: reset every rotary cache to cold, then fire one embed
    per worker concurrently over a length-ascending mixed burst. Assert zero
    exceptions and that every concurrent vector matches its single-thread serial
    baseline within ``_ATOL``.
    """
    # Keep each forward single-threaded so the Python threads — not torch intra-op
    # threads — are the concurrency, and a constrained box is not oversubscribed.
    # Set here (not at module scope) so it only runs when the test actually runs,
    # never mutating the process-global thread count during collection of skipped
    # runs in other tiers.
    torch.set_num_threads(1)
    model = real_embedder._load()

    # Single-thread, one text at a time — the uncorrupted reference vectors.
    baseline = {text: _embed_one(real_embedder, text) for text in _TEXTS}

    # Ascending lengths so worker k's forward may trigger a resize while workers
    # k-1 / k+1 are mid-forward reading the shared cache.
    burst = sorted((_TEXTS * (_N_WORKERS // len(_TEXTS) + 1))[:_N_WORKERS], key=len)

    errors: list[str] = []
    mismatches: list[str] = []

    def task(text: str) -> None:
        try:
            vec = _embed_one(real_embedder, text)
        except Exception as exc:  # noqa: BLE001 — capturing the race is the point
            errors.append(f"{type(exc).__name__}: {exc}")
            return
        base = baseline[text]
        if vec.shape != base.shape or not np.allclose(vec, base, atol=_ATOL):
            mismatches.append(f"len={len(text)} shape={vec.shape} vs {base.shape}")

    for round_idx in range(_ROUNDS):
        n_reset = _reset_rotary(model)
        if round_idx == 0:
            assert n_reset > 0, (
                "_reset_rotary matched no rotary modules — the cache attribute was "
                "likely renamed upstream, so the reset is a no-op and this test no "
                "longer opens the race window it exists to guard."
            )
        with cf.ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            list(pool.map(task, burst))

    assert errors == [], f"{len(errors)} concurrent embed(s) raised: {errors[:5]}"
    assert mismatches == [], (
        f"{len(mismatches)} concurrent vector(s) diverged from the serial "
        f"baseline: {mismatches[:5]}"
    )
