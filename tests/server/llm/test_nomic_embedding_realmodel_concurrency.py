"""Opt-in real native Nomic inference and shared-model concurrency proofs.

Native Transformers owns rotary implementation details. Assert actual inference,
model provenance, long input handling and equivalence to a serial baseline;
legacy cache-attribute resets no longer exercise this implementation.
RUN_LOW_PRIORITY=1 enables these keyless, real-model tests. Once enabled, model
loading or inference failures fail the test rather than silently skipping it.
"""

from __future__ import annotations

import concurrent.futures as cf
from functools import wraps
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch

from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder
from tests.server.test_utils import skip_in_precommit, skip_low_priority

_N_WORKERS = 4
_ROUNDS = 20
_ATOL = 1e-4  # inference is deterministic; a race corrupts grossly, not by 1e-4

# Exercise mixed input lengths against the serialized reference.
_LONG = " ".join(["the quick brown fox jumps over the lazy dog"] * 12)  # ~120 tok
_TEXTS = [
    "hi",
    "short phrase here now",
    "a medium length sentence that carries a bit more token weight than its peers",
    "user asked about billing and the agent explained the invoice line items here " * 2,
    _LONG,
    _LONG[: len(_LONG) * 2 // 3],
]


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
    """Load the real native model; enabled qualification must fail on errors."""
    embedder = NomicEmbedder()
    model = embedder._load()
    assert type(model[0].auto_model).__module__.startswith(
        "transformers.models.nomic_bert."
    )
    return embedder


@skip_in_precommit
@skip_low_priority
def test_model_lock_holds_under_real_concurrency(real_embedder: NomicEmbedder) -> None:
    """Concurrent real encodes preserve the single-thread reference vectors."""
    # Keep each forward single-threaded so the Python threads — not torch intra-op
    # threads — are the concurrency, and a constrained box is not oversubscribed.
    # Set here (not at module scope) so it only runs when the test actually runs,
    # never mutating the process-global thread count during collection of skipped
    # runs in other tiers.
    torch.set_num_threads(1)

    # Single-thread, one text at a time — the uncorrupted reference vectors.
    baseline = {text: _embed_one(real_embedder, text) for text in _TEXTS}

    burst = sorted(_TEXTS, key=len)

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

    for _ in range(_ROUNDS):
        with cf.ThreadPoolExecutor(max_workers=_N_WORKERS) as pool:
            list(pool.map(task, burst))

    assert errors == [], f"{len(errors)} concurrent embed(s) raised: {errors[:5]}"
    assert mismatches == [], (
        f"{len(mismatches)} concurrent vector(s) diverged from the serial "
        f"baseline: {mismatches[:5]}"
    )


@skip_in_precommit
@skip_low_priority
def test_native_nomic_encodes_long_input(real_embedder: NomicEmbedder) -> None:
    """Exercise the removed-helper failure path beyond the 2048-token base."""
    torch.set_num_threads(2)
    model = real_embedder._load()
    text = "search_document: " + (
        "timeline dates user preferences and explicit milestones. " * 260
    )
    token_count = len(model.tokenizer(text)["input_ids"])
    assert 2048 < token_count < model.max_seq_length == 8192
    forward_lengths: list[int] = []

    auto_model = model[0].auto_model
    original_forward = auto_model.forward

    @wraps(original_forward)
    def observe_forward(*args: object, **kwargs: Any) -> Any:
        forward_lengths.append(int(kwargs["input_ids"].shape[-1]))
        return original_forward(*args, **kwargs)

    # Sentence Transformers can call forward directly, bypassing nn.Module hooks.
    with patch.object(auto_model, "forward", observe_forward):
        vectors = np.asarray(real_embedder.embed([text]))
    assert forward_lengths == [token_count]

    assert vectors.shape == (1, 512)
    assert np.isfinite(vectors).all()
    assert np.isclose(np.linalg.norm(vectors[0]), 1)
