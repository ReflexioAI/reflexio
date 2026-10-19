"""Shared configuration owner for Torch's CPU intra-op thread pool.

Model loaders call this before construction. Importing the module (or leaving
Nomic's optional budget unset) does not import Torch.
"""

from __future__ import annotations

import logging
import threading

from reflexio.server.env_utils import positive_int_env

_LOGGER = logging.getLogger(__name__)
_THREAD_CONFIGURATION_LOCK = threading.Lock()


def configure_torch_threads(*, default: int | None = None) -> None:
    """Apply one budget; accept the old Nomic name as an equivalent alias.

    Conflicting explicit budgets cannot configure one shared pool. Reject that
    ambiguity before touching Torch rather than let model-load order decide.
    Nomic retains its unset/no-op default; CPU E5 supplies its existing default.
    Invalid values retain the caller's default, using the shared env validator.
    """
    canonical = positive_int_env("REFLEXIO_INFERENCE_TORCH_THREADS", 0, _LOGGER)
    legacy = positive_int_env("REFLEXIO_EMBED_TORCH_THREADS", 0, _LOGGER)
    if canonical and legacy and canonical != legacy:
        raise ValueError(
            "Conflicting Torch thread budgets: set REFLEXIO_INFERENCE_TORCH_THREADS "
            "and remove REFLEXIO_EMBED_TORCH_THREADS, or give both the same value"
        )
    threads = canonical or legacy or default
    if threads is None:
        return
    with _THREAD_CONFIGURATION_LOCK:
        import torch

        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
