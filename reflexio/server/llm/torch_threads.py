"""Shared configuration owner for Torch's CPU intra-op thread pool.

Model loaders call this before construction. Importing the module (or leaving
Nomic's optional budget unset) does not import Torch.
"""

from __future__ import annotations

import logging
import os
import threading

from reflexio.removed_env import reject_removed_env
from reflexio.server.env_utils import positive_int_env

_LOGGER = logging.getLogger(__name__)
_THREAD_CONFIGURATION_LOCK = threading.Lock()


def configure_torch_threads(*, default: int | None = None) -> None:
    """Apply the shared canonical budget before importing or configuring Torch.

    Nomic keeps its unset/no-op default; CPU E5 supplies its existing default.
    Invalid values retain the caller's default using the shared env validator.
    """
    reject_removed_env(os.environ, canonical="REFLEXIO_INFERENCE_TORCH_THREADS")
    threads = (
        positive_int_env("REFLEXIO_INFERENCE_TORCH_THREADS", 0, _LOGGER) or default
    )
    if threads is None:
        return
    with _THREAD_CONFIGURATION_LOCK:
        import torch

        if torch.get_num_threads() != threads:
            torch.set_num_threads(threads)
