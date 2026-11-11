"""Verified native context capacity shared by preparation and its consumer."""

# Complete provider initialization before importing the third-party library.
import reflexio.provider_bootstrap  # noqa: F401

# isort: split

import litellm

from .compaction import CompactionError


def verified_native_context(model: str) -> int:
    """Read the same verified context limit at selection and provider dispatch."""
    # LiteLLM's catalogue lacks M2.7 in the pinned dependency. Its official
    # provider limit is verified in the design doc; never guess for an
    # unknown/custom route.
    if model == "minimax/MiniMax-M2.7":
        context_tokens = 204_800
    else:
        try:
            info = litellm.get_model_info(model)
            context_tokens = int(
                info.get("max_input_tokens") or info.get("max_tokens") or 0
            )
        except Exception as exc:  # noqa: BLE001 - catalogue lookup has no typed unknown-route error.
            raise CompactionError("native model context is not verified") from exc
    if context_tokens <= 0:
        raise CompactionError("native model has no verified usable context budget")
    return context_tokens
