"""Public exports loaded on demand so inference stays independent of app code."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from reflexio.server.llm.rerank.cross_encoder_reranker import (
        score_pairs,
        score_pairs_with_model,
    )

_EXPORTS = {
    name: module
    for module, names in {
        "reflexio.server.llm.rerank.cross_encoder_reranker": (
            "score_pairs",
            "score_pairs_with_model",
        )
    }.items()
    for name in names
}

__all__ = ["score_pairs", "score_pairs_with_model"]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
