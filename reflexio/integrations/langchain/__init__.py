"""LangChain integration for Reflexio.

Provides callback handlers, tools, retrievers, and prompt helpers
to enable LangChain agents to self-improve via Reflexio.

Install with: pip install reflexio-client[langchain]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Prompt helpers have no langchain dependency — always available
from reflexio.integrations.langchain.prompt import get_reflexio_context

if TYPE_CHECKING:
    from reflexio.integrations.langchain.callback import ReflexioCallbackHandler
    from reflexio.integrations.langchain.middleware import ReflexioChatModel
    from reflexio.integrations.langchain.retriever import ReflexioRetriever
    from reflexio.integrations.langchain.tool import ReflexioSearchTool


def __getattr__(name: str) -> type:
    """Lazy import for LangChain-dependent classes.

    Raises ImportError with install instructions if langchain-core is missing.
    """
    _lazy_imports = {
        "ReflexioCallbackHandler": "reflexio.integrations.langchain.callback",
        "ReflexioSearchTool": "reflexio.integrations.langchain.tool",
        "ReflexioRetriever": "reflexio.integrations.langchain.retriever",
        "ReflexioChatModel": "reflexio.integrations.langchain.middleware",
    }
    if name in _lazy_imports:
        import importlib

        module = importlib.import_module(_lazy_imports[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ReflexioCallbackHandler",
    "ReflexioChatModel",
    "ReflexioRetriever",
    "ReflexioSearchTool",
    "get_reflexio_context",
]
