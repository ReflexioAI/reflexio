"""Import-boundary guards for service-only local inference."""

from __future__ import annotations

import subprocess
import sys


def _assert_modules_not_loaded(import_target: str, forbidden: list[str]) -> None:
    script = (
        f"import sys; import {import_target}; "
        f"forbidden={forbidden!r}; "
        "loaded=[name for name in forbidden if name in sys.modules]; "
        "assert not loaded, loaded"
    )
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603


def test_main_embedding_client_imports_no_local_model_runner() -> None:
    _assert_modules_not_loaded(
        "reflexio.server.llm._litellm_embedding",
        [
            "reflexio.server.llm.providers.local_embedding_provider",
            "reflexio.server.llm.providers.nomic_embedding_provider",
        ],
    )


def test_main_rerank_client_imports_no_cross_encoder_runner() -> None:
    _assert_modules_not_loaded(
        "reflexio.server.llm.rerank",
        ["reflexio.server.llm.rerank.cross_encoder_model"],
    )
