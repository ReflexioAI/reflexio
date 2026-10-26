"""Qualify real native Nomic inference from an installed full-package wheel."""

from __future__ import annotations

import json
import math
import os
import sys
from functools import wraps
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any
from unittest.mock import patch

import reflexio
from reflexio.server.llm.providers import nomic_embedding_provider
from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder


def main() -> None:
    """Fail on source imports, legacy loading, truncation or invalid vectors."""
    package_path = Path(reflexio.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Qualification requires an installed wheel, not editable source"
        )
    installed = distribution("reflexio-ai")
    direct_url = json.loads(installed.read_text("direct_url.json") or "{}")
    if direct_url.get("dir_info", {}).get("editable"):
        raise RuntimeError("Qualification rejects editable installs")
    files = {str(path) for path in installed.files or []}
    for relative, origin in (
        ("reflexio/__init__.py", reflexio.__file__),
        (
            "reflexio/server/llm/providers/nomic_embedding_provider.py",
            nomic_embedding_provider.__file__,
        ),
    ):
        if (
            relative not in files
            or Path(str(installed.locate_file(relative))).resolve()
            != Path(origin).resolve()
        ):
            raise RuntimeError("Imported source is not owned by the installed wheel")
    os.environ["REFLEXIO_NATIVE_COMPUTE_THREADS"] = "2"
    embedder = NomicEmbedder()
    model = embedder._load()
    model_class = type(model[0].auto_model).__module__
    if not model_class.startswith("transformers.models.nomic_bert."):
        raise RuntimeError(f"Expected native Nomic implementation, got {model_class}")
    long_text = "search_document: " + (
        "timeline dates user preferences and explicit milestones. " * 260
    )
    tokens = len(model.tokenizer(long_text)["input_ids"])
    if not 2048 < tokens < model.max_seq_length == 8192:
        raise RuntimeError(
            "Long-input qualification no longer covers the model contract"
        )
    texts = ["search_query: dates", "search_document: explicit milestones", long_text]
    forward_lengths: list[int] = []

    auto_model = model[0].auto_model
    original_forward = auto_model.forward

    @wraps(original_forward)
    def observe_forward(*args: object, **kwargs: Any) -> Any:
        forward_lengths.append(int(kwargs["input_ids"].shape[-1]))
        return original_forward(*args, **kwargs)

    # Sentence Transformers can call forward directly, bypassing nn.Module hooks.
    with patch.object(auto_model, "forward", observe_forward):
        # Encode separately to avoid padding the short probes to the long input.
        for text in texts:
            vectors = embedder.embed([text])
            if len(vectors) != 1 or len(vectors[0]) != 512:
                raise RuntimeError("Unexpected embedding dimensions")
            vector = vectors[0]
            if not all(math.isfinite(value) for value in vector):
                raise RuntimeError("Nonfinite embedding")
            if not math.isclose(
                sum(value * value for value in vector), 1, abs_tol=1e-4
            ):
                raise RuntimeError("Embedding is not normalized")
    if len(forward_lengths) != len(texts) or forward_lengths[-1] != tokens:
        raise RuntimeError("Long probe was truncated before model inference")
    print(
        json.dumps(
            {
                "result": "passed",
                "sdk": version("reflexio-ai"),
                "transformers": version("transformers"),
                "model_class": model_class,
                "package_path": str(package_path),
                "probes": len(texts),
                "long_input_tokens": tokens,
                "forward_input_tokens": forward_lengths,
                "dimensions": 512,
            }
        )
    )


if __name__ == "__main__":
    main()
