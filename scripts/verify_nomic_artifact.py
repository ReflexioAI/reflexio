"""Qualify real native Nomic inference from an installed full-package wheel."""

from __future__ import annotations

import json
import math
import os
import sys
from importlib.metadata import version
from pathlib import Path

import reflexio
from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder


def main() -> None:
    """Fail on source imports, legacy loading, truncation or invalid vectors."""
    package_path = Path(reflexio.__file__).resolve()
    if not package_path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Qualification requires an installed wheel, not editable source"
        )
    os.environ["REFLEXIO_EMBED_TORCH_THREADS"] = "2"
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
    # Encode separately to avoid padding the short probes to the long input.
    for text in texts:
        vectors = embedder.embed([text])
        if len(vectors) != 1 or len(vectors[0]) != 512:
            raise RuntimeError("Unexpected embedding dimensions")
        vector = vectors[0]
        if not all(math.isfinite(value) for value in vector):
            raise RuntimeError("Nonfinite embedding")
        if not math.isclose(sum(value * value for value in vector), 1, abs_tol=1e-4):
            raise RuntimeError("Embedding is not normalized")
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
                "dimensions": 512,
            }
        )
    )


if __name__ == "__main__":
    main()
