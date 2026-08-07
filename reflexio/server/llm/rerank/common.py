"""Shared reranker contract without model-runtime imports."""

from __future__ import annotations

from reflexio.server.env_utils import env_str, env_truthy

ENGLISH_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MULTILINGUAL_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ENGLISH_RERANK_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"
MULTILINGUAL_RERANK_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANK_MODEL = ENGLISH_RERANK_MODEL
RERANK_ENABLED_ENV_VAR = "REFLEXIO_RERANK_ENABLED"

RERANK_MODEL_REVISIONS = {
    ENGLISH_RERANK_MODEL: ENGLISH_RERANK_REVISION,
    MULTILINGUAL_RERANK_MODEL: MULTILINGUAL_RERANK_REVISION,
}

# The English values preserve the existing production policy. Multilingual
# values are selected by the committed calibration report and are intentionally
# keyed by retrieval arm because raw cross-encoder logits are model-specific.
RERANK_FLOOR_DEFAULTS: dict[str, dict[str, float]] = {
    ENGLISH_RERANK_MODEL: {
        "profiles": -3.0,
        "user_playbooks": -3.0,
        "agent_playbooks": -3.0,
    },
    MULTILINGUAL_RERANK_MODEL: {
        "profiles": 2.18,
        "user_playbooks": 2.87,
        "agent_playbooks": -6.75,
    },
}


class CrossEncoderUnavailableError(RuntimeError):
    """Raised when reranking is disabled or its service/model is unavailable.

    ``report_failure`` distinguishes an operational remote-service failure from
    the expected local-daemon fallback path. Unified search always fails open;
    it reports only the former.
    """

    def __init__(self, message: str, *, report_failure: bool = True) -> None:
        super().__init__(message)
        self.report_failure = report_failure


def reranker_enabled() -> bool:
    """Return whether the deployment-wide reranker feature is enabled."""
    return env_truthy(env_str(RERANK_ENABLED_ENV_VAR, "true"))


def reranker_model_for_embedding(embedding_model: str) -> str:
    """Return the fixed reranker paired with an embedding image variant."""
    if embedding_model == "local/multilingual-e5-small":
        return MULTILINGUAL_RERANK_MODEL
    return ENGLISH_RERANK_MODEL


def reranker_revision(model: str) -> str:
    """Return the immutable Hugging Face revision for a supported reranker."""
    try:
        return RERANK_MODEL_REVISIONS[model]
    except KeyError as exc:
        raise CrossEncoderUnavailableError(
            f"Unsupported reranker model: {model}"
        ) from exc


def resolve_retrieval_floor(model: str, arm: str, configured: float | None) -> float:
    """Resolve a nullable organization floor without changing numeric overrides."""
    if configured is not None:
        return configured
    try:
        return RERANK_FLOOR_DEFAULTS[model][arm]
    except KeyError as exc:
        raise CrossEncoderUnavailableError(
            f"No calibrated retrieval floor for model={model!r}, arm={arm!r}"
        ) from exc


__all__ = [
    "RERANK_ENABLED_ENV_VAR",
    "ENGLISH_RERANK_MODEL",
    "MULTILINGUAL_RERANK_MODEL",
    "ENGLISH_RERANK_REVISION",
    "MULTILINGUAL_RERANK_REVISION",
    "RERANK_MODEL_REVISIONS",
    "RERANK_FLOOR_DEFAULTS",
    "RERANK_MODEL",
    "CrossEncoderUnavailableError",
    "reranker_enabled",
    "reranker_model_for_embedding",
    "reranker_revision",
    "resolve_retrieval_floor",
]
