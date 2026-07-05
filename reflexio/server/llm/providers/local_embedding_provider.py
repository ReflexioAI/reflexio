"""Local in-process embedder using Chroma's ONNX all-MiniLM-L6-v2.

Lets reflexio run without any external embedding API key. Activation is
opt-in via ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1`` and requires the
``chromadb`` pip package to be installed (we re-use its packaged ONNX
model + tokenizer rather than re-bundling them).

The model natively produces 384-dim vectors; reflexio's storage schema
expects 512 dims (``EMBEDDING_DIMENSIONS`` in the vec0 virtual tables).
We zero-pad each vector to 512 inside this module so the rest of
reflexio is unchanged. Cosine similarity is preserved on the 384-dim
subspace — safe as long as *all* embeddings in a given DB come from
this provider (mixing providers has always required a DB wipe).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows only
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX only
    msvcrt = None  # type: ignore[assignment]

_LOGGER = logging.getLogger(__name__)

_ENV_ENABLE = "CLAUDE_SMART_USE_LOCAL_EMBEDDING"
_MODEL_KEY = "local/minilm-l6-v2"

# Reflexio's storage schema (vec0 virtual tables) expects this dimension.
# MiniLM-L6-v2 natively produces 384; we pad with zeros to _TARGET_DIM.
_NATIVE_DIM = 384
_TARGET_DIM = 512

# Conservative character budget to stay under MiniLM's 256-token hard cap.
# ~4 chars/token in English prose; leave headroom so we never raise the
# ValueError that ONNXMiniLM_L6_V2 throws on over-length input.
_MAX_CHARS = 800

# Chroma keeps this file list private inside ONNXMiniLM_L6_V2's download helper.
# Keep this in sync with chromadb's all-MiniLM-L6-v2 extracted archive layout.
_MINILM_EXPECTED_FILES = (
    "config.json",
    "model.onnx",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
)


class LocalEmbedderError(RuntimeError):
    """Raised when the local embedder is called without chromadb installed."""


class LocalEmbedder:
    """Lazily-loaded singleton wrapping Chroma's ONNXMiniLM_L6_V2."""

    _instance: LocalEmbedder | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._ef: Any | None = None
        self._ef_lock = threading.Lock()

    @classmethod
    def get(cls) -> LocalEmbedder:
        """Return the process-wide singleton, constructing it on first use."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load(self) -> Any:
        """Lazy-import and instantiate the ONNX embedding function."""
        if self._ef is not None:
            return self._ef
        with self._ef_lock:
            if self._ef is not None:
                return self._ef
            try:
                from chromadb.utils.embedding_functions import (  # type: ignore[import-not-found]
                    ONNXMiniLM_L6_V2,
                )
            except ImportError as exc:
                raise LocalEmbedderError(
                    f"{_ENV_ENABLE}=1 but `chromadb` is not installed. "
                    "Install with `uv add chromadb` or `pip install chromadb`."
                ) from exc
            self._ef = ONNXMiniLM_L6_V2()
            _LOGGER.info(
                "Initialized local ONNX embedder (model=%s, cache=%s)",
                ONNXMiniLM_L6_V2.MODEL_NAME,
                ONNXMiniLM_L6_V2.DOWNLOAD_PATH,
            )
            return self._ef

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents, returning 512-dim padded vectors.

        Args:
            texts: Documents to embed. Each is truncated to ``_MAX_CHARS``
                characters to stay under the 256-token cap of MiniLM-L6-v2.

        Returns:
            list[list[float]]: One vector per input, each exactly
                ``_TARGET_DIM`` (512) floats with the last 128 positions
                zero-padded.
        """
        ef = self._load()
        safe_inputs = [(text or "")[:_MAX_CHARS] for text in texts]
        try:
            raw = ef(safe_inputs)
        except Exception as exc:  # noqa: BLE001 - Chroma raises varied cache errors.
            raw = self._retry_embed_after_cache_clear(ef, exc, safe_inputs)
        return [_pad(vec) for vec in raw]

    def _retry_embed_after_cache_clear(
        self, failed_ef: Any, exc: Exception, safe_inputs: list[str]
    ) -> Any:
        with self._ef_lock:
            if self._ef is not None and self._ef is not failed_ef:
                return self._ef(safe_inputs)

            embedding_cls = type(failed_ef)
            recovery = _recoverable_minilm_cache(embedding_cls, exc)
            if recovery is None:
                raise exc
            cache_path, cache_was_complete = recovery

            with _exclusive_file_lock(_minilm_cache_lock_path(cache_path)):
                if _should_clear_minilm_cache(
                    embedding_cls, cache_path, exc, cache_was_complete
                ):
                    shutil.rmtree(cache_path, ignore_errors=True)
                    recovery_action = "clearing"
                    _LOGGER.warning(
                        "Failed to use local MiniLM embedder; cleared cache %s "
                        "and retrying once",
                        cache_path,
                        exc_info=True,
                    )
                else:
                    recovery_action = "waiting for another process to refresh"
                    _LOGGER.info(
                        "MiniLM cache %s was refreshed by another process; retrying",
                        cache_path,
                    )
                try:
                    fresh_ef = embedding_cls()
                    self._ef = fresh_ef
                    return fresh_ef(safe_inputs)
                except Exception as retry_exc:
                    raise LocalEmbedderError(
                        "Local MiniLM cache recovery failed after "
                        f"{recovery_action} {cache_path}. Delete this cache "
                        "directory and restart Reflexio, or configure a cloud "
                        "embedding provider."
                    ) from retry_exc


def _pad(vec: Any) -> list[float]:
    """Zero-pad a 384-dim vector to ``_TARGET_DIM`` as a plain list[float]."""
    as_list = list(vec) if not isinstance(vec, list) else vec
    floats = [float(x) for x in as_list]
    if len(floats) == _TARGET_DIM:
        return floats
    if len(floats) > _TARGET_DIM:
        return floats[:_TARGET_DIM]
    return floats + [0.0] * (_TARGET_DIM - len(floats))


def _recoverable_minilm_cache(
    embedding_cls: Any, exc: Exception
) -> tuple[Path, bool] | None:
    cache_path = _minilm_cache_path(embedding_cls)
    if cache_path is None:
        return None
    cache_was_complete = _minilm_cache_complete(embedding_cls, cache_path)
    if cache_was_complete and not _complete_minilm_cache_error_identifies_cache(
        embedding_cls, cache_path, exc
    ):
        return None

    return cache_path, cache_was_complete


def _minilm_cache_path(embedding_cls: Any) -> Path | None:
    download_path = getattr(embedding_cls, "DOWNLOAD_PATH", None)
    if not download_path:
        return None

    model_name = getattr(embedding_cls, "MODEL_NAME", None)
    cache_path = Path(str(download_path)).expanduser()
    if not model_name or cache_path.name != model_name:
        return None

    return cache_path


def _minilm_cache_lock_path(cache_path: Path) -> Path:
    return cache_path.parent / f".{cache_path.name}.lock"


@contextmanager
def _exclusive_file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        lock_file.seek(0)
        if lock_file.read(1) == b"":
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)

        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:
            msvcrt.locking(  # type: ignore[reportAttributeAccessIssue]
                lock_file.fileno(),
                msvcrt.LK_LOCK,  # type: ignore[reportAttributeAccessIssue]
                1,
            )
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(  # type: ignore[reportAttributeAccessIssue]
                    lock_file.fileno(),
                    msvcrt.LK_UNLCK,  # type: ignore[reportAttributeAccessIssue]
                    1,
                )
        else:
            yield


def _should_clear_minilm_cache(
    embedding_cls: Any, cache_path: Path, exc: Exception, cache_was_complete: bool
) -> bool:
    if not cache_was_complete:
        return not _minilm_cache_complete(embedding_cls, cache_path)

    if not _minilm_cache_complete(embedding_cls, cache_path):
        return True

    return _complete_minilm_cache_error_identifies_cache(embedding_cls, cache_path, exc)


def _complete_minilm_cache_error_identifies_cache(
    embedding_cls: Any, cache_path: Path, exc: Exception
) -> bool:
    chain = _exception_chain(exc)
    archive_name = str(getattr(embedding_cls, "ARCHIVE_FILENAME", ""))
    messages = "\n".join(str(chain_exc) for chain_exc in chain)
    return (
        str(cache_path) in messages
        or bool(archive_name and archive_name in messages)
        or "does not match expected SHA256" in messages
    )


def _minilm_cache_complete(embedding_cls: Any, cache_path: Path) -> bool:
    extracted_folder = str(getattr(embedding_cls, "EXTRACTED_FOLDER_NAME", "onnx"))
    return all(
        (cache_path / extracted_folder / file_name).exists()
        for file_name in _MINILM_EXPECTED_FILES
    )


def _exception_chain(exc: Exception) -> list[BaseException]:
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


_REGISTERED = False


def is_chromadb_importable() -> bool:
    """Return True when the ``chromadb`` package is importable.

    Independent of :data:`_ENV_ENABLE` (``CLAUDE_SMART_USE_LOCAL_EMBEDDING``).
    Used by callers that want to know whether the local fallback is
    *possible* regardless of whether claude-smart has explicitly opted in.

    Returns:
        bool: True if ``importlib.util.find_spec("chromadb")`` finds the
            package, False otherwise.
    """
    return importlib.util.find_spec("chromadb") is not None


def register_if_chromadb_available() -> bool:
    """Make the local embedder available whenever ``chromadb`` imports.

    This is the env-var-independent companion to
    :func:`register_if_enabled`. Called once from ``litellm_client`` at
    module import so the local-embedder dispatch is wired up regardless
    of whether claude-smart's opt-in env var is set. Idempotent — safe
    to call more than once per process.

    The actual routing is done by a prefix check in
    ``LiteLLMClient.get_embedding(s)``; this function's job is to
    eagerly probe for ``chromadb`` and log a clear message.

    Returns:
        bool: True if the embedder is usable after this call, False
            when ``chromadb`` is not importable.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if not is_chromadb_importable():
        _LOGGER.debug("Local embedder not registered: `chromadb` is not installed.")
        return False
    _REGISTERED = True
    _LOGGER.debug("Registered local MiniLM embedding handler (model=%s)", _MODEL_KEY)
    return True


def register_if_enabled() -> bool:
    """Backwards-compatible alias for :func:`register_if_chromadb_available`.

    Historically gated on ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1``; now
    delegates to the chromadb-only check so the local embedder is also
    available as a silent fallback when no cloud embedder is configured.
    The claude-smart opt-in env var continues to drive
    :func:`is_local_embedder_available` (provider-priority ordering),
    which is the public contract the env var actually owns.

    Returns:
        bool: True if the embedder is usable after this call, False
            otherwise.
    """
    return register_if_chromadb_available()


def is_enabled() -> bool:
    """Return True when a previous registration call has succeeded.

    Returns:
        bool: True if the provider is currently registered and usable in
            this process, False otherwise.
    """
    return _REGISTERED


def is_local_embedder_available() -> bool:
    """Return True iff both the env flag is set and ``chromadb`` imports.

    Unlike :func:`is_enabled`, this does not require
    :func:`register_if_enabled` to have run. It is the predicate
    ``model_defaults.detect_available_providers`` uses to decide whether
    to surface ``"local"`` as an option.

    Returns:
        bool: True when ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1``
            AND ``chromadb`` is importable.
    """
    if os.environ.get(_ENV_ENABLE) != "1":
        return False
    return is_chromadb_importable()


__all__ = [
    "LocalEmbedder",
    "LocalEmbedderError",
    "is_chromadb_importable",
    "is_enabled",
    "is_local_embedder_available",
    "register_if_chromadb_available",
    "register_if_enabled",
]
