"""Local in-process all-MiniLM-L6-v2 embeddings using ONNX Runtime.

Lets reflexio run without any external embedding API key. Activation is
opt-in via ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1`` (or automatic fallback
when no cloud embedder is configured). Requires ONNX Runtime, tokenizers,
and NumPy; no vector database package or server is involved.

The model natively produces 384-dim vectors; reflexio's storage schema
expects 512 dims (``EMBEDDING_DIMENSIONS`` in the vec0 virtual tables).
We zero-pad each vector to 512 inside this module so the rest of
reflexio is unchanged. Cosine similarity is preserved on the 384-dim
subspace — safe as long as *all* embeddings in a given DB come from
this provider (mixing providers has always required a DB wipe).
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import tarfile
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, cast

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
# The tokenizer also truncates to 256 tokens, including special tokens.
_MAX_CHARS = 800

# Files in the checksum-pinned ONNX archive.
_MINILM_EXPECTED_FILES = (
    "config.json",
    "model.onnx",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.txt",
)


class LocalEmbedderError(RuntimeError):
    """Raised when the local embedder cannot load its dependencies or model."""


class ONNXMiniLM:
    """Minimal CPU inference adapter for the existing MiniLM model artifact.

    Retains the previous tokenizer, pooling, and normalization contract.
    Model provenance: https://github.com/chroma-core/onnx-embedding (Apache-2.0).
    The legacy cache location avoids downloading or reindexing on upgrade.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"
    DOWNLOAD_PATH = Path.home() / ".cache" / "chroma" / "onnx_models" / MODEL_NAME
    EXTRACTED_FOLDER_NAME = "onnx"
    ARCHIVE_FILENAME = "onnx.tar.gz"
    MODEL_DOWNLOAD_URL = (
        "https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz"
    )
    MODEL_SHA256 = "913d7300ceae3b2dbc2c50d1de4baacab4be7b9380491c27fab7418616a16ec3"

    def __init__(self) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._np = np
        self._ort = ort
        self._tokenizer_cls = Tokenizer
        self._tokenizer: Any = None
        self._session: Any = None

    def _download_model(self, cache: Path) -> None:
        """Verify before extracting; publish a complete directory under the lock."""
        import httpx

        with tempfile.TemporaryDirectory(prefix=".minilm-", dir=cache.parent) as tmp:
            staging = Path(tmp)
            archive = staging / self.ARCHIVE_FILENAME
            digest = hashlib.sha256()
            size = 0
            with httpx.stream("GET", self.MODEL_DOWNLOAD_URL, timeout=60) as response:
                response.raise_for_status()
                with archive.open("wb") as output:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > 128 * 1024 * 1024:
                            raise ValueError(
                                "MiniLM archive exceeds the download limit"
                            )
                        digest.update(chunk)
                        output.write(chunk)
            if digest.hexdigest() != self.MODEL_SHA256:
                raise ValueError("MiniLM archive does not match expected SHA256")

            extracted = staging / self.EXTRACTED_FOLDER_NAME
            extracted.mkdir()
            with tarfile.open(archive) as bundle:
                for name in _MINILM_EXPECTED_FILES:
                    member = bundle.getmember(f"{self.EXTRACTED_FOLDER_NAME}/{name}")
                    if not member.isfile():
                        raise ValueError(f"MiniLM archive member is not a file: {name}")
                    source = cast(BinaryIO, bundle.extractfile(member))
                    # Never extract archive paths or links onto the filesystem.
                    with source, (extracted / name).open("wb") as output:
                        shutil.copyfileobj(source, output)
            cache.mkdir(parents=True, exist_ok=True)
            target = cache / self.EXTRACTED_FOLDER_NAME
            if target.exists():
                shutil.rmtree(target)
            extracted.replace(target)

    def _load_model(self) -> None:
        cache = Path(self.DOWNLOAD_PATH)
        with _exclusive_file_lock(_minilm_cache_lock_path(cache)):
            if not _minilm_cache_complete(type(self), cache):
                self._download_model(cache)
            model_dir = cache / self.EXTRACTED_FOLDER_NAME
            tokenizer_path = model_dir / "tokenizer.json"
            try:
                tokenizer = self._tokenizer_cls.from_file(str(tokenizer_path))
            except Exception as exc:  # noqa: BLE001 - tokenizers uses generic Exception.
                raise LocalEmbedderError(f"Failed to load {tokenizer_path}") from exc
            tokenizer.enable_truncation(max_length=256)
            tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)  # noqa: S106 - tokenizer marker
            options = self._ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            options.graph_optimization_level = (
                self._ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self._session = self._ort.InferenceSession(
                str(model_dir / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            self._tokenizer = tokenizer

    def __call__(self, documents: list[str]) -> list[list[float]]:
        if not documents:
            return []
        if self._session is None:
            self._load_model()
        np = self._np
        vectors: list[list[float]] = []
        for start in range(0, len(documents), 32):
            encoded = [
                self._tokenizer.encode(text) for text in documents[start : start + 32]
            ]
            ids = np.array([item.ids for item in encoded], dtype=np.int64)
            attention = np.array(
                [item.attention_mask for item in encoded], dtype=np.int64
            )
            hidden = self._session.run(
                None,
                {
                    "input_ids": ids,
                    "attention_mask": attention,
                    "token_type_ids": np.zeros_like(ids),
                },
            )[0]
            mask = np.broadcast_to(np.expand_dims(attention, -1), hidden.shape)
            pooled = np.sum(hidden * mask, 1) / np.clip(mask.sum(1), 1e-9, None)
            norms = np.linalg.norm(pooled, axis=1)
            norms[norms == 0] = 1e-12
            vectors.extend((pooled / norms[:, None]).astype(np.float32).tolist())
        return vectors


class LocalEmbedder:
    """Lazily-loaded singleton wrapping direct ONNX MiniLM inference."""

    _instance: LocalEmbedder | None = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._ef: Any | None = None
        # Guards lazy construction / cache-recovery swaps of ``self._ef``.
        self._ef_lock = threading.Lock()
        # Serializes the actual encode call. The shared ONNX embedding function
        # is NOT thread-safe — concurrent ``ef(...)`` calls interleave and
        # corrupt each other's padding/attention buffers (observed in prod as
        # tensor-shape mismatches). ``threading.Lock`` is non-reentrant, so this
        # is a SEPARATE lock from ``_ef_lock`` (the recovery path re-acquires
        # ``_ef_lock`` while we hold this one): fixed order is ``_encode_lock``
        # outer, ``_ef_lock`` inner, which cannot deadlock. NOTE: this lock is
        # not FIFO-fair — do not turn it into a fairness queue without measuring.
        self._encode_lock = threading.Lock()

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
                self._ef = ONNXMiniLM()
            except ImportError as exc:
                raise LocalEmbedderError(
                    "Local MiniLM dependencies are missing. Install with "
                    "`pip install onnxruntime tokenizers numpy`."
                ) from exc
            _LOGGER.info(
                "Initialized local ONNX embedder (model=%s, cache=%s)",
                ONNXMiniLM.MODEL_NAME,
                ONNXMiniLM.DOWNLOAD_PATH,
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
        safe_inputs = [(text or "")[:_MAX_CHARS] for text in texts]
        # Serialize the encode: the shared ONNX embedder is not thread-safe.
        # ``_load()`` and ``_retry_embed_after_cache_clear()`` both acquire
        # ``_ef_lock`` internally; running them under ``_encode_lock`` fixes the
        # lock order (``_encode_lock`` outer) so recovery can never deadlock.
        with self._encode_lock:
            ef = self._load()
            try:
                raw = ef(safe_inputs)
            except Exception as exc:  # noqa: BLE001 - Runtime/tokenizer cache errors vary.
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
            # The adapter takes the same file lock when loading/downloading.
            # Release it first to avoid a nested flock deadlock.
            try:
                fresh_ef = embedding_cls()
                self._ef = fresh_ef
                return fresh_ef(safe_inputs)
            except Exception as retry_exc:
                raise LocalEmbedderError(
                    "Local MiniLM cache recovery failed after "
                    f"{recovery_action} {cache_path}. Delete this cache "
                    "directory, restart Reflexio, and retry local embedding."
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
    cache_is_complete = _minilm_cache_complete(embedding_cls, cache_path)
    if not cache_is_complete:
        return True

    if not cache_was_complete:
        return False

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


def are_local_embedding_dependencies_available() -> bool:
    """Check all ONNX dependencies without importing heavyweight modules."""
    return all(
        importlib.util.find_spec(package) is not None
        for package in ("onnxruntime", "tokenizers", "numpy")
    )


def register_if_available() -> bool:
    """Set the legacy registration flag when ONNX dependencies are discoverable.

    Idempotent and independent of the opt-in environment flag. The inference
    service loads ``LocalEmbedder`` directly and does not require registration.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    if not are_local_embedding_dependencies_available():
        _LOGGER.debug(
            "Local embedder not registered: ONNX dependencies are not installed."
        )
        return False
    _REGISTERED = True
    _LOGGER.debug("Registered local MiniLM embedding handler (model=%s)", _MODEL_KEY)
    return True


def register_if_enabled() -> bool:
    """Backwards-compatible alias for :func:`register_if_available`.

    Historically gated on ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1``; now
    delegates to the dependency-only check so the local embedder is also
    available as a silent fallback when no cloud embedder is configured.
    The claude-smart opt-in env var continues to drive
    :func:`is_local_embedder_available` (provider-priority ordering),
    which is the public contract the env var actually owns.

    Returns:
        bool: True if the embedder is usable after this call, False
            otherwise.
    """
    return register_if_available()


def is_enabled() -> bool:
    """Return True when a previous registration call has succeeded.

    Returns:
        bool: True if the provider is currently registered and usable in
            this process, False otherwise.
    """
    return _REGISTERED


def is_local_embedder_available() -> bool:
    """Return True iff both the env flag is set and the ONNX dependencies are installed.

    Unlike :func:`is_enabled`, this does not require
    :func:`register_if_enabled` to have run. It is the predicate
    ``model_defaults.detect_available_providers`` uses to decide whether
    to surface ``"local"`` as an option.

    Returns:
        bool: True when ``CLAUDE_SMART_USE_LOCAL_EMBEDDING=1``
            AND the ONNX dependencies are importable.
    """
    if os.environ.get(_ENV_ENABLE) != "1":
        return False
    return are_local_embedding_dependencies_available()


__all__ = [
    "LocalEmbedder",
    "LocalEmbedderError",
    "are_local_embedding_dependencies_available",
    "is_enabled",
    "is_local_embedder_available",
    "register_if_available",
    "register_if_enabled",
]
