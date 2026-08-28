"""Tests for the local ONNX embedding provider."""

from __future__ import annotations

import hashlib
import io
import sys
import tarfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm import _litellm_embedding
from reflexio.server.llm.providers import local_embedding_provider as lep
from reflexio.server.llm.providers.local_embedding_provider import (
    LocalEmbedder,
    are_local_embedding_dependencies_available,
    is_local_embedder_available,
    register_if_available,
    register_if_enabled,
)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Each test starts with a fresh registration flag + clean singleton."""
    lep._REGISTERED = False
    LocalEmbedder._instance = None


def _fake_minilm_module(return_vec: list[float] | None = None) -> MagicMock:
    """Build a stand-in MiniLM inference adapter.

    Args:
        return_vec: 384-dim vector the mocked ``ONNXMiniLM`` will
            return for every input. Defaults to a simple ramp.

    Returns:
        MagicMock: Parent module object with the ``ONNXMiniLM``
            class attached for injection into the provider.
    """
    if return_vec is None:
        return_vec = [float(i) / 384.0 for i in range(384)]

    ef_instance = MagicMock()
    ef_instance.side_effect = lambda docs: [list(return_vec) for _ in docs]

    ef_class = MagicMock(return_value=ef_instance)
    ef_class.MODEL_NAME = "all-MiniLM-L6-v2"
    ef_class.DOWNLOAD_PATH = "/fake/cache"

    mod = MagicMock()
    mod.ONNXMiniLM = ef_class
    return mod


def _install_fake_minilm(
    monkeypatch: pytest.MonkeyPatch, vec: list[float] | None = None
) -> MagicMock:
    """Inject a fake MiniLM inference adapter."""
    fake = _fake_minilm_module(vec)
    monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)
    return fake


class TestAvailability:
    @pytest.mark.parametrize("missing", ["onnxruntime", "tokenizers", "numpy"])
    def test_each_inference_dependency_is_required(self, monkeypatch, missing):
        monkeypatch.setattr(
            lep.importlib.util,
            "find_spec",
            lambda name: None if name == missing else object(),
        )
        assert not are_local_embedding_dependencies_available()

    def test_not_available_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        assert is_local_embedder_available() is False

    def test_not_available_without_local_dependencies(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert is_local_embedder_available() is False

    def test_available_when_both_conditions_met(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert is_local_embedder_available() is True


class TestIsLocalDependenciesAvailable:
    def test_are_local_embedding_dependencies_available_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns True when ``importlib.util.find_spec`` finds ONNX dependencies."""
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert are_local_embedding_dependencies_available() is True

    def test_are_local_embedding_dependencies_available_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when ``importlib.util.find_spec`` returns None."""
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert are_local_embedding_dependencies_available() is False

    def test_are_local_embedding_dependencies_available_independent_of_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importability does not depend on ``CLAUDE_SMART_USE_LOCAL_EMBEDDING``."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert are_local_embedding_dependencies_available() is True


class TestRegisterIfAvailable:
    def test_register_if_available_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registers whenever ONNX dependencies import — env var is not required."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_available() is True
        assert lep.is_enabled() is True

    def test_register_if_available_local_dependencies_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Does not register when ONNX dependencies are not importable."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert register_if_available() is False
        assert lep.is_enabled() is False

    def test_register_if_available_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_available() is True
        assert register_if_available() is True


class TestRegisterIfEnabled:
    """``register_if_enabled`` is now an alias for ``register_if_available``.

    The legacy env-var gate (``CLAUDE_SMART_USE_LOCAL_EMBEDDING``) only
    drives provider-priority ordering via ``is_local_embedder_available``;
    registration of the LiteLLM dispatch hook depends solely on whether
    `ONNX dependencies` imports.
    """

    def test_local_dependencies_missing_does_not_register(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert register_if_enabled() is False
        assert lep.is_enabled() is False

    def test_registers_with_local_dependencies_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var is no longer required — ONNX dependencies alone are sufficient."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_enabled() is True
        assert lep.is_enabled() is True

    def test_registers_when_env_and_local_dependencies_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_enabled() is True
        assert lep.is_enabled() is True

    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_enabled() is True
        assert register_if_enabled() is True  # second call no-ops cleanly


class TestEmbedPadding:
    def test_real_minilm_attrs_match_cache_recovery_contract(self) -> None:
        minilm_cls = lep.ONNXMiniLM

        assert minilm_cls.MODEL_NAME == "all-MiniLM-L6-v2"
        assert minilm_cls.DOWNLOAD_PATH.name == minilm_cls.MODEL_NAME
        assert minilm_cls.EXTRACTED_FOLDER_NAME == "onnx"
        assert minilm_cls.ARCHIVE_FILENAME == "onnx.tar.gz"

    @pytest.mark.parametrize("cache_shape", ["partial", "complete"])
    def test_corrupt_minilm_cache_is_cleared_and_retried_once(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path, cache_shape: str
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        extracted_dir = cache_dir / "onnx"
        cache_dir.mkdir(parents=True)
        if cache_shape == "complete":
            extracted_dir.mkdir()
            for file_name in lep._MINILM_EXPECTED_FILES:
                (extracted_dir / file_name).write_text("present")
            first_error: Exception = RuntimeError(
                f"Failed to load {extracted_dir / 'model.onnx'}"
            )
        else:
            (cache_dir / "partial-download").write_text("corrupt")
            first_error = tarfile.ReadError("bad checksum")
        attempts = 0

        class FailingThenWorkingMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, docs: list[str]) -> list[list[float]]:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise first_error
                return [[0.1] * 384 for _ in docs]

        fake = MagicMock()
        fake.ONNXMiniLM = FailingThenWorkingMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)

        result = LocalEmbedder.get().embed(["hello"])

        assert attempts == 2
        assert result == [[0.1] * 384 + [0.0] * 128]
        assert not cache_dir.exists()

    def test_constructor_failure_does_not_clear_cache(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "model").write_text("valid")

        class FailingConstructorMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)

            def __init__(self) -> None:
                raise ValueError("The onnxruntime python package is not installed")

        fake = MagicMock()
        fake.ONNXMiniLM = FailingConstructorMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)

        with pytest.raises(ValueError, match="onnxruntime"):
            LocalEmbedder.get().embed(["hello"])

        assert cache_dir.exists()
        assert (cache_dir / "model").exists()

    def test_retry_reuses_embedder_recovered_by_another_thread(self, tmp_path) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "partial-download").write_text("corrupt")

        class MiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

        failed_ef = MagicMock()
        recovered_ef = MagicMock(return_value=[[0.3] * 384])
        embedder = LocalEmbedder()
        embedder._ef = recovered_ef

        result = embedder._retry_embed_after_cache_clear(
            failed_ef, tarfile.ReadError("bad checksum"), ["hello"]
        )

        assert result == [[0.3] * 384]
        recovered_ef.assert_called_once_with(["hello"])
        assert cache_dir.exists()
        assert (cache_dir / "partial-download").exists()

    def test_cache_clear_is_locked_and_retry_can_reacquire_lock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "partial-download").write_text("corrupt")
        lock_active = False
        attempts = 0

        @contextmanager
        def fake_file_lock(_lock_path) -> Iterator[None]:
            nonlocal lock_active
            lock_active = True
            try:
                yield
            finally:
                lock_active = False

        def checked_rmtree(*args, **kwargs) -> None:
            assert lock_active
            shutil_rmtree(*args, **kwargs)

        class FailingThenWorkingMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, docs: list[str]) -> list[list[float]]:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise tarfile.ReadError("bad checksum")
                assert not lock_active  # adapter must acquire its own download lock
                return [[0.2] * 384 for _ in docs]

        fake = MagicMock()
        fake.ONNXMiniLM = FailingThenWorkingMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)
        monkeypatch.setattr(lep, "_exclusive_file_lock", fake_file_lock)
        shutil_rmtree = lep.shutil.rmtree
        monkeypatch.setattr(lep.shutil, "rmtree", checked_rmtree)

        result = LocalEmbedder.get().embed(["hello"])

        assert attempts == 2
        assert result == [[0.2] * 384 + [0.0] * 128]

    def test_retry_exhaustion_includes_manual_cache_recovery_hint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "partial-download").write_text("corrupt")
        attempts = 0

        class AlwaysFailingMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, _docs: list[str]) -> list[list[float]]:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise tarfile.ReadError("bad checksum")
                raise RuntimeError("still corrupt")

        fake = MagicMock()
        fake.ONNXMiniLM = AlwaysFailingMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)

        with pytest.raises(lep.LocalEmbedderError) as exc_info:
            LocalEmbedder.get().embed(["hello"])

        message = str(exc_info.value)
        assert str(cache_dir) in message
        assert (
            "Delete this cache directory, restart Reflexio, and retry local embedding"
        ) in message
        assert attempts == 2

    def test_retry_keeps_cache_refreshed_by_another_process(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        extracted_dir = cache_dir / "onnx"
        cache_dir.mkdir(parents=True)
        (cache_dir / "partial-download").write_text("corrupt")
        attempts = 0

        @contextmanager
        def fake_file_lock(_lock_path) -> Iterator[None]:
            extracted_dir.mkdir(parents=True)
            for file_name in lep._MINILM_EXPECTED_FILES:
                (extracted_dir / file_name).write_text("present")
            yield

        class RecoveredMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, docs: list[str]) -> list[list[float]]:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise tarfile.ReadError("bad checksum")
                return [[0.4] * 384 for _ in docs]

        fake = MagicMock()
        fake.ONNXMiniLM = RecoveredMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)
        monkeypatch.setattr(lep, "_exclusive_file_lock", fake_file_lock)
        rmtree = MagicMock()
        monkeypatch.setattr(lep.shutil, "rmtree", rmtree)

        result = LocalEmbedder.get().embed(["hello"])

        assert result == [[0.4] * 384 + [0.0] * 128]
        assert attempts == 2
        rmtree.assert_not_called()
        assert extracted_dir.exists()

    def test_complete_cache_with_unrelated_tar_error_is_not_recoverable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        extracted_dir = cache_dir / "onnx"
        extracted_dir.mkdir(parents=True)
        for file_name in lep._MINILM_EXPECTED_FILES:
            (extracted_dir / file_name).write_text("present")

        class UnrelatedTarErrorMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, _docs: list[str]) -> list[list[float]]:
                raise tarfile.ReadError("bad checksum")

        fake = MagicMock()
        fake.ONNXMiniLM = UnrelatedTarErrorMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)
        rmtree = MagicMock()
        monkeypatch.setattr(lep.shutil, "rmtree", rmtree)

        with pytest.raises(tarfile.ReadError, match="bad checksum"):
            LocalEmbedder.get().embed(["hello"])

        rmtree.assert_not_called()
        assert extracted_dir.exists()

    def test_384_vector_padded_to_512(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_minilm(monkeypatch)

        result = LocalEmbedder.get().embed(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 512
        # First 384 positions are the native vector; last 128 are zero-padded.
        assert all(x == 0.0 for x in result[0][384:])
        assert any(x != 0.0 for x in result[0][:384])

    def test_batch_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_minilm(monkeypatch)

        result = LocalEmbedder.get().embed(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(vec) == 512 for vec in result)

    def test_long_input_truncated_by_char_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Very long strings are clipped to stay under MiniLM's 256-token cap."""
        fake = _install_fake_minilm(monkeypatch)

        long_text = "x" * 10_000
        LocalEmbedder.get().embed([long_text])

        # The mock embedding function receives the truncated input,
        # not the original 10_000 characters.
        ef_instance = fake.ONNXMiniLM.return_value
        call_args = ef_instance.call_args.args[0]
        assert len(call_args[0]) <= 1000

    def test_missing_local_dependencies_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "onnxruntime", None)

        with pytest.raises(lep.LocalEmbedderError, match="dependencies are missing"):
            LocalEmbedder.get().embed(["hi"])


class TestModelDownload:
    @pytest.fixture
    def model_archive(self, monkeypatch, tmp_path):
        """Serve a tiny real tar archive through the HTTP streaming interface."""
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            for name in lep._MINILM_EXPECTED_FILES:
                content = f"fixture-{name}".encode()
                member = tarfile.TarInfo(f"onnx/{name}")
                member.size = len(content)
                bundle.addfile(member, io.BytesIO(content))
            # Unselected archive paths must never be written to disk.
            member = tarfile.TarInfo("../../outside")
            member.size = 1
            bundle.addfile(member, io.BytesIO(b"x"))
        payload = archive.getvalue()
        response = MagicMock()
        response.iter_bytes.side_effect = lambda _size: iter([payload])
        stream = MagicMock()
        stream.return_value.__enter__.return_value = response
        monkeypatch.setattr("httpx.stream", stream)
        monkeypatch.setattr(
            lep.ONNXMiniLM, "MODEL_SHA256", hashlib.sha256(payload).hexdigest()
        )
        cache = tmp_path / lep.ONNXMiniLM.MODEL_NAME
        monkeypatch.setattr(lep.ONNXMiniLM, "DOWNLOAD_PATH", cache)
        return lep.ONNXMiniLM(), cache, response, stream

    def test_verified_archive_extracts_only_expected_regular_files(
        self, model_archive, tmp_path
    ):
        engine, cache, response, _ = model_archive
        engine._download_model(cache)
        response.raise_for_status.assert_called_once()
        assert lep._minilm_cache_complete(type(engine), cache)
        assert (cache / "onnx" / "model.onnx").read_text() == "fixture-model.onnx"
        assert not (tmp_path / "outside").exists()
        assert not list(tmp_path.glob(".minilm-*"))

    def test_bad_checksum_never_publishes_cache(
        self, model_archive, monkeypatch, tmp_path
    ):
        engine, cache, _, _ = model_archive
        monkeypatch.setattr(engine, "MODEL_SHA256", "0" * 64)
        with pytest.raises(ValueError, match="SHA256"):
            engine._download_model(cache)
        assert not cache.exists()
        assert not list(tmp_path.glob(".minilm-*"))

    def test_interrupted_download_does_not_publish_partial_cache(
        self, model_archive, tmp_path
    ):
        import httpx

        engine, cache, response, _ = model_archive

        def interrupted(_size):
            yield b"partial"
            raise httpx.ReadError("connection lost")

        response.iter_bytes.side_effect = interrupted
        with pytest.raises(httpx.ReadError):
            engine._download_model(cache)
        assert not cache.exists()
        assert not list(tmp_path.glob(".minilm-*"))

    def test_links_in_expected_members_are_rejected(self, model_archive, monkeypatch):
        engine, cache, response, _ = model_archive
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
            member = tarfile.TarInfo("onnx/config.json")
            member.type = tarfile.SYMTYPE
            member.linkname = "../../outside"
            bundle.addfile(member)
        payload = archive.getvalue()
        monkeypatch.setattr(engine, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
        response.iter_bytes.side_effect = lambda _size: iter([payload])
        with pytest.raises(ValueError, match="not a file"):
            engine._download_model(cache)
        assert not cache.exists()

    def test_concurrent_cold_load_downloads_once(self, model_archive, monkeypatch):
        from concurrent.futures import ThreadPoolExecutor

        engine, cache, _, stream = model_archive
        tokenizer_cls = MagicMock()
        monkeypatch.setattr(engine._ort, "InferenceSession", MagicMock())
        other = lep.ONNXMiniLM()
        for adapter in (engine, other):
            monkeypatch.setattr(adapter, "_tokenizer_cls", tokenizer_cls)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda adapter: adapter._load_model(), [engine, other]))
        assert stream.call_count == 1
        assert lep._minilm_cache_complete(type(engine), cache)
        tokenizer_cls.from_file.return_value.enable_truncation.assert_called_with(
            max_length=256
        )
        tokenizer_cls.from_file.return_value.enable_padding.assert_called_with(
            pad_id=0, pad_token="[PAD]", length=256
        )


class TestEncodeSerialization:
    """The shared ONNX embedding function is not thread-safe.

    ``LocalEmbedder.embed()`` must serialize the actual encode call so
    concurrent publishes can't interleave and corrupt shared buffers.
    """

    def test_encode_is_serialized_across_threads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """8 threads on ONE embedder never enter the encode callable at once.

        Reverting the ``self._encode_lock`` guard in ``embed()`` makes this
        fail: the stub records/raises on concurrent entry.
        """
        state = {"active": False, "reentered": False, "calls": 0}
        guard = threading.Lock()

        def encode(docs: list[str]) -> list[list[float]]:
            with guard:
                if state["active"]:
                    state["reentered"] = True
                    raise AssertionError("embedder entered by two threads at once")
                state["active"] = True
                state["calls"] += 1
            try:
                time.sleep(0.02)  # widen the race window
            finally:
                with guard:
                    state["active"] = False
            return [[0.1] * 384 for _ in docs]

        ef_instance = MagicMock()
        ef_instance.side_effect = encode
        ef_class = MagicMock(return_value=ef_instance)
        ef_class.MODEL_NAME = "all-MiniLM-L6-v2"
        ef_class.DOWNLOAD_PATH = "/fake/cache"
        fake = MagicMock()
        fake.ONNXMiniLM = ef_class
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)

        embedder = LocalEmbedder.get()
        payloads = [
            ["short"],
            ["x" * 500],
            ["a", "b", "c"],
            ["", "y" * 50],
            ["one"],
            ["p", "q"],
            ["z" * 700],
            ["m", "n", "o", "p"],
        ]
        results: list[list[list[float]]] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def worker(texts: list[str]) -> None:
            try:
                out = embedder.embed(texts)
            except BaseException as exc:  # noqa: BLE001
                with results_lock:
                    errors.append(exc)
                return
            with results_lock:
                results.append(out)

        threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert state["reentered"] is False
        assert state["calls"] == len(payloads)
        assert len(results) == len(payloads)
        assert all(len(vec) == 512 for out in results for vec in out)

    def test_concurrent_cache_recovery_does_not_deadlock(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Recovery path (``_ef_lock``) nests safely inside ``_encode_lock``.

        The recovery path re-embeds while holding ``_ef_lock``; because it runs
        under the outer ``_encode_lock``, no encode call ever overlaps and the
        two locks never deadlock. Drives it concurrently to prove both.
        """
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        cache_dir.mkdir(parents=True)
        (cache_dir / "partial-download").write_text("corrupt")

        guard = threading.Lock()
        state = {"active": False, "reentered": False, "attempts": 0}

        class FlakyMiniLM:
            MODEL_NAME = "all-MiniLM-L6-v2"
            DOWNLOAD_PATH = str(cache_dir)
            EXTRACTED_FOLDER_NAME = "onnx"
            ARCHIVE_FILENAME = "onnx.tar.gz"

            def __call__(self, docs: list[str]) -> list[list[float]]:
                with guard:
                    if state["active"]:
                        state["reentered"] = True
                        raise AssertionError("encode entered by two threads at once")
                    state["active"] = True
                    state["attempts"] += 1
                    first = state["attempts"] == 1
                try:
                    time.sleep(0.01)
                    if first:
                        raise tarfile.ReadError("bad checksum")
                finally:
                    with guard:
                        state["active"] = False
                return [[0.2] * 384 for _ in docs]

        fake = MagicMock()
        fake.ONNXMiniLM = FlakyMiniLM
        monkeypatch.setattr(lep, "ONNXMiniLM", fake.ONNXMiniLM)

        embedder = LocalEmbedder.get()
        results: list[list[list[float]]] = []
        errors: list[BaseException] = []
        results_lock = threading.Lock()

        def worker() -> None:
            try:
                out = embedder.embed(["hello", "world"])
            except BaseException as exc:  # noqa: BLE001
                with results_lock:
                    errors.append(exc)
                return
            with results_lock:
                results.append(out)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)

        # No thread is stuck: the _encode_lock/_ef_lock ordering can't deadlock.
        assert all(not thread.is_alive() for thread in threads)
        assert errors == []
        assert state["reentered"] is False
        assert len(results) == 6
        assert all(len(vec) == 512 for out in results for vec in out)


class TestLiteLLMClientShortCircuit:
    """When model starts with ``local/`` and the provider is enabled,
    LiteLLMClient.get_embedding(s) must delegate to LocalEmbedder and
    never call ``litellm.embedding``."""

    def test_get_embedding_routes_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        # CLAUDE_SMART_USE_LOCAL_EMBEDDING=1 selects the local embedding *service*
        # (daemon) mode (see embedding_provider_mode). The request must route to
        # that service and never reach litellm. Mock the service call so the test
        # is hermetic (no running daemon required).
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda texts, **_kwargs: [[0.0] * 512 for _ in texts],
        )

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embedding("hello", model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 512

    def test_get_embeddings_routes_to_local(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        # CLAUDE_SMART_USE_LOCAL_EMBEDDING=1 routes the batch path to the local
        # embedding *service* (daemon) too. Mock the service call for hermeticity.
        monkeypatch.setenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", "1")
        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda texts, **_kwargs: [[0.0] * 512 for _ in texts],
        )

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embeddings(["a", "b"], model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 2
        assert all(len(vec) == 512 for vec in result)

    def test_non_local_model_still_calls_litellm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sanity: unchanged model strings flow through the normal path."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        # Force "cloud" provider mode so the embedding-service short-circuit
        # in get_embedding() doesn't intercept the call before litellm runs.
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        fake_response = MagicMock()
        fake_response.data = [{"embedding": [0.1] * 512, "index": 0}]

        with patch.object(
            litellm_client.litellm, "embedding", return_value=fake_response
        ) as mock_embedding:
            client.get_embedding("hello", model="text-embedding-3-small")

        mock_embedding.assert_called_once()

    def test_get_embedding_routes_to_local_without_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Layer A path 3: ONNX dependencies available + env var UNSET still routes local/*."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda *_args, **_kwargs: [[0.1] * 512],
        )

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embedding("hello", model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 512

    def test_get_embeddings_routes_to_local_without_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Layer A path 3 on the batch path: ONNX dependencies only is enough."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda *_args, **_kwargs: [[0.1] * 512, [0.2] * 512],
        )

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embeddings(["a", "b"], model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 2
        assert all(len(vec) == 512 for vec in result)

    def test_get_embedding_local_service_failure_does_not_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When local inference is down, do not construct an in-process model."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import (
            LiteLLMClient,
            LiteLLMConfig,
        )

        monkeypatch.setattr(
            _litellm_embedding,
            "get_service_embeddings",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("service unavailable")
            ),
        )

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with (
            patch.object(litellm_client.litellm, "embedding") as mock_embedding,
            pytest.raises(RuntimeError, match="service unavailable"),
        ):
            client.get_embedding("hello", model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
