"""Tests for the local ONNX embedding provider."""

from __future__ import annotations

import sys
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from reflexio.server.llm import _litellm_embedding
from reflexio.server.llm.providers import local_embedding_provider as lep
from reflexio.server.llm.providers.local_embedding_provider import (
    LocalEmbedder,
    is_chromadb_importable,
    is_local_embedder_available,
    register_if_chromadb_available,
    register_if_enabled,
)


@pytest.fixture(autouse=True)
def _reset_state() -> None:
    """Each test starts with a fresh registration flag + clean singleton."""
    lep._REGISTERED = False
    LocalEmbedder._instance = None


def _fake_chroma_module(return_vec: list[float] | None = None) -> MagicMock:
    """Build a stand-in ``chromadb.utils.embedding_functions`` module.

    Args:
        return_vec: 384-dim vector the mocked ``ONNXMiniLM_L6_V2`` will
            return for every input. Defaults to a simple ramp.

    Returns:
        MagicMock: Parent module object with the ``ONNXMiniLM_L6_V2``
            class attached and ready to be registered with
            ``sys.modules``.
    """
    if return_vec is None:
        return_vec = [float(i) / 384.0 for i in range(384)]

    ef_instance = MagicMock()
    ef_instance.side_effect = lambda docs: [list(return_vec) for _ in docs]

    ef_class = MagicMock(return_value=ef_instance)
    ef_class.MODEL_NAME = "all-MiniLM-L6-v2"
    ef_class.DOWNLOAD_PATH = "/fake/cache"

    mod = MagicMock()
    mod.ONNXMiniLM_L6_V2 = ef_class
    return mod


def _install_fake_chroma(
    monkeypatch: pytest.MonkeyPatch, vec: list[float] | None = None
) -> MagicMock:
    """Register a fake chromadb.utils.embedding_functions in sys.modules."""
    fake = _fake_chroma_module(vec)
    # Create minimal chromadb parent packages so the provider's relative
    # import works regardless of whether the real chromadb is installed.
    monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
    monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
    monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)
    return fake


class TestAvailability:
    def test_not_available_without_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        assert is_local_embedder_available() is False

    def test_not_available_without_chromadb(
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


class TestIsChromadbImportable:
    def test_is_chromadb_importable_when_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns True when ``importlib.util.find_spec`` finds chromadb."""
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert is_chromadb_importable() is True

    def test_is_chromadb_importable_when_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns False when ``importlib.util.find_spec`` returns None."""
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert is_chromadb_importable() is False

    def test_is_chromadb_importable_independent_of_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Importability does not depend on ``CLAUDE_SMART_USE_LOCAL_EMBEDDING``."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert is_chromadb_importable() is True


class TestRegisterIfChromadbAvailable:
    def test_register_if_chromadb_available_no_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registers whenever chromadb imports — env var is not required."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_chromadb_available() is True
        assert lep.is_enabled() is True

    def test_register_if_chromadb_available_chromadb_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Does not register when chromadb is not importable."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert register_if_chromadb_available() is False
        assert lep.is_enabled() is False

    def test_register_if_chromadb_available_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_chromadb_available() is True
        assert register_if_chromadb_available() is True


class TestRegisterIfEnabled:
    """``register_if_enabled`` is now an alias for ``register_if_chromadb_available``.

    The legacy env-var gate (``CLAUDE_SMART_USE_LOCAL_EMBEDDING``) only
    drives provider-priority ordering via ``is_local_embedder_available``;
    registration of the LiteLLM dispatch hook depends solely on whether
    ``chromadb`` imports.
    """

    def test_chromadb_missing_does_not_register(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)
        assert register_if_enabled() is False
        assert lep.is_enabled() is False

    def test_registers_with_chromadb_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Env var is no longer required — chromadb alone is sufficient."""
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())
        assert register_if_enabled() is True
        assert lep.is_enabled() is True

    def test_registers_when_env_and_chromadb_set(
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
    def test_real_chroma_minilm_attrs_match_cache_recovery_contract(self) -> None:
        embedding_functions = pytest.importorskip("chromadb.utils.embedding_functions")
        minilm_cls = embedding_functions.ONNXMiniLM_L6_V2

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
        fake.ONNXMiniLM_L6_V2 = FailingThenWorkingMiniLM
        monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)

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
        fake.ONNXMiniLM_L6_V2 = FailingConstructorMiniLM
        monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)

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

    def test_cache_clear_and_retry_run_under_filesystem_lock(
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
                assert lock_active
                return [[0.2] * 384 for _ in docs]

        fake = MagicMock()
        fake.ONNXMiniLM_L6_V2 = FailingThenWorkingMiniLM
        monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)
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
        fake.ONNXMiniLM_L6_V2 = AlwaysFailingMiniLM
        monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)

        with pytest.raises(lep.LocalEmbedderError) as exc_info:
            LocalEmbedder.get().embed(["hello"])

        message = str(exc_info.value)
        assert str(cache_dir) in message
        assert "Delete this cache directory and restart Reflexio" in message
        assert "cloud embedding provider" in message
        assert attempts == 2

    def test_retry_keeps_cache_refreshed_by_another_process(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        cache_dir = tmp_path / "chroma" / "onnx_models" / "all-MiniLM-L6-v2"
        extracted_dir = cache_dir / "onnx"
        extracted_dir.mkdir(parents=True)
        for file_name in lep._MINILM_EXPECTED_FILES:
            (extracted_dir / file_name).write_text("present")
        attempts = 0

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
        fake.ONNXMiniLM_L6_V2 = RecoveredMiniLM
        monkeypatch.setitem(sys.modules, "chromadb", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils", MagicMock())
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", fake)
        rmtree = MagicMock()
        monkeypatch.setattr(lep.shutil, "rmtree", rmtree)

        result = LocalEmbedder.get().embed(["hello"])

        assert result == [[0.4] * 384 + [0.0] * 128]
        assert attempts == 2
        rmtree.assert_not_called()
        assert extracted_dir.exists()

    def test_384_vector_padded_to_512(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_chroma(monkeypatch)

        result = LocalEmbedder.get().embed(["hello"])

        assert len(result) == 1
        assert len(result[0]) == 512
        # First 384 positions are the native vector; last 128 are zero-padded.
        assert all(x == 0.0 for x in result[0][384:])
        assert any(x != 0.0 for x in result[0][:384])

    def test_batch_embedding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_chroma(monkeypatch)

        result = LocalEmbedder.get().embed(["a", "b", "c"])

        assert len(result) == 3
        assert all(len(vec) == 512 for vec in result)

    def test_long_input_truncated_by_char_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Very long strings are clipped to stay under MiniLM's 256-token cap."""
        fake = _install_fake_chroma(monkeypatch)

        long_text = "x" * 10_000
        LocalEmbedder.get().embed([long_text])

        # The mock embedding function receives the truncated input,
        # not the original 10_000 characters.
        ef_instance = fake.ONNXMiniLM_L6_V2.return_value
        call_args = ef_instance.call_args.args[0]
        assert len(call_args[0]) <= 1000

    def test_missing_chromadb_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "chromadb", None)
        monkeypatch.setitem(sys.modules, "chromadb.utils", None)
        monkeypatch.setitem(sys.modules, "chromadb.utils.embedding_functions", None)

        with pytest.raises(lep.LocalEmbedderError, match="chromadb"):
            LocalEmbedder.get().embed(["hi"])


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
        """Layer A path 3: chromadb available + env var UNSET still routes local/*."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        _install_fake_chroma(monkeypatch)
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embedding("hello", model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 512

    def test_get_embeddings_routes_to_local_without_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Layer A path 3 on the batch path: chromadb only is enough."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import LiteLLMClient, LiteLLMConfig

        _install_fake_chroma(monkeypatch)
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: MagicMock())

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with patch.object(litellm_client.litellm, "embedding") as mock_embedding:
            result = client.get_embeddings(["a", "b"], model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
        assert len(result) == 2
        assert all(len(vec) == 512 for vec in result)

    def test_get_embedding_local_without_chromadb_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When local/* is requested but chromadb is missing, raise a clear error."""
        from reflexio.server.llm import litellm_client
        from reflexio.server.llm.litellm_client import (
            LiteLLMClient,
            LiteLLMClientError,
            LiteLLMConfig,
        )

        # Force "inprocess" provider mode so the embedding-service short-circuit
        # doesn't intercept the local/* model before the chromadb-import check.
        # Clearing the env vars is not enough: model-driven routing probes the
        # local daemon, so if one happens to be running on this host the call
        # would route there. Force the service gate off to stay hermetic.
        monkeypatch.delenv("REFLEXIO_EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("REFLEXIO_EMBEDDING_SERVICE_URL", raising=False)
        monkeypatch.delenv("CLAUDE_SMART_USE_LOCAL_EMBEDDING", raising=False)
        monkeypatch.setattr(
            _litellm_embedding, "should_use_embedding_service", lambda _model: False
        )
        monkeypatch.setattr(lep.importlib.util, "find_spec", lambda _name: None)

        client = LiteLLMClient(LiteLLMConfig(model="claude-code/default"))

        with (
            patch.object(litellm_client.litellm, "embedding") as mock_embedding,
            pytest.raises((LiteLLMClientError, RuntimeError), match="chromadb"),
        ):
            client.get_embedding("hello", model="local/minilm-l6-v2")

        mock_embedding.assert_not_called()
