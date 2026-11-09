"""One offline-policy input, including direct native calls after imports."""

import os
import subprocess
import sys
from unittest.mock import Mock

import pytest

from reflexio.cli import env_loader
from reflexio.server.llm import input_budget
from reflexio.server.llm.providers.nomic_embedding_provider import NomicEmbedder
from reflexio.server.llm.rerank import cross_encoder_model
from reflexio.server.llm.rerank.common import MULTILINGUAL_RERANK_MODEL


@pytest.mark.parametrize("value", ["", "0", "1", " ", "private-sentinel"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("consumer", ["embedding", "reranking", "tokenizer"])
def test_obsolete_offline_input_refused_before_native_work(
    monkeypatch, value, cached, consumer
):
    # Modules are already imported; a cached native model must not bypass admission.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", value)
    native_work = Mock(side_effect=AssertionError("native work started"))
    if consumer == "embedding":
        runner = NomicEmbedder()
        runner._model = object() if cached else None
        monkeypatch.setattr(runner, "_validate_device", native_work)
        operation = runner._load
    elif consumer == "reranking":
        runner = cross_encoder_model.CrossEncoderRunner(MULTILINGUAL_RERANK_MODEL)
        runner._model = object() if cached else None
        monkeypatch.setattr(cross_encoder_model, "_import_cross_encoder", native_work)
        monkeypatch.setattr(cross_encoder_model, "reranker_enabled", lambda: True)
        operation = runner._get_model
    else:
        monkeypatch.setattr(input_budget.litellm, "token_counter", native_work)

        def operation():
            return input_budget.measure_request_input_tokens(
                {"model": "gpt-4o-mini", "messages": []}
            )

    with pytest.raises(
        ValueError, match="TRANSFORMERS_OFFLINE.*HF_HUB_OFFLINE"
    ) as error:
        operation()
    assert "private-sentinel" not in str(error.value)
    native_work.assert_not_called()


@pytest.mark.parametrize("source", ["process", "file"])
@pytest.mark.parametrize("value", ["", "0", "1", "private-sentinel"])
def test_existing_env_admission_rejects_offline_alias(
    monkeypatch, tmp_path, source, value
):
    monkeypatch.delenv("PYTHON_DOTENV_DISABLED", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    env_file = tmp_path / "selected.env"
    env_file.write_text("HF_HUB_OFFLINE=1\n")
    if source == "process":
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", value)
    else:
        env_file.write_text(f"TRANSFORMERS_OFFLINE={value}\nHF_HUB_OFFLINE=1\n")
    with pytest.raises(ValueError, match="TRANSFORMERS_OFFLINE") as error:
        env_loader._load_dotenv_pruned(env_file)
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "private-sentinel" not in str(error.value)
    assert "effective" in str(error.value)


@pytest.mark.parametrize(
    "value,offline",
    [(None, False), ("", False), ("0", False), ("1", True), ("YES", True)],
)
def test_canonical_policy_uses_native_hub_semantics(value, offline):
    # Fresh interpreter is essential: Hub resolves its environment at import time.
    env = dict(os.environ)
    env.pop("TRANSFORMERS_OFFLINE", None)
    env.pop("HF_HUB_OFFLINE", None)
    if value is not None:
        env["HF_HUB_OFFLINE"] = value
    program = (
        "import os; from reflexio.removed_env import reject_removed_env; "
        "reject_removed_env(os.environ, canonical='HF_HUB_OFFLINE'); "
        "from huggingface_hub.constants import HF_HUB_OFFLINE; "
        f"assert HF_HUB_OFFLINE is {offline!r}"
    )
    subprocess.run(  # noqa: S603 - fixed interpreter and test-owned program
        [sys.executable, "-c", program], env=env, check=True, timeout=20
    )
