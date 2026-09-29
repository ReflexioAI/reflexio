"""Inference can import without application clients, schemas or LLM orchestration."""

import os
import subprocess
import sys


def test_inference_import_does_not_load_application_packages(tmp_path):
    env = dict(os.environ, REFLEXIO_ENV_FILE=str(tmp_path / "absent.env"))
    code = """
import importlib.abc
import sys

class RejectApplicationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('reflexio.client', 'reflexio.models',
                     'reflexio.server.llm.litellm_client',
                     'reflexio.server.llm.rerank.cross_encoder_reranker')
        if any(fullname == prefix or fullname.startswith(prefix + '.') for prefix in forbidden):
            raise AssertionError('Inference imported application code: ' + fullname)

sys.meta_path.insert(0, RejectApplicationImports())
from reflexio.server.llm.embedding_service import create_embedding_app
from reflexio.server.llm.embedding_policy import MULTILINGUAL_E5_MODEL
assert callable(create_embedding_app)
assert MULTILINGUAL_E5_MODEL == 'local/multilingual-e5-small'
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr


def test_public_facades_preserve_object_identity():
    import reflexio
    from reflexio.client import ReflexioClient
    from reflexio.models.config_schema import Config
    from reflexio.server import llm
    from reflexio.server.llm import rerank
    from reflexio.server.llm.litellm_client import LiteLLMClient
    from reflexio.server.llm.rerank.cross_encoder_reranker import score_pairs

    assert reflexio.ReflexioClient is ReflexioClient
    assert reflexio.Config is Config
    assert llm.LiteLLMClient is LiteLLMClient
    assert rerank.score_pairs is score_pairs
    assert "ReflexioClient" in dir(reflexio)
    assert "LiteLLMClient" in dir(llm)


def test_legacy_policy_and_env_imports_preserve_identity():
    from reflexio.server import env_utils
    from reflexio.server.llm import embedding_policy, llm_utils
    from reflexio.server.services import embedding_text

    assert llm_utils.positive_int_env is env_utils.positive_int_env
    for name in (
        "EmbeddingModelPolicy",
        "embedding_input",
        "is_multilingual_e5_model",
        "resolve_retrieval_threshold",
        "resolve_clustering_similarity",
    ):
        assert getattr(embedding_text, name) is getattr(embedding_policy, name)


def test_all_public_exports_remain_importable():
    import reflexio
    from reflexio.server import llm
    from reflexio.server.llm import rerank

    for module in (reflexio, llm, rerank):
        for name in module.__all__:
            getattr(module, name)
