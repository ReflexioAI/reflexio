"""Importer-surface guard for the ``litellm_client`` facade (Tier-2.5 decomposition).

``litellm_client.py`` is decomposed into a facade + 3 concern mixins + 3
stateless/type leaves. The facade MUST stay identity-preserving so all ~102
importers across both repos need zero edits. This guard is the single contract
test (per code-review-and-invariants rule #1) that turns "a move dropped a
re-export" from a scattered ImportError hunt into one failing assertion.

The public + internal name lists were AST-scanned from BOTH repos (every
``from ...litellm_client import <name>`` site). Do NOT trim them without
re-scanning — a name here is imported somewhere and dropping it breaks that site.

SINK-2 (identity): re-exports are by import binding, never redefinition. The
module-level ``_TRUNCATION_WARNED_MODELS`` set and each ``_Completion*Snapshot``
class must be the SAME object/class the moved code uses and tests touch — the
identity asserts below enforce that once the leaf/mixin modules exist.

SINK-3: the facade keeps ``import litellm`` so the ~40 tests that patch
``litellm_client.litellm.completion/.embedding/.get_model_info`` keep resolving
the shared ``litellm`` module object.
"""

import importlib

import pytest

FACADE = "reflexio.server.llm.litellm_client"

# The 5 public names (facade ``__all__``), also re-exported via server/llm/__init__.
PUBLIC_SYMBOLS = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LiteLLMClientError",
    "ToolCallingChatResponse",
    "create_litellm_client",
]

# Test-imported internals that ImportError at collection if dropped. AST-scanned
# from tests/server/llm/test_litellm_client_unit.py (imports) + :785 (_TRUNCATION).
INTERNAL_SYMBOLS = [
    "LLMHardTimeoutError",
    "StructuredOutputParseError",
    "_CompletionErrorSnapshot",
    "_get_embedding_encoding",
    "_get_embedding_limit",
    "_litellm_completion_worker",
    "_truncate_for_embedding",
    "_TRUNCATION_WARNED_MODELS",
]

# Sibling leaf/mixin modules the split introduces. Each must cold-import cleanly
# (no cycle) once created; guarded so the test is meaningful pre- and post-split.
SPLIT_MODULES = [
    "reflexio.server.llm._litellm_types",
    "reflexio.server.llm._litellm_json_extraction",
    "reflexio.server.llm._litellm_subprocess",
    "reflexio.server.llm._litellm_embedding",
    "reflexio.server.llm._litellm_structured_output",
    "reflexio.server.llm._litellm_text_generation",
]


@pytest.fixture(scope="module")
def facade():
    return importlib.import_module(FACADE)


@pytest.mark.parametrize("name", PUBLIC_SYMBOLS)
def test_public_symbol_importable(facade, name):
    assert hasattr(facade, name), f"{name} missing from {FACADE} (public surface)"


@pytest.mark.parametrize("name", INTERNAL_SYMBOLS)
def test_internal_symbol_importable(facade, name):
    assert hasattr(facade, name), (
        f"{name} missing from {FACADE} (test-imported internal). "
        "Re-export it by import binding (SINK-2), never redefine."
    )


def test_all_equals_public_surface(facade):
    assert set(facade.__all__) == set(PUBLIC_SYMBOLS)


def test_facade_keeps_litellm_module_attr(facade):
    """SINK-3: ~40 tests patch ``litellm_client.litellm.<fn>`` — the facade must
    keep ``import litellm`` (do not let ruff prune it as F401)."""
    assert hasattr(facade, "litellm"), (
        "litellm_client must keep `import litellm` at module top (SINK-3); "
        "~40 tests patch litellm_client.litellm.completion/.embedding/.get_model_info"
    )


def test_package_init_reexports_public_surface():
    pkg = importlib.import_module("reflexio.server.llm")
    for name in PUBLIC_SYMBOLS:
        assert hasattr(pkg, name), f"{name} missing from reflexio.server.llm re-export"


@pytest.mark.parametrize("module", SPLIT_MODULES)
def test_split_module_cold_imports(module):
    """Each new sibling module imports cleanly (no cycle). Skipped until created."""
    try:
        importlib.import_module(module)
    except ModuleNotFoundError:
        pytest.skip(f"{module} not created yet")


def test_boot_import_paths():
    """Boot smoke: both the direct-module and package re-export paths resolve."""
    import reflexio.server.llm.litellm_client as _facade_mod  # noqa: F401
    from reflexio.server.llm import (  # noqa: F401
        LiteLLMClient,
        LiteLLMClientError,
        LiteLLMConfig,
        ToolCallingChatResponse,
        create_litellm_client,
    )


# ---------------------------------------------------------------------------
# SINK-2 identity asserts. Skipped until the owning module exists (Task 1/2),
# then enforce that the facade re-export IS the same object the moved code uses.
# ---------------------------------------------------------------------------

_SNAPSHOT_CLASSES = [
    "_CompletionMessageSnapshot",
    "_CompletionChoiceSnapshot",
    "_PromptTokenDetailsSnapshot",
    "_CompletionUsageSnapshot",
    "_CompletionResponseSnapshot",
    "_CompletionErrorSnapshot",
]


@pytest.mark.parametrize("name", _SNAPSHOT_CLASSES)
def test_snapshot_class_identity(facade, name):
    """Each snapshot re-exported by the facade IS the class the subprocess
    worker constructs and tests isinstance-check (SINK-2)."""
    try:
        subprocess_mod = importlib.import_module(
            "reflexio.server.llm._litellm_subprocess"
        )
    except ModuleNotFoundError:
        pytest.skip("_litellm_subprocess not created yet")
    if not hasattr(facade, name):
        pytest.skip(f"{name} not re-exported yet")
    assert getattr(facade, name) is getattr(subprocess_mod, name)


def test_truncation_warned_models_identity(facade):
    """The facade ``_TRUNCATION_WARNED_MODELS`` IS the set the embedding code
    mutates and the test autouse ``.clear()`` fixture touches (SINK-2)."""
    try:
        embedding_mod = importlib.import_module(
            "reflexio.server.llm._litellm_embedding"
        )
    except ModuleNotFoundError:
        pytest.skip("_litellm_embedding not created yet")
    assert facade._TRUNCATION_WARNED_MODELS is embedding_mod._TRUNCATION_WARNED_MODELS
