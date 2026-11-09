"""One owner for sorting failed LLM calls: ``reflexio/server/llm/failure.py``.

``classify_llm_failure`` is the verdict and ``exception_chain`` is its one walk
over ``__cause__``/``__context__``. Before them, five private chain walks in
four modules sorted the same failures with slightly different rules. This guard
derives every candidate from the source tree and fails on:

1. **A private chain walk.** A module that imports the LLM layer
   (``reflexio.server.llm``, or lives in it) and reads ``__cause__`` /
   ``__context__`` / ``__suppress_context__`` outside ``failure.py``. Callers
   that need more detail than the verdict branch on it and read
   ``exception_chain``. Recorded in ``CHAIN_READS`` by ``module::function``.
2. **A swallowed LLM failure that never reads the verdict.** An ``except``
   naming an LLM error type whose handler does not re-raise decides the
   failure itself, so its module must call ``classify_llm_failure`` or
   ``describe_tool_loop_error``. Recorded in ``SWALLOWS`` by module.
3. **A deleted predicate coming back.** ``TOMBSTONES`` names the predicates
   the verdict replaced; defining, importing or reading one fails.

Every entry carries its reason, and an entry the scan no longer finds fails, so
the registries shrink as the code does.

Blind spot, recorded rather than hidden: rule 1 scopes by import, so a module
that walks a chain of LLM exceptions it received without importing the LLM
layer is not seen (``background_work.is_transient_failure`` is such a module:
it classifies database and network transience, follows ``__cause__`` only, and
lets an injected classifier decide the rest).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.deploy_guard

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "reflexio"
_OWNER = "server/llm/failure.py"
_LLM_PACKAGE = "reflexio.server.llm"
_CHAIN_ATTRS = frozenset({"__cause__", "__context__", "__suppress_context__"})
_LLM_ERRORS = frozenset(
    {
        "LiteLLMClientError",
        "ProviderCircuitOpenError",
        "ProviderCapSaturatedError",
        "StructuredOutputParseError",
        "StructuredOutputRepairError",
    }
)
_VERDICT_READERS = frozenset({"classify_llm_failure", "describe_tool_loop_error"})

#: Predicates the verdict replaced (PR1 and PR2 of llm-failure-verdict).
TOMBSTONES = frozenset(
    {
        "provider_wait_seconds",
        "is_provider_unavailable_error",
        "_is_expected_transient_llm_error",
        "_is_timeout_error",
        "cap_saturated",
    }
)

#: Chain reads outside ``failure.py`` in LLM-importing modules, with why.
CHAIN_READS: dict[str, str] = {
    "server/services/offline_tuner/open_world/rule_evidence_runtime.py"
    "::rejected_model_output": (
        "decision 2A: on top of the verdict, follows __cause__ only, because a "
        "failed corrective retry carries the earlier HANDLED rejection as its "
        "__context__ and must not be relabelled as the model's answer"
    ),
    "server/services/durable_learning/worker.py::root_cause": (
        "log-only: names the innermost explicit cause in a failure locator"
    ),
    "server/services/durable_learning/worker.py::_cause_chain": (
        "classifies our own input-budget, evidence-changed and saved-output "
        "errors (parking decisions), not provider failures; the verdict has no "
        "answer for them"
    ),
    "server/services/generation_service.py::callback": (
        "log-only: the agent-success log line names the wrapped error's type"
    ),
    "server/llm/providers/local_embedding_provider.py::_exception_chain": (
        "local embedding model cache errors (file paths, checksums), not an "
        "LLM call failure"
    ),
    "server/services/unified_search_service.py::_combined_rpc_unavailable": (
        "database capability errors (pgcode 42883, PGRST202), not an LLM call failure"
    ),
}

#: Modules that swallow an LLM failure without reading the verdict, with why.
SWALLOWS: dict[str, str] = {
    "server/services/agent_success_evaluation/components/"
    "retrieved_learning_evaluator.py": (
        "KNOWN GAP: a judge failure becomes a failed/degraded chunk, so an "
        "outage spends generation_service's bounded retrieved-learning "
        "re-arms. Waiting needs the outage carried through the run outcome "
        "and a product decision on partial (degraded) runs"
    ),
}


def _modules() -> dict[str, ast.Module]:
    return {
        path.relative_to(_SOURCE_ROOT).as_posix(): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(_SOURCE_ROOT.rglob("*.py"))
    }


def _imports_llm(module: str, tree: ast.Module) -> bool:
    if module.startswith("server/llm/"):
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            _LLM_PACKAGE
        ):
            return True
        if isinstance(node, ast.Import) and any(
            alias.name.startswith(_LLM_PACKAGE) for alias in node.names
        ):
            return True
    return False


def _chain_reads(modules: dict[str, ast.Module], owner: str = _OWNER) -> set[str]:
    """``module::function`` for each chain read (module scope is ``<module>``)."""
    found: set[str] = set()

    def visit(module: str, node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (
                child.name
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                else scope
            )
            if _reads_chain(child):
                found.add(f"{module}::{scope}")
            visit(module, child, inner)

    for module, tree in modules.items():
        if module != owner and _imports_llm(module, tree):
            visit(module, tree, "<module>")
    return found


def _reads_chain(node: ast.AST) -> bool:
    """``x.__cause__`` (a load) or ``getattr(x, "__cause__", ...)``."""
    if isinstance(node, ast.Attribute):
        return node.attr in _CHAIN_ATTRS and isinstance(node.ctx, ast.Load)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and node.args[1].value in _CHAIN_ATTRS
    )


def _calls_a_verdict_reader(tree: ast.Module, readers: frozenset[str]) -> bool:
    """A call, not an import: importing the verdict and ignoring it is no read."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in readers:
                return True
    return False


def _always_raises(handler: ast.ExceptHandler) -> bool:
    """Every path re-raises: the last statement raises and nothing exits early.

    Conservative: a ``return`` / ``break`` / ``continue`` anywhere in the
    handler (outside a nested function) counts as a swallowing path.
    """
    if not isinstance(handler.body[-1], ast.Raise):
        return False
    pending: list[ast.AST] = list(handler.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.Return | ast.Break | ast.Continue):
            return False
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            pending.extend(ast.iter_child_nodes(node))
    return True


def _names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.rsplit(".", 1)[-1])
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
    return names


def _swallows(
    modules: dict[str, ast.Module], readers: frozenset[str] = _VERDICT_READERS
) -> set[str]:
    found: set[str] = set()
    for module, tree in modules.items():
        if _calls_a_verdict_reader(tree, readers):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler) or node.type is None:
                continue
            types = node.type.elts if isinstance(node.type, ast.Tuple) else [node.type]
            caught = {
                t.id if isinstance(t, ast.Name) else getattr(t, "attr", "")
                for t in types
            }
            propagates = _always_raises(node)
            if caught & _LLM_ERRORS and not propagates:
                found.add(module)
    return found


def test_no_private_chain_walk_outside_the_verdict() -> None:
    found = _chain_reads(_modules())
    unrecorded = sorted(found - CHAIN_READS.keys())
    assert unrecorded == [], (
        "These functions read __cause__/__context__ of LLM failures outside "
        "reflexio/server/llm/failure.py. Branch on classify_llm_failure and "
        "read exception_chain for detail, or record the reason in CHAIN_READS: "
        f"{unrecorded}"
    )
    stale = sorted(CHAIN_READS.keys() - found)
    assert stale == [], f"Retire these CHAIN_READS entries: {stale}"


def test_every_swallowed_llm_failure_reads_the_verdict() -> None:
    found = _swallows(_modules())
    unrecorded = sorted(found - SWALLOWS.keys())
    assert unrecorded == [], (
        "These modules catch an LLM failure, do not re-raise, and never read "
        "the verdict. Act on classify_llm_failure (OutageWait waits uncharged), "
        f"or record the reason in SWALLOWS: {unrecorded}"
    )
    stale = sorted(SWALLOWS.keys() - found)
    assert stale == [], f"Retire these SWALLOWS entries: {stale}"


def test_no_deleted_predicate_comes_back() -> None:
    revived = sorted(
        f"{module}: {name}"
        for module, tree in _modules().items()
        for name in _names(tree) & TOMBSTONES
    )
    assert revived == [], f"Use classify_llm_failure instead: {revived}"


def test_every_entry_carries_a_reason() -> None:
    blank = [key for key, why in {**CHAIN_READS, **SWALLOWS}.items() if not why.strip()]
    assert blank == []


def test_the_scan_is_not_blind() -> None:
    """A scan that finds nothing would pass every check above."""
    modules = _modules()
    assert f"{_OWNER}::exception_chain" in _chain_reads(modules, owner="")
    # rule_evidence_runtime swallows a client error and reads the verdict.
    rule_evidence = "server/services/offline_tuner/open_world/rule_evidence_runtime.py"
    assert rule_evidence not in _swallows(modules)
    assert rule_evidence in _swallows(modules, readers=frozenset())


@pytest.mark.parametrize(
    ("source", "rule"),
    [
        ("def f(e):\n    return e.__context__\n", "chain"),
        ("def f(e):\n    return getattr(e, '__cause__', None)\n", "chain"),
        (
            "def f(call, retry):\n"
            "    try:\n"
            "        return call()\n"
            "    except LiteLLMClientError:\n"
            "        if retry:\n"
            "            raise\n"
            "        return None\n",
            "swallow",
        ),
        (
            "from reflexio.server.llm.failure import classify_llm_failure\n"
            "def f(call):\n"
            "    try:\n"
            "        return call()\n"
            "    except LiteLLMClientError:\n"
            "        return None\n",
            "swallow",
        ),
        (
            "def f(call, retry):\n"
            "    try:\n"
            "        return call()\n"
            "    except LiteLLMClientError:\n"
            "        if retry:\n"
            "            return None\n"
            "        raise\n",
            "swallow",
        ),
    ],
    ids=[
        "attribute",
        "getattr",
        "raise-on-one-branch",
        "imported-not-called",
        "early-return-then-raise",
    ],
)
def test_the_scan_detects_each_shape(source: str, rule: str) -> None:
    tree = ast.parse("import reflexio.server.llm\n" + source)
    if rule == "chain":
        assert _chain_reads({"server/scratch.py": tree}) == {"server/scratch.py::f"}
    else:
        assert _swallows({"server/scratch.py": tree}) == {"server/scratch.py"}
