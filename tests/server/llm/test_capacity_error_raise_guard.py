"""The capacity error is raised only for capacity; integrity has its own family.

``InputBudgetConfigurationError`` means a safe model/endpoint capacity could
not be established. A window whose frozen inputs are unavailable, changed or
malformed raises ``ExtractionInputIntegrityError`` instead. Both keep the same
hold and the same public ``input_budget_configuration_error`` reason; only the
internal label (logs, metrics) tells them apart, so a non-capacity reason
raised as the capacity class silently mislabels every hold it causes.

The scan walks every module under ``reflexio/`` and fails two ways:

- a construction of ``InputBudgetConfigurationError`` (or a class deriving
  from it directly) that :data:`CAPACITY_RAISES` does not record;
- a :data:`CAPACITY_RAISES` entry the scan no longer finds.

:data:`CAPACITY_RAISES` is keyed by module and message, so rewording a capacity
raise retires its entry and must be re-recorded deliberately. The scan sees
direct constructions only: an alias (``Err = InputBudgetConfigurationError``)
or a ``type(...)`` call would escape it, and none exists today.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[3] / "reflexio"
CAPACITY = "InputBudgetConfigurationError"
INPUT_BUDGET = "reflexio/server/llm/input_budget.py"

#: Every construction of the capacity class, by (module, message). Each one
#: refuses a request because model or endpoint capacity is undeclared,
#: conflicting or uncountable, or because the measured capacity (or the
#: call budget that spends it) leaves no room -- the class's meaning.
CAPACITY_RAISES: frozenset[tuple[str, str]] = frozenset(
    {
        (
            "reflexio/server/services/extraction/profile_package_preparation.py",
            "'Profile fixed request lacks capacity'",
        ),
        (
            "reflexio/server/services/playbook/components/context_compression.py",
            "'No consolidation input capacity remains'",
        ),
        (
            "reflexio/server/services/playbook/components/context_compression.py",
            "'Compression must reserve draft and audit calls'",
        ),
    }
) | frozenset(
    (INPUT_BUDGET, message)
    for message in (
        "'Custom endpoint requires an explicit output reservation'",
        "'Invalid context-limit declarations'",
        "'Context limits require positive integers'",
        "'Conflicting provider endpoint declarations'",
        "'Effective output reservation is required'",
        "'Missing effective model'",
        "'Complete provider messages are required'",
        "'Complete prompt counting failed'",
        "'Missing effective model or prompt target'",
        "'Custom endpoint requires a context declaration'",
        "'Unknown model context capacity'",
    )
)

#: Classes deriving directly from the capacity class.
CAPACITY_SUBCLASSES: frozenset[tuple[str, str]] = frozenset(
    {(INPUT_BUDGET, "ExtractionInputIntegrityError")}
)


def _names_capacity(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == CAPACITY
    return isinstance(node, ast.Attribute) and node.attr == CAPACITY


def _scan() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Return (constructions, direct subclasses) of the capacity class."""
    raises: set[tuple[str, str]] = set()
    subclasses: set[tuple[str, str]] = set()
    for path in sorted(PACKAGE.rglob("*.py")):
        module = path.relative_to(PACKAGE.parent).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and _names_capacity(node.func):
                message = ast.unparse(node.args[0]) if node.args else "<none>"
                raises.add((module, message))
            elif isinstance(node, ast.Raise) and (
                node.exc is not None and _names_capacity(node.exc)
            ):
                raises.add((module, "<none>"))
            elif isinstance(node, ast.ClassDef) and any(
                _names_capacity(base) for base in node.bases
            ):
                subclasses.add((module, node.name))
    return raises, subclasses


def test_scan_sees_the_capacity_raises() -> None:
    """The walk is not empty by construction."""
    raises, subclasses = _scan()
    assert len(raises) >= len(CAPACITY_RAISES)
    assert subclasses


def test_capacity_error_is_raised_only_for_capacity() -> None:
    raises, _ = _scan()
    unrecorded = sorted(raises - CAPACITY_RAISES)
    assert not unrecorded, (
        "Raise ExtractionInputIntegrityError for a non-capacity failure, or "
        f"record a genuine capacity raise in CAPACITY_RAISES: {unrecorded}"
    )


def test_only_the_integrity_family_derives_from_the_capacity_error() -> None:
    _, subclasses = _scan()
    unrecorded = sorted(subclasses - CAPACITY_SUBCLASSES)
    assert not unrecorded, (
        f"Derive non-capacity errors from ExtractionInputIntegrityError: {unrecorded}"
    )


def test_every_recorded_capacity_raise_still_exists() -> None:
    raises, subclasses = _scan()
    assert not sorted(CAPACITY_RAISES - raises)
    assert not sorted(CAPACITY_SUBCLASSES - subclasses)
