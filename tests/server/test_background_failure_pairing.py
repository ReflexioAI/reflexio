"""Every background failure scope has a success report for the same unit.

``report_background_failure`` escalates a transient failure once its scope has
failed several consecutive times over several minutes; only
``report_background_success`` for that scope ends the streak. A failure site
with no matching success therefore never resets: blips days apart add up to a
page. And two failure sites sharing one scope prefix let one unit's success
clear the other's streak -- the aggregation run and org scopes once built the
same string.

What this guard enforces, per module under ``reflexio/server``:

1. every ``report_background_failure(scope=...)`` has a
   ``report_background_success(...)`` whose scope has the same constant prefix
   (the literal text before the first placeholder);
2. every success prefix matches some failure prefix (a typo in a success
   scope would otherwise pass rule 1 only because nothing checked it);
3. no two failure call sites share a prefix -- which subsumes two sites with
   an identical scope expression;
4. every scope resolves to a string with a non-empty constant prefix. A scope
   held in a local or ``self`` attribute is followed to its single assignment.

Non-goals: it cannot prove the success call sits where the unit completes (a
success in the wrong branch passes), and it reads modules, not call graphs.
Those stay review questions; the per-site tests pin the placement.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import reflexio.server

_SERVER_ROOT = Path(reflexio.server.__file__).parent
_FAILURE = "report_background_failure"
_SUCCESS = "report_background_success"


@dataclass(frozen=True)
class _Call:
    kind: str  # _FAILURE or _SUCCESS
    line: int
    prefix: str | None  # None when the scope could not be resolved
    expression: str


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _scope_arg(call: ast.Call, kind: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == "scope":
            return keyword.value
    if kind == _SUCCESS and call.args:
        return call.args[0]
    return None


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Map every node to its innermost enclosing function (or the module)."""
    owner: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):  # breadth-first: inner functions overwrite
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                owner[child] = node
    return owner


def _assigned_values(
    within: ast.AST, matches: Callable[[ast.expr], bool]
) -> list[ast.expr]:
    values = []
    for node in ast.walk(within):
        if isinstance(node, ast.Assign):
            if any(matches(target) for target in node.targets):
                values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and node.value is not None
            and matches(node.target)
        ):
            values.append(node.value)
    return values


def _resolve(
    expr: ast.expr, tree: ast.Module, owner: dict[ast.AST, ast.AST], depth: int = 0
) -> ast.expr | None:
    """Follow a Name or ``self.attr`` to its single assigned value."""
    if depth > 4:
        return None
    if isinstance(expr, ast.Name):
        name = expr.id
        values = _assigned_values(
            owner.get(expr, tree),
            lambda t: isinstance(t, ast.Name) and t.id == name,
        )
    elif (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    ):
        attr = expr.attr
        values = _assigned_values(
            tree,
            lambda t: (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and t.attr == attr
            ),
        )
    else:
        return expr
    if len({ast.dump(value) for value in values}) != 1:
        return None
    return _resolve(values[0], tree, owner, depth + 1)


def _prefix(expr: ast.expr) -> tuple[str, bool] | None:
    """Constant text before the first placeholder, and whether it is all of it."""
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value, True
    if isinstance(expr, ast.JoinedStr):
        text = ""
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                text += part.value
            else:
                return text, False
        return text, True
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "format"
        and isinstance(expr.func.value, ast.Constant)
        and isinstance(expr.func.value.value, str)
    ):
        template = expr.func.value.value
        head = template.split("{", 1)[0]
        return head, head == template
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left = _prefix(expr.left)
        if left is None:
            return None
        if not left[1]:
            return left
        right = _prefix(expr.right)
        if right is None:
            return left[0], False
        return left[0] + right[0], right[1]
    return None


def _calls_in(source: str) -> list[_Call]:
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind = _called_name(node)
        if kind not in (_FAILURE, _SUCCESS):
            continue
        arg = _scope_arg(node, kind)
        resolved = _resolve(arg, tree, owner) if arg is not None else None
        prefix = _prefix(resolved) if resolved is not None else None
        calls.append(
            _Call(
                kind=kind,
                line=node.lineno,
                prefix=prefix[0] if prefix and prefix[0] else None,
                expression=ast.unparse(arg) if arg is not None else "<missing>",
            )
        )
    return calls


def _violations(source: str) -> list[str]:
    calls = _calls_in(source)
    problems = [
        f"line {c.line}: {c.kind} scope {c.expression!r} has no constant prefix "
        "(use a literal, an f-string or a single assignment)"
        for c in calls
        if c.prefix is None
    ]
    failures: dict[str, list[int]] = defaultdict(list)
    successes: set[str] = set()
    for call in calls:
        if call.prefix is None:
            continue
        if call.kind == _FAILURE:
            failures[call.prefix].append(call.line)
        else:
            successes.add(call.prefix)
    for prefix, lines in sorted(failures.items()):
        if prefix not in successes:
            problems.append(
                f"lines {lines}: failure scope prefix {prefix!r} has no "
                f"{_SUCCESS} with that prefix -- its streak would never reset"
            )
        if len(lines) > 1:
            problems.append(
                f"lines {lines}: failure sites share the scope prefix {prefix!r} "
                "-- one unit's success would clear the other's streak"
            )
    problems.extend(
        f"success scope prefix {prefix!r} matches no failure scope in this "
        "module (a typo leaves the failure's streak unreset)"
        for prefix in sorted(successes - failures.keys())
    )
    return problems


def _server_modules() -> list[Path]:
    return sorted(
        path for path in _SERVER_ROOT.rglob("*.py") if path.name != "background_work.py"
    )


def test_every_failure_scope_is_paired_and_unique() -> None:
    report = {}
    for path in _server_modules():
        source = path.read_text()
        if _FAILURE not in source and _SUCCESS not in source:
            continue
        problems = _violations(source)
        if problems:
            report[str(path.relative_to(_SERVER_ROOT))] = problems
    assert report == {}


def test_the_scan_sees_every_failure_call() -> None:
    """The guard is only as good as its selection: count calls two ways.

    The AST count must equal a plain-text count of call openings, so a call
    spelled in a way the AST walk misses cannot drop out unnoticed.
    """
    pattern = re.compile(rf"(?<!def ){_FAILURE}\(")
    ast_calls = 0
    text_calls = 0
    for path in _server_modules():
        source = path.read_text()
        text_calls += len(pattern.findall(source))
        ast_calls += sum(c.kind == _FAILURE for c in _calls_in(source))
    assert ast_calls == text_calls
    assert ast_calls >= 25  # the floor: the #582 sites plus the converted loops


# --- the rules fail on the shapes they exist to catch ------------------------


def _check(source: str) -> list[str]:
    return _violations(source)


def test_missing_success_is_reported() -> None:
    source = (
        "def tick(org):\n"
        "    try:\n"
        "        work()\n"
        "    except Exception as exc:\n"
        "        report_background_failure(log, 'e', exc, scope=f'job:{org}')\n"
    )
    assert any("has no report_background_success" in p for p in _check(source))


def test_shared_prefix_is_reported_even_when_both_succeed() -> None:
    source = (
        "def run(org, version):\n"
        "    scope = f'job:{org}:{version}'\n"
        "    try:\n"
        "        work()\n"
        "    except Exception as exc:\n"
        "        report_background_failure(log, 'e', exc, scope=scope)\n"
        "    else:\n"
        "        report_background_success(scope)\n"
        "def outer(org):\n"
        "    try:\n"
        "        run(org, 1)\n"
        "    except Exception as exc:\n"
        "        report_background_failure(log, 'e', exc, scope=f'job:{org}')\n"
        "    else:\n"
        "        report_background_success(f'job:{org}')\n"
    )
    assert any("share the scope prefix 'job:'" in p for p in _check(source))


def test_unresolvable_scope_is_reported() -> None:
    source = (
        "def tick(scope):\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        "    report_background_success(scope)\n"
    )
    assert any("no constant prefix" in p for p in _check(source))


def test_success_typo_is_reported() -> None:
    source = (
        "def tick():\n"
        "    report_background_failure(log, 'e', exc, scope='tagging-callback')\n"
        "    report_background_success('tagging-callbak')\n"
    )
    problems = _check(source)
    assert any("matches no failure scope" in p for p in problems)
    assert any("has no report_background_success" in p for p in problems)


def test_self_attribute_scope_is_followed() -> None:
    source = (
        "class Beat:\n"
        "    def __init__(self, org):\n"
        "        self._scope = f'beat:{org}'\n"
        "    def run(self):\n"
        "        report_background_failure(log, 'e', exc, scope=self._scope)\n"
        "        report_background_success(self._scope)\n"
    )
    assert _check(source) == []
