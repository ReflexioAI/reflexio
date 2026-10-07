"""Every background failure scope has a success report for the same unit.

``report_background_failure`` escalates a transient failure once its scope has
failed several consecutive times over several minutes; only
``report_background_success`` for that scope ends the streak. A failure site
with no matching success therefore never resets: blips days apart add up to a
page. A success whose scope names a DIFFERENT unit -- a sibling, an enclosing
org, another tenant -- erases a real outage. And two failure sites sharing a
scope prefix let one unit's success clear the other's streak; the aggregation
run and org scopes once built the same string.

What this guard enforces, per module under ``reflexio/server``:

1. every failure scope has a success scope of the same SHAPE: the same
   constant text and the same ordered placeholder expressions (normalised
   source), so ``f"job:{org}:{project}"`` does not match
   ``f"job:{project}:{org}"``;
2. every success shape matches some failure shape (a typo or a stale success
   is not silently carried);
3. no two failure call sites share a constant prefix, which subsumes two sites
   with an identical scope;
4. every scope resolves to a shape with a non-empty constant prefix. A scope
   held in a local or ``self`` attribute is followed to its single plain
   assignment; any other binding of that name (a second assignment, ``+=``, a
   walrus, a loop/``with``/unpacking target, an import, a ``match`` capture,
   a ``def``/``class``, a ``global``/``nonlocal`` declaration, a ``type``
   alias or type parameter, a parameter) makes it unresolvable, because the
   value at the call is then not the one read;
5. every reference to either function is a direct call -- import aliases are
   resolved, and anything else fails because the guard could not see what it
   is called with: assigning the function to a name, passing it elsewhere, or
   ``functools.partial`` (a partial's bound ``scope=`` can be overridden where
   it is invoked, so the scope the guard reads is not the one reported);
6. neither function name appears as a string constant, which is how
   ``getattr(module, "report_background_failure")`` would reach it without a
   reference the walk can see. Import statements carry names, not strings, so
   they are unaffected.

Non-goals: it cannot prove the success call sits where the unit completes, or
that the unit is retried at all (a success in the wrong branch passes; a
one-shot callback converted to the policy passes). Those are review questions;
the per-site tests pin the placement.
"""

from __future__ import annotations

import ast
import io
import string
import tokenize
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import reflexio.server

_SERVER_ROOT = Path(reflexio.server.__file__).parent
_FAILURE = "report_background_failure"
_SUCCESS = "report_background_success"
_CANONICAL = (_FAILURE, _SUCCESS)

# ("c", text) is constant text; ("p", source) is a placeholder expression.
Shape = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _Call:
    kind: str  # _FAILURE or _SUCCESS
    line: int
    shape: Shape | None  # None when the scope could not be resolved
    expression: str

    @property
    def prefix(self) -> str:
        if self.shape and self.shape[0][0] == "c":
            return self.shape[0][1]
        return ""


@dataclass
class _Module:
    calls: list[_Call]
    stray_references: list[int]  # lines of references that are not calls
    string_mentions: list[int]  # lines of string constants naming a function


def _aliases(tree: ast.Module) -> dict[str, str]:
    """Local name -> canonical function name, from ``from ... import``."""
    names = {name: name for name in _CANONICAL}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _CANONICAL:
                    names[alias.asname or alias.name] = alias.name
    return names


def _canonical(expr: ast.expr, aliases: dict[str, str]) -> str | None:
    if isinstance(expr, ast.Name):
        return aliases.get(expr.id)
    if isinstance(expr, ast.Attribute) and expr.attr in _CANONICAL:
        return expr.attr
    return None


def _scope_arg(
    args: list[ast.expr], keywords: list[ast.keyword], kind: str
) -> ast.expr | None:
    for keyword in keywords:
        if keyword.arg == "scope":
            return keyword.value
    if kind == _SUCCESS and args:
        return args[0]
    return None


def _enclosing_functions(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Map every node to its innermost enclosing function (or the module)."""
    owner: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):  # breadth-first: inner functions overwrite
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef):
            for child in ast.walk(node):
                owner[child] = node
    return owner


def _binding_targets(node: ast.AST) -> list[ast.expr]:
    """Every expression ``node`` binds or rebinds, unpacking included."""
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, ast.AugAssign | ast.AnnAssign | ast.NamedExpr):
        return [node.target]
    if isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
        return [node.target]
    if isinstance(node, ast.withitem) and node.optional_vars is not None:
        return [node.optional_vars]
    if isinstance(node, ast.Delete):
        return list(node.targets)
    if isinstance(node, ast.TypeAlias):  # type n = ...
        return [node.name]
    return []


def _bound_names(node: ast.AST) -> list[str]:
    """The plain names ``node`` binds, for bindings that are not expressions.

    ``except ... as n``; an import (``import a.b`` binds ``a``, ``from m import
    x`` binds ``x``, ``as`` binds its alias); the capture names of a ``match``
    pattern (``case str(n)``, ``case [*n]``, ``case {**n}``); ``def n``,
    ``async def n`` and ``class n``; ``global n`` / ``nonlocal n``, which
    make ``n`` refer to a binding outside the function; and PEP 695 type
    parameters (``def f[n]()``).
    """
    name: str | None = None
    if isinstance(node, ast.Global | ast.Nonlocal):
        return list(node.names)
    if isinstance(node, ast.ExceptHandler):
        name = node.name
    elif isinstance(node, ast.alias):
        if node.name != "*":
            name = node.asname or node.name.split(".")[0]
    elif isinstance(node, ast.MatchAs | ast.MatchStar):
        name = node.name
    elif isinstance(node, ast.MatchMapping):
        name = node.rest
    elif isinstance(
        node,
        ast.FunctionDef
        | ast.AsyncFunctionDef
        | ast.ClassDef
        | ast.TypeVar
        | ast.ParamSpec
        | ast.TypeVarTuple,
    ):
        name = node.name
    return [name] if name is not None else []


def _single_assigned_value(
    within: ast.AST, matches: Callable[[ast.expr], bool], parameters: set[str]
) -> ast.expr | None:
    """The value of the ONE plain assignment binding what ``matches``, if any.

    A second binding of any kind -- another assignment, ``+=``, a walrus, a
    loop or ``with`` target, tuple unpacking, ``del``, an ``except ... as`` or
    import alias, or a parameter of the same name -- means the value the call
    sees is not decidable from one assignment, so the scope is unresolvable
    rather than guessed.
    """
    if any(matches(ast.Name(id=name)) for name in parameters):
        return None
    binders: list[ast.AST] = []
    for node in ast.walk(within):
        if any(
            matches(part)
            for target in _binding_targets(node)
            for part in ast.walk(target)
            if isinstance(part, ast.expr)
        ):
            binders.append(node)
        elif node is not within and any(
            matches(ast.Name(id=name)) for name in _bound_names(node)
        ):
            # `within` itself is skipped: a function's own name binds it in
            # the ENCLOSING scope, not inside the body being resolved.
            binders.append(node)
    if len(binders) != 1:
        return None
    (binder,) = binders
    if (
        isinstance(binder, ast.Assign)
        and len(binder.targets) == 1
        and matches(binder.targets[0])  # the target itself, not a tuple element
    ):
        return binder.value
    if isinstance(binder, ast.AnnAssign) and binder.value is not None:
        return binder.value
    return None


def _parameters(function: ast.AST) -> set[str]:
    if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
        return set()
    arguments = function.args
    every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    every += [arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None]
    return {arg.arg for arg in every}


def _resolve(
    expr: ast.expr, tree: ast.Module, owner: dict[ast.AST, ast.AST], depth: int = 0
) -> ast.expr | None:
    """Follow a Name or ``self.attr`` to its single plain assignment."""
    if depth > 4:
        return None
    if isinstance(expr, ast.Name):
        name = expr.id
        function = owner.get(expr, tree)
        value = _single_assigned_value(
            function,
            lambda t: isinstance(t, ast.Name) and t.id == name,
            _parameters(function),
        )
    elif (
        isinstance(expr, ast.Attribute)
        and isinstance(expr.value, ast.Name)
        and expr.value.id == "self"
    ):
        attr = expr.attr
        value = _single_assigned_value(
            tree,
            lambda t: (
                isinstance(t, ast.Attribute)
                and isinstance(t.value, ast.Name)
                and t.value.id == "self"
                and t.attr == attr
            ),
            set(),
        )
    else:
        return expr
    if value is None:
        return None
    return _resolve(value, tree, owner, depth + 1)


def _merge(parts: list[tuple[str, str]]) -> Shape:
    merged: list[tuple[str, str]] = []
    for kind, text in parts:
        if kind == "c" and not text:
            continue
        if kind == "c" and merged and merged[-1][0] == "c":
            merged[-1] = ("c", merged[-1][1] + text)
        else:
            merged.append((kind, text))
    return tuple(merged)


def _placeholder(value: ast.expr, conversion: str, spec: str) -> str:
    """One normal form for an f-string field and a ``.format`` field."""
    return f"{ast.unparse(value)}!{conversion}:{spec}"


def _spec_text(spec: ast.expr | None) -> str | None:
    if spec is None:
        return ""
    if isinstance(spec, ast.JoinedStr) and all(
        isinstance(v, ast.Constant) for v in spec.values
    ):
        return "".join(str(v.value) for v in spec.values if isinstance(v, ast.Constant))
    return None  # a computed format spec: refuse rather than guess


def _shape_parts(expr: ast.expr) -> list[tuple[str, str]] | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [("c", expr.value)]
    if isinstance(expr, ast.JoinedStr):
        parts: list[tuple[str, str]] = []
        for part in expr.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                parts.append(("c", part.value))
            elif isinstance(part, ast.FormattedValue):
                conversion = chr(part.conversion) if part.conversion != -1 else ""
                spec = _spec_text(part.format_spec)
                if spec is None:
                    return None
                parts.append(("p", _placeholder(part.value, conversion, spec)))
            else:
                return None
        return parts
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "format"
        and isinstance(expr.func.value, ast.Constant)
        and isinstance(expr.func.value.value, str)
    ):
        if any(isinstance(arg, ast.Starred) for arg in expr.args):
            return None  # "{}:{}".format(*key): which value lands where is unknown
        keywords = {k.arg: k.value for k in expr.keywords if k.arg is not None}
        parts = []
        auto = 0
        for literal, field, spec, conversion in string.Formatter().parse(
            expr.func.value.value
        ):
            parts.append(("c", literal))
            if field is None:
                continue
            if field == "":
                field = str(auto)
                auto += 1
            value = (
                expr.args[int(field)]
                if field.isdigit() and int(field) < len(expr.args)
                else keywords.get(field)
            )
            if value is None:
                return None
            parts.append(("p", _placeholder(value, conversion or "", spec or "")))
        return parts
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left, right = _shape_parts(expr.left), _shape_parts(expr.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def _shape(expr: ast.expr) -> Shape | None:
    parts = _shape_parts(expr)
    if parts is None:
        return None
    shape = _merge(parts)
    if not shape or shape[0][0] != "c":
        return None  # a scope must start with constant text naming the worker
    return shape


def _scan(source: str) -> _Module:
    tree = ast.parse(source)
    owner = _enclosing_functions(tree)
    aliases = _aliases(tree)
    calls: list[_Call] = []
    accounted: set[int] = set()  # id() of reference nodes that are analysed
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kind = _canonical(node.func, aliases)
        args, keywords = node.args, node.keywords
        reference: ast.expr = node.func
        if kind is None:
            continue
        accounted.add(id(reference))
        arg = _scope_arg(args, keywords, kind)
        resolved = _resolve(arg, tree, owner) if arg is not None else None
        calls.append(
            _Call(
                kind=kind,
                line=node.lineno,
                shape=_shape(resolved) if resolved is not None else None,
                expression=ast.unparse(arg) if arg is not None else "<missing>",
            )
        )
    stray = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Name | ast.Attribute)
        and _canonical(node, aliases) is not None
        and id(node) not in accounted
    ]
    strings = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip() in _CANONICAL
    ]
    return _Module(calls=calls, stray_references=stray, string_mentions=strings)


def _violations(source: str) -> list[str]:
    module = _scan(source)
    problems = [
        f"lines {module.stray_references}: a reference to the failure-policy "
        "functions that is not a direct call (partial included) -- the guard "
        "cannot see what it is called with"
    ] * bool(module.stray_references)
    problems += [
        f"lines {module.string_mentions}: a failure-policy function named in a "
        "string (getattr and friends) -- the guard cannot see that call"
    ] * bool(module.string_mentions)
    problems += [
        f"line {c.line}: {c.kind} scope {c.expression!r} has no resolvable shape "
        "with a constant prefix (use a literal, an f-string or a single assignment)"
        for c in module.calls
        if c.shape is None
    ]
    failures: dict[Shape, list[int]] = defaultdict(list)
    prefixes: dict[str, list[int]] = defaultdict(list)
    successes: set[Shape] = set()
    for call in module.calls:
        if call.shape is None:
            continue
        if call.kind == _FAILURE:
            failures[call.shape].append(call.line)
            prefixes[call.prefix].append(call.line)
        else:
            successes.add(call.shape)
    problems += [
        f"lines {lines}: failure scope {shape} has no {_SUCCESS} of the same "
        "shape -- its streak would never reset"
        for shape, lines in sorted(failures.items())
        if shape not in successes
    ]
    problems += [
        f"lines {lines}: failure sites share the scope prefix {prefix!r} "
        "-- one unit's success would clear the other's streak"
        for prefix, lines in sorted(prefixes.items())
        if len(lines) > 1
    ]
    problems += [
        f"success scope {shape} matches no failure scope in this module "
        "(a typo or stale success leaves the failure's streak unreset)"
        for shape in sorted(successes - failures.keys())
    ]
    return problems


def _server_modules() -> list[Path]:
    return sorted(
        path for path in _SERVER_ROOT.rglob("*.py") if path.name != "background_work.py"
    )


def test_every_failure_scope_is_paired_and_unique() -> None:
    report = {}
    for path in _server_modules():
        source = path.read_text()
        if not any(name in source for name in _CANONICAL):
            continue
        problems = _violations(source)
        if problems:
            report[str(path.relative_to(_SERVER_ROOT))] = problems
    assert report == {}


def _name_tokens(source: str, names: set[str]) -> int:
    """Count identifier tokens in ``names`` plus string literals naming either
    function, ignoring comments."""
    count = 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NAME and tok.string in names:
            count += 1
        elif tok.type == tokenize.STRING:
            try:
                value = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):
                continue
            if isinstance(value, str) and value.strip() in _CANONICAL:
                count += 1
    return count


def test_the_scan_accounts_for_every_mention() -> None:
    """Selection check, by a second route that does not share the AST walk.

    Every identifier token naming either function or one of its aliases must
    be an import, a call or a stray reference (which the
    pairing test then rejects). A mention the walk skipped would leave the
    token count higher than what the scan accounted for.
    """
    total_calls = 0
    for path in _server_modules():
        source = path.read_text()
        tree = ast.parse(source)
        aliases = _aliases(tree)
        imported = sum(
            (alias.asname is not None) + 1
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name in _CANONICAL
        )
        module = _scan(source)
        accounted = (
            imported
            + len(module.calls)
            + len(module.stray_references)
            + len(module.string_mentions)
        )
        assert _name_tokens(source, set(aliases)) == accounted, path
        total_calls += sum(c.kind == _FAILURE for c in module.calls)
    assert total_calls >= 25  # the floor: the #582 sites plus the converted loops


# --- the rules fail on the shapes they exist to catch ------------------------


def test_missing_success_is_reported() -> None:
    source = (
        "def tick(org):\n"
        "    try:\n"
        "        work()\n"
        "    except Exception as exc:\n"
        "        report_background_failure(log, 'e', exc, scope=f'job:{org}')\n"
    )
    assert any("has no report_background_success" in p for p in _violations(source))


def test_swapped_placeholders_do_not_pair() -> None:
    source = (
        "def tick(org, project):\n"
        "    report_background_failure(log, 'e', exc, scope=f'job:{org}:{project}')\n"
        "    report_background_success(f'job:{project}:{org}')\n"
    )
    problems = _violations(source)
    assert any("has no report_background_success" in p for p in problems)
    assert any("matches no failure scope" in p for p in problems)


def test_format_and_fstring_of_the_same_shape_pair() -> None:
    source = (
        "def tick(org, project):\n"
        "    report_background_failure(\n"
        "        log, 'e', exc, scope='job:{}:{}'.format(org, project)\n"
        "    )\n"
        "    report_background_success(f'job:{org}:{project}')\n"
    )
    assert _violations(source) == []


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
    assert any("share the scope prefix 'job:'" in p for p in _violations(source))


def test_unresolvable_scope_is_reported() -> None:
    source = (
        "def tick(scope):\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        "    report_background_success(scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


def test_success_typo_is_reported() -> None:
    source = (
        "def tick():\n"
        "    report_background_failure(log, 'e', exc, scope='tagging-loop')\n"
        "    report_background_success('tagging-lop')\n"
    )
    problems = _violations(source)
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
    assert _violations(source) == []


def test_aliased_import_is_seen() -> None:
    source = (
        "from reflexio.server.background_work import (\n"
        "    report_background_failure as fail,\n"
        ")\n"
        "def tick(org):\n"
        "    fail(log, 'e', exc, scope=f'job:{org}')\n"
    )
    assert any("has no report_background_success" in p for p in _violations(source))


def test_partial_is_rejected_even_with_a_matching_success() -> None:
    """A partial's bound scope can be overridden at the call, so it is refused."""
    source = (
        "import functools\n"
        "def tick(exc):\n"
        "    fail = functools.partial(report_background_failure, scope='job:a')\n"
        "    fail(log, 'event', exc, scope='job:b')\n"
        "    report_background_success('job:a')\n"
    )
    assert any("not a direct call" in p for p in _violations(source))


def test_a_reference_that_is_not_a_call_is_reported() -> None:
    source = (
        "def tick(org):\n"
        "    handlers = [report_background_failure]\n"
        "    report_background_success(f'job:{org}')\n"
    )
    assert any("not a direct call" in p for p in _violations(source))


def test_getattr_by_name_is_reported() -> None:
    source = (
        "import reflexio.server.background_work as bw\n"
        "def tick(exc):\n"
        "    fail = getattr(bw, 'report_background_failure')\n"
        "    fail(log, 'e', exc, scope='x')\n"
    )
    assert any("named in a string" in p for p in _violations(source))


@pytest.mark.parametrize(
    "rebinding",
    [
        "    scope += ':x'\n",
        "    scope = 'job:a'\n",
        "    (scope := 'job:b')\n",
        "    for scope in ['job:c']:\n        pass\n",
        "    scope, other = 'job:d', 1\n",
    ],
    ids=["augmented", "second-assignment", "walrus", "loop-target", "unpacking"],
)
def test_a_rebound_scope_variable_is_unresolvable(rebinding: str) -> None:
    source = (
        "def tick(exc):\n"
        "    scope = 'job:a'\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        + rebinding
        + "    report_background_success(scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


def test_a_rebound_self_attribute_scope_is_unresolvable() -> None:
    source = (
        "class Beat:\n"
        "    def __init__(self, org):\n"
        "        self._scope = f'beat:{org}'\n"
        "    def run(self, exc):\n"
        "        report_background_failure(log, 'e', exc, scope=self._scope)\n"
        "        self._scope += ':x'\n"
        "        report_background_success(self._scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


@pytest.mark.parametrize(
    "rebinding",
    [
        "    from somewhere import scope\n",
        "    import scope.sub\n",
        "    match exc:\n        case str(scope):\n            pass\n",
        "    match exc:\n        case [*scope]:\n            pass\n",
        "    match exc:\n        case {**scope}:\n            pass\n",
    ],
    ids=["from-import", "dotted-import", "match-as", "match-star", "match-rest"],
)
def test_an_import_or_match_capture_rebinding_a_scope_is_unresolvable(
    rebinding: str,
) -> None:
    source = (
        "def tick(exc):\n"
        "    scope = 'job:a'\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        + rebinding
        + "    report_background_success(scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


@pytest.mark.parametrize(
    "rebinding",
    [
        "    def scope():\n        pass\n",
        "    async def scope():\n        pass\n",
        "    class scope:\n        pass\n",
        "    global scope\n",
        "    type scope = int\n",
    ],
    ids=["def", "async-def", "class", "global", "type-alias"],
)
def test_a_def_class_global_or_type_rebinding_a_scope_is_unresolvable(
    rebinding: str,
) -> None:
    source = (
        "def tick(exc):\n"
        + ("    global scope\n" if "global" in rebinding else "")
        + "    scope = 'job:a'\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        + ("" if "global" in rebinding else rebinding)
        + "    report_background_success(scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


def test_nonlocal_rebinding_a_scope_is_unresolvable() -> None:
    source = (
        "def outer(exc):\n"
        "    scope = 'job:a'\n"
        "    def tick():\n"
        "        nonlocal scope\n"
        "        scope = 'job:a'\n"
        "        report_background_failure(log, 'e', exc, scope=scope)\n"
        "        report_background_success(scope)\n"
        "    tick()\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))


def test_a_type_parameter_rebinding_a_scope_is_unresolvable() -> None:
    source = (
        "def tick(exc):\n"
        "    scope = 'job:a'\n"
        "    report_background_failure(log, 'e', exc, scope=scope)\n"
        "    def helper[scope]() -> None:\n"
        "        pass\n"
        "    report_background_success(scope)\n"
    )
    assert any("no resolvable shape" in p for p in _violations(source))
