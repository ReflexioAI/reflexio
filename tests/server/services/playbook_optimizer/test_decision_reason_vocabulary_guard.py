"""Guard: ``decision_reason`` writers must pass a fixed string, never an expression.

The invariant
-------------
``playbook_optimization_jobs.decision_reason`` is a durable ``TEXT NOT NULL``
column that is read straight back into ``PlaybookOptimizationJob`` and shown to
operators, and a Postgres trigger compares it against literal values. It is a
controlled vocabulary: every writer names a decision the optimizer made.

An expression in that keyword is how customer content gets in. ``str(exc)`` on
an arbitrary exception is the concrete case -- a pydantic ``ValidationError``
raised on a provider response renders the model's own output into its message,
so persisting the message persists evidence text. The class name, the traceback
and the tags belong in the structured-logging / error-reporting path instead,
which every failure site already uses.

Why a guard rather than review
------------------------------
The leak is invisible at the call site: ``decision_reason=str(exc)`` reads like
helpful diagnostics, produces no test failure, and only shows its teeth on the
one exception whose message happens to quote a transcript. A behavioural test
pins the one site it exercises; this scan pins the shape of every site.

Scan approach
-------------
Parse every module under ``reflexio/server/`` with :mod:`ast` and collect each
call to a job writer that passes ``decision_reason=``. The value must be a
string literal, or a conditional expression whose branches are all string
literals (the committed / winner-persisted-only site). Anything else fails.

Known blind spots (documented, not silently tolerated)
------------------------------------------------------
* Calls are matched by attribute/function NAME, so a writer reached through an
  alias or ``getattr`` is invisible. The non-vacuity test below fails if the
  scan ever stops finding the known writers, which is what would happen if the
  call shape changed wholesale.
* ``services/storage/`` is skipped. That layer does not DECIDE a reason: it
  forwards the one it was handed and hydrates the stored value back into the
  entity (``PlaybookOptimizationJob(decision_reason=row["decision_reason"])``),
  which is a read and would otherwise be a permanent false positive. The
  vocabulary is set by the services that call into it, which is what is scanned.
* Only this package is scanned. ``reflexio_ext`` calls
  ``update_playbook_optimization_job`` in exactly one place
  (``offline_tuner/open_world/runner.py``) and passes no ``decision_reason``;
  raw SQL in either package that writes the column directly is also out of
  scope -- those sites all assign literals today.
"""

from __future__ import annotations

import ast
from pathlib import Path

SERVER_ROOT = Path(__file__).resolve().parents[4] / "reflexio" / "server"

WRITER_NAMES = {
    "update_playbook_optimization_job",
    "create_playbook_optimization_job",
    "PlaybookOptimizationJob",
}


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def _is_fixed_vocabulary(value: ast.expr) -> bool:
    """True when the expression can only ever produce a fixed string."""
    if isinstance(value, ast.Constant):
        return isinstance(value.value, str)
    if isinstance(value, ast.IfExp):
        return _is_fixed_vocabulary(value.body) and _is_fixed_vocabulary(value.orelse)
    return False


def _decision_reason_writes() -> list[tuple[str, int, bool]]:
    """Collect ``(relative_path, lineno, is_fixed)`` for every writer call."""
    found: list[tuple[str, int, bool]] = []
    for path in sorted(SERVER_ROOT.rglob("*.py")):
        if "storage" in path.relative_to(SERVER_ROOT).parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _called_name(node.func) not in WRITER_NAMES:
                continue
            for keyword in node.keywords:
                if keyword.arg != "decision_reason":
                    continue
                found.append(
                    (
                        str(path.relative_to(SERVER_ROOT)),
                        keyword.value.lineno,
                        _is_fixed_vocabulary(keyword.value),
                    )
                )
    return found


def test_scan_actually_finds_the_known_writers():
    """Non-vacuity: a scan that finds nothing would pass the guard below."""
    writes = _decision_reason_writes()
    assert len(writes) >= 3, f"scan found {len(writes)} writers; it has gone blind"
    optimizer_writes = [
        w for w in writes if w[0].endswith("playbook_optimizer/optimizer.py")
    ]
    assert optimizer_writes, f"optimizer.py contributed no writers; scanned {writes}"


def test_every_decision_reason_write_is_a_fixed_string():
    dynamic = [
        (path, lineno) for path, lineno, fixed in _decision_reason_writes() if not fixed
    ]
    assert not dynamic, (
        "decision_reason must be a fixed controlled-vocabulary string, never an "
        "expression -- an exception message or other interpolated value can carry "
        f"customer content into a durable column. Offending writes: {dynamic}"
    )
