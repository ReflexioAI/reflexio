"""Input validators cannot reintroduce blocking locks behind retained run locks."""

import ast
from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[4]
    / "reflexio/server/services/storage/storage_base/_extraction_stream.py"
)
REQUIRED = {
    "_validate_completed_window_execution",
    "adopt_extraction_overflow",
    "_load_locked_extraction_overflow",
    "_consume_answer_dependencies",
    "_lock_extraction_answers",
    "_lock_historical_rows",
    "validate_agent_history_inputs",
    "consume_agent_output_dependencies",
}


def _blocking_validators(source):
    failures = []
    for method in ast.walk(ast.parse(source)):
        if not isinstance(method, ast.FunctionDef):
            continue
        if method.name not in REQUIRED and not method.name.startswith(
            "validate_extraction_"
        ):
            continue
        failures.extend(
            method.name
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "lock"
        )
    return failures


def test_all_execution_input_and_checkpoint_validators_use_the_nonwaiting_seam():
    source = SOURCE.read_text()
    names = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
    }
    assert names >= REQUIRED, "A missing/renamed guard target needs an explicit review"
    assert _blocking_validators(source) == []


def test_guard_rejects_a_new_validator_that_uses_the_owner_lock():
    assert _blocking_validators(
        "def validate_extraction_new_input(db):\n    return db.lock()\n"
    ) == ["validate_extraction_new_input"]
    assert (
        _blocking_validators(
            "def validate_extraction_new_input(db):\n    return db.validation_lock()\n"
        )
        == []
    )
