"""Pure JSON-extraction/repair helpers for structured-output parsing (Tier-2.5 leaf).

Stateless leaf module: these six functions never touched ``self`` on the former
``LiteLLMClient`` (they only called each other), so they move to a dependency-free
leaf. ``_maybe_parse_structured_output`` (StructuredOutputMixin) calls them.

Bodies are moved VERBATIM from the former ``litellm_client.py`` methods — the only
change is dropping the (unused) ``self`` parameter and rewriting the intra-cluster
``self._x(...)`` calls to direct calls. No behavior change.
"""

import json
import re

# Python-to-JSON keyword replacements used by _sanitize_json_string.
_PYTHON_TO_JSON_REPLACEMENTS = {"True": "true", "False": "false", "None": "null"}


def _extract_json_from_string(content: str) -> str:
    """
    Extract JSON from a string, handling markdown code blocks.

    Args:
        content: String potentially containing JSON.

    Returns:
        Extracted JSON string.
    """
    content = content.strip()

    # Prefer a balanced JSON container first. Structured JSON may contain
    # markdown fences inside string values; grabbing the first code block
    # would extract the inner snippet instead of the response object.
    json_container = _extract_first_json_container(content)
    if json_container is not None:
        return json_container

    # Try to extract from markdown code blocks
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    matches = re.findall(json_block_pattern, content)
    if matches:
        return matches[0].strip()

    return content


def _extract_first_json_container(content: str) -> str | None:
    """Return the first balanced JSON-like object/array in ``content``."""
    for start_idx, ch in enumerate(content):
        if ch not in "{[":
            continue
        end_idx = _find_json_container_end(content, start_idx)
        if end_idx is None:
            continue
        candidate = content[start_idx : end_idx + 1]
        if _is_parseable_json_candidate(candidate):
            return candidate
    return None


def _find_json_container_end(content: str, start_idx: int) -> int | None:
    """Find the matching end of a JSON container, respecting strings."""
    pairs = {"{": "}", "[": "]"}
    stack = [pairs[content[start_idx]]]
    in_str = False
    escape = False

    for idx in range(start_idx + 1, len(content)):
        ch = content[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in ("}", "]"):
            if not stack or stack.pop() != ch:
                return None
            if not stack:
                return idx
    return None


def _is_parseable_json_candidate(candidate: str) -> bool:
    """Return True if a balanced candidate can parse after normal sanitizing."""
    try:
        json.loads(candidate)
        return True
    except Exception:
        try:
            json.loads(_sanitize_json_string(candidate))
            return True
        except Exception:
            return False


def _looks_truncated_json(json_str: str) -> bool:
    """
    Return True when a JSON-like string appears to end before it is complete.

    This intentionally only treats content with a JSON container opener as
    truncation. Plain text that is not JSON should proceed to the normal
    parse failure path.

    Args:
        json_str: Extracted JSON-like response text.

    Returns:
        True if the response has unclosed containers or strings.
    """
    stripped = json_str.strip()
    start_indices = [
        idx for idx in (stripped.find("{"), stripped.find("[")) if idx != -1
    ]
    if not stripped or not start_indices:
        return False
    stripped = stripped[min(start_indices) :]

    stack: list[str] = []
    in_str = False
    escape = False
    pairs = {"{": "}", "[": "]"}

    for ch in stripped:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in ("}", "]") and (not stack or stack.pop() != ch):
            return False

    return in_str or bool(stack)


def _sanitize_json_string(json_str: str) -> str:
    """
    Sanitize a JSON-like string that uses Python-style syntax into valid JSON.

    Handles common LLM issues: single quotes, Python True/False/None,
    and trailing commas before closing braces/brackets.

    Args:
        json_str: A JSON-like string that may contain Python-style syntax.

    Returns:
        A sanitized string closer to valid JSON.
    """
    s = json_str

    # Walk character-by-character to:
    #   1. Replace single-quoted strings with double-quoted strings
    #   2. Replace Python True/False/None with JSON true/false/null ONLY outside strings
    #   3. Handle escaped apostrophes inside single-quoted strings (e.g. 'didn\'t')
    #   4. Escape literal double quotes that end up inside double-quoted strings
    result = []
    in_double = False
    in_single = False
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and (in_double or in_single):
            # Escaped character inside a string
            if i + 1 < len(s):
                next_ch = s[i + 1]
                if in_single and next_ch == "'":
                    # \' inside single-quoted string → literal apostrophe
                    # In JSON double-quoted strings, apostrophe needs no escape
                    result.append("'")
                    i += 2
                    continue
                result.append(ch)
                result.append(next_ch)
                i += 2
                continue
            result.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            result.append(ch)
        elif ch == "'" and not in_double:
            in_single = not in_single
            result.append('"')  # swap single → double
        else:
            # Escape unescaped double quotes inside single-quoted strings
            # (they become part of a double-quoted JSON string)
            if in_single and ch == '"':
                result.append('\\"')
            else:
                result.append(ch)
        i += 1
    s = "".join(result)

    # Replace Python booleans/None with JSON equivalents only outside quoted strings.
    # We walk the already-double-quoted result so we only need to track double quotes.
    output = []
    in_str = False
    j = 0
    while j < len(s):
        if s[j] == "\\" and in_str:
            output.append(s[j : j + 2])
            j += 2
            continue
        if s[j] == '"':
            in_str = not in_str
            output.append(s[j])
            j += 1
            continue
        if not in_str:
            matched = False
            for py_val, json_val in _PYTHON_TO_JSON_REPLACEMENTS.items():
                if s[j : j + len(py_val)] == py_val:
                    # Check word boundaries
                    before = s[j - 1] if j > 0 else " "
                    after = s[j + len(py_val)] if j + len(py_val) < len(s) else " "
                    if (
                        not before.isalnum()
                        and before != "_"
                        and not after.isalnum()
                        and after != "_"
                    ):
                        output.append(json_val)
                        j += len(py_val)
                        matched = True
                        break
            if not matched:
                output.append(s[j])
                j += 1
        else:
            output.append(s[j])
            j += 1
    s = "".join(output)

    # Remove trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", s)
